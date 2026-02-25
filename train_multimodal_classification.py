# train_multimodal_classification.py

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from collections import Counter
from multimodal_regression_model import MultiModalRegressionModel  # 用现成模型，只替换最后 mlp 层



from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # 放在最前面屏蔽所有 RDKit 控制台 warning


# ========== 超参数 ==========
DATA_PATH = "data/BBBP-C.csv"  # 包含 SMILES 和 label 的 CSV
SMILES_COLUMN = "smiles"
LABEL_COLUMN = "p_np"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 2e-4
PATIENCE = 5

SPE_VOCAB_PATH = "models/vocab-spe.pkl"
AWD_VOCAB_PATH = "models/vocab-awd.pkl"
AWD_MODEL_PATH = "models/smiles_encoder.pth"
DMPNN_MODEL_PATH = "models/best_encoder.pth"

SAVE_DIR = "classification_checkpoints"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


os.makedirs(SAVE_DIR, exist_ok=True)

# ========== 数据集类 ==========
class SmilesClassificationDataset(Dataset):
    def __init__(self, csv_path, smiles_col="SMILES", label_col="Label"):
        df = pd.read_csv(csv_path)
        self.smiles = df[smiles_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()


    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


# ========== 损失函数：带权 BCE ==========
def get_pos_weight(subset):
    """
    subset: torch.utils.data.Subset 包含 train_set
    """
    # subset.dataset 是原始 Dataset，subset.indices 是样本索引
    labels = [subset.dataset.labels[i] if isinstance(subset.dataset.labels[i], (int, float)) 
              else subset.dataset.labels[i].item() 
              for i in subset.indices]

    counter = Counter(labels)
    neg, pos = counter.get(0, 0), counter.get(1, 0)

    if pos == 0:
        print("⚠️ Warning: No positive samples in training set. pos_weight set to 1.0")
        pos_weight = 1.0
    else:
        pos_weight = neg / pos

    return torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)


# ========== 修改后的模型 ==========
class MultiModalClassifier(MultiModalRegressionModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(  # 重新定义为二分类输出
            nn.Linear(kwargs.get("fusion_dim", 256), kwargs.get("mlp_hidden_dim", 128)),
            nn.ReLU(),
            nn.Dropout(kwargs.get("dropout", 0.1)),
            nn.Linear(kwargs.get("mlp_hidden_dim", 128), 1)  # 单神经元 + BCE
        )

    def forward(self, smiles_list):
        return super().forward(smiles_list)


# ========== 指标计算 ==========
def compute_metrics(y_true, y_pred):
    from sklearn.metrics import roc_auc_score

    y_pred_cls = (y_pred > 0.5).astype(int)
    report = classification_report(y_true, y_pred_cls, digits=4, output_dict=False)
    print(report)
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred_cls))
    try:
        auc = roc_auc_score(y_true, y_pred)
        print(f"ROC-AUC: {auc:.4f}")
    except ValueError:
        print("⚠️ ROC-AUC 无法计算（可能是验证集中某一类数量为 0）")



# ========== 训练主函数 ==========
def train():
    print("📦 Loading dataset...")
    dataset = SmilesClassificationDataset(DATA_PATH, SMILES_COLUMN, LABEL_COLUMN)

    # 提取标签
    labels = [label for _, label in dataset]
    # Stratified 分层划分
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(split.split(X=labels, y=labels))
    # 构造子集
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    pos_weight = get_pos_weight(train_set)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    print("🧠 Building model...")
    model = MultiModalClassifier(
        spe_vocab_path=SPE_VOCAB_PATH,
        awd_vocab_path=AWD_VOCAB_PATH,
        awd_model_path=AWD_MODEL_PATH,
        dmpnn_model_path=DMPNN_MODEL_PATH,
        device=DEVICE
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    no_improve_epochs = 0

    print("🚀 Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for smiles, label in train_loader:
            smiles = list(smiles)
            label = label.to(DEVICE).unsqueeze(1)
            logits = model(smiles)
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for smiles, label in val_loader:
                smiles = list(smiles)
                label = label.to(DEVICE).unsqueeze(1)
                logits = model(smiles)
                val_logits.append(torch.sigmoid(logits))
                val_labels.append(label)

        val_preds = torch.cat(val_logits, dim=0).flatten().cpu().numpy()
        val_targets = torch.cat(val_labels, dim=0).flatten().cpu().numpy()
        val_loss = loss_fn(torch.tensor(val_preds), torch.tensor(val_targets)).item()

        print(f"[Epoch {epoch:02d}] TrainLoss={sum(train_losses)/len(train_losses):.4f} | "
              f"ValLoss={val_loss:.4f}")
        compute_metrics(val_targets, val_preds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_classifier.pth"))
            print("✅ Best model saved.")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                print("🛑 Early stopping.")
                break

    print("🎉 Training complete.")


if __name__ == "__main__":
    train()
