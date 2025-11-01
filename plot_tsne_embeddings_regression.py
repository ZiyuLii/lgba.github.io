import os
import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from multimodal_regression_model import MultiModalRegressionModel

# ---------- 配置 -------------
DATA_PATH = "data/Lipophilicity.csv"     # 跟你训练时的路径保持一致
SMILES_COLUMN = "SMILES"          # 跟训练保持一致
LABEL_COLUMN = "pIC50"             # 回归目标列名，确保是连续值

SPE_VOCAB_PATH = "models/vocab-spe.pkl"
AWD_VOCAB_PATH = "models/vocab-awd.pkl"
AWD_MODEL_PATH = "models/smiles_encoder.pth"
DMPNN_MODEL_PATH = "models/best_encoder.pth"

MODEL_CKPT_PATH = "regression_checkpoints/Lipophilicity_best_model.pth"  # 训练好回归模型的权重路径
DEVICE = torch.device("cpu")

N_SAMPLES = None   # 可选：t-SNE时随机抽样数量，太大降维慢（建议<2000）
os.makedirs("regression_point", exist_ok=True)

# ---------- 加载数据 -------------
df = pd.read_csv(DATA_PATH)
smiles_list = df[SMILES_COLUMN].astype(str).tolist()
targets = df[LABEL_COLUMN].astype(float).tolist()  # 回归目标是连续值
if N_SAMPLES is not None and len(smiles_list) > N_SAMPLES:
    idx = np.random.choice(len(smiles_list), N_SAMPLES, replace=False)
    smiles_list = [smiles_list[i] for i in idx]
    targets = [targets[i] for i in idx]

# ---------- 加载模型 -------------
def get_embedding_model(trained=True):
    model = MultiModalRegressionModel(
        spe_vocab_path=SPE_VOCAB_PATH,
        awd_vocab_path=AWD_VOCAB_PATH,
        awd_model_path=AWD_MODEL_PATH,
        dmpnn_model_path=DMPNN_MODEL_PATH,
        device=DEVICE
    ).to(DEVICE)
    if trained and os.path.exists(MODEL_CKPT_PATH):
        state_dict = torch.load(MODEL_CKPT_PATH, map_location=DEVICE)
        state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(state_dict, strict=False)
        print("✅ 加载已训练回归模型参数")
    else:
        print("❗使用未训练初始模型参数")
    model.eval()
    return model

# --------- 获取融合embedding ---------
def extract_embeddings(model, smiles_list, batch_size=32):
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            awd_vec = model.awd_encoder(batch)
            dmpnn_vec = model.dmpnn_encoder(batch)
            fused = model.fusion(awd_vec, dmpnn_vec)
            all_embeds.append(fused.cpu().numpy())
    return np.concatenate(all_embeds, axis=0)

# --------- 主流程 ----------
model_init = get_embedding_model(trained=False)
emb_init = extract_embeddings(model_init, smiles_list)

model_trained = get_embedding_model(trained=True)
emb_trained = extract_embeddings(model_trained, smiles_list)

# --------- t-SNE降维 ----------
tsne = TSNE(n_components=2, random_state=42)
emb_init_2d = tsne.fit_transform(emb_init)
emb_trained_2d = tsne.fit_transform(emb_trained)
targets = np.array(targets)

# --------- 绘图 ----------
def plot_tsne_regression(emb_2d, targets, title, fname):
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=targets, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Target value')
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"保存图像: {fname}")

plot_tsne_regression(emb_init_2d, targets, "初始多模态embedding回归分布（未训练）", "tsne_initial_regression.png")
plot_tsne_regression(emb_trained_2d, targets, "训练后多模态embedding回归分布", "tsne_trained_regression.png")

def save_tsne_points(emb_2d, targets, fname):
    df_points = pd.DataFrame({
        "x": emb_2d[:, 0],
        "y": emb_2d[:, 1],
        "target": targets
    })
    df_points.to_csv(fname, index=False)
    print(f"保存点坐标: {fname}")

save_tsne_points(emb_init_2d, targets, "regression_point/tsne_initial_points.csv")
save_tsne_points(emb_trained_2d, targets, "regression_point/tsne_trained_points.csv")

print("全部完成，可查看tsne_initial_regression.png和tsne_trained_regression.png")