import torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from datetime import datetime
from torchvision.transforms import functional as F
import torch.nn.functional as FT
from branch_Ffn import QualityHead
from branch_Fsn import NoiseIQA_S
from branch_Ffn import NoiseIQA_F
"""
构建各个特征提取子网络
"""
# ==================== 超参 ====================
IMG_SIZE      = 224   # 输入分辨率
BATCH_SIZE    = 5    # 4090 建议 32；显存不足可调小
EPOCHS        = 5     # 预训练 5–10 个 epoch 已够
LR            = 2e-4  # AdamW 初始学习率
DATA_ROOT     = r'F:\image_data\UWIQA_train\UWIQA_test'  # 数据根目录，子文件夹随便命名
MOS_FILE     = r"F:\image_data\UWIQA_train\UWIQA_test\mos_result\mos.xlsx"
VAE_WEIGHT_PATH = r"F:\image_data\VAE\VAE_train_data\simple_cae_epoch8.pth"
SAVE_Fsn_PATH     = r'F:\image_data\UWIQA_train\Fn_epoch5.pth' # 权重保存路径
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FREEZE_EPS   = 2

# ------------------------------------ Noise branch ---------------------------------------------------------------------
class CrossFusionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # 空→频 & 频→空 双向 MHA
        self.attn_s2f = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # Gate‐FFN 维度从dim-dim*4-dim
        self.ffn_s = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )
        # LayerNorms for residuals
        self.norm_s = nn.LayerNorm(dim)

    def forward(self, Xs, Xf, alpha_s=1.0,  gamma_s=1.0):
        # Xs, Xf: (B, N, dim)
        # 空→频
        A_s, _ = self.attn_s2f(Xs, Xf, Xf)      # queries= Xs, keys/values = Xf
        Xs = self.norm_s(Xs + alpha_s * A_s)
        # Gate‐FFN (域内自适应)
        Xs = Xs + gamma_s * self.ffn_s(Xs)
        return Xs

class CrossFusionTransformer(nn.Module):
    def __init__(self, C=128, H=IMG_SIZE, W=IMG_SIZE,patch_size=32,in_channels=128,out_channels=128,
                 num_layers=4, num_heads=8, embed_dim=64):
        # 根据数据集容量，小数据集---num_layers=3, num_heads=4, embed_dim=64
        #              中数据集---num_layers=3/4, num_heads=6/4, embed_dim=96
        #              大数据集---num_layers=3/4, num_heads=8, embed_dim=128   根据效果进行尝试    若后续需要细粒度融合，在把patch_size设置16后重新考虑参数取值
        super().__init__()
        self.C, self.H, self.W = C, H, W
        self.patch_size = patch_size
        self.Hc = H// self.patch_size
        self.Wc = W// self.patch_size
        self.N = (H//self.patch_size)*(W//self.patch_size)
        self.dim = embed_dim

        # 1. Patch‐Embed：用一个大卷积＋Pointwise 投影
        #    Depthwise+Pointwise 版本，或直接 Conv2d(in→embed, k=p, s=p)
        self.patch_dw = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=self.patch_size, stride=self.patch_size,
            groups=in_channels  # Depthwise
        )
        self.patch_pw = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=1, stride=1  # Pointwise
        )

        # 2. Learnable 2D PosEmb and Domain Bias
        self.pos_emb = nn.Parameter(torch.randn(1, self.N, embed_dim))
        self.ds      = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.df      = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. Cross‐Fusion Blocks
        self.layers = nn.ModuleList([
            CrossFusionBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        # 4. Final merge (这里简单相加)
        self.final_norm = nn.LayerNorm(embed_dim)

        # 5.↓↓↓ PixelShuffle 上采样 Head ↓↓↓
        # 1×1 映射：embed_dim → embed_dim * (p*p)
        self.up_proj = nn.Conv2d(
            embed_dim,
            out_channels * (patch_size**2),
            kernel_size=1
        )
        self.upscale  = patch_size

    def forward(self, Fs, Ff, alpha_s_list=None, gamma_s_list=None):
        """
        Fs, Ff: Tensor (B, C, H, W)
        alpha_*, gamma_*: optional lists of length num_layers for per-layer weights  即融合门控系数，这里都取了1
        """
        B = Fs.shape[0]

        # --- Patch‐Embed  ---
        # Depthwise 下采 + Pointwise 投影 → (B, embed_dim, H/p, W/p)
        Ps = self.patch_dw(Fs)
        Ps = self.patch_pw(Ps)
        Pf = self.patch_dw(Ff)
        Pf = self.patch_pw(Pf)

        # --- Token 化  ---
        # (B, embed_dim, H/p, W/p) -> (B, N, embed_dim)
        Bs, Cs, Hc, Wc = Ps.shape
        Ps = Ps.flatten(2).transpose(1, 2)  # B × N × dim
        Pf = Pf.flatten(2).transpose(1, 2)

        # --- 加 位置 + 域 编码 ---
        Ps = Ps + self.pos_emb + self.ds
        Pf = Pf + self.pos_emb + self.df

        # --- Cross‐Fusion 层  ---
        for i, layer in enumerate(self.layers):
            αs = alpha_s_list[i] if alpha_s_list else 1.0
            γs = gamma_s_list[i] if gamma_s_list else 1.0
            Ps = layer(Ps, Pf, αs, γs)

        # --- 合并 & 还原特征图  ---
        P = self.final_norm(Ps)  # B × N × dim
        P = P.transpose(1, 2)  # B × dim × N
        P = P.view(B, self.dim, self.Hc, self.Wc)  # B × dim × H/p × W/p

        # （可选）上采样回原始尺寸，再接后续 Head
        x = self.up_proj(P)  # (B, dim*p², H/p, W/p)
        out = FT.pixel_shuffle(x, upscale_factor=self.upscale)
        return out


# 1.单分支运行测试所用的模型FnIQA
class FnIQA(nn.Module):
    def __init__(self,fsn_channels: int = 128, vae_weight=VAE_WEIGHT_PATH):
        super().__init__()
        self.Fsn = NoiseIQA_S(fsn_channels,vae_weight)
        self.Ffn = NoiseIQA_F()
        self.fusion = CrossFusionTransformer()
        self.regressor = QualityHead()

    def forward(self, img_RGB,img_gray):
        Fsn = self.Fsn(img_RGB)    # [B,C,H,W]
        Ffn = self.Ffn(img_gray)
        Fn = self.fusion(Fsn,Ffn)
        pred = self.regressor(Fn)
        return pred

# 2.SGUIQA运行测试所用的模型FnIQA_3
class FnIQA_3(nn.Module):
    def __init__(self,fsn_channels, vae_weight):
        super().__init__()
        self.Fsn = NoiseIQA_S(fsn_channels,vae_weight)
        self.Ffn = NoiseIQA_F()
        self.fusion = CrossFusionTransformer()

    # 透传 VAE 管理接口
    def freeze_vae(self, freeze: bool = True):
        return self.Fsn.freeze_vae(freeze)

    def vae_params(self):
        return self.Fsn.vae_params()

    def forward(self, img_RGB,img_gray):
        Fsn = self.Fsn(img_RGB)    # [B,C,H,W]
        Ffn = self.Ffn(img_gray)
        Fn = self.fusion(Fsn,Ffn)
        return Fn



class IQADataset(Dataset):
    """
    IQA 数据集读取：
      • 输入  : RGB 图像 (3, 256, 265)，数值范围 [0,1]
      • 输出  : MOS 归一化到 [0,1] 的标量张量 shape=(1,)
    """
    def __init__(self, DATA_ROOT: str, MOS_FILE: str):
        self.root_dir = Path(DATA_ROOT)
        df = pd.read_excel(MOS_FILE)

        self.samples = [
            (self.root_dir / row.NAME, float(row.MOS))
            for _, row in df.iterrows()
        ]

        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE,IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mos = self.samples[idx]
        img_RGB = self.tf(Image.open(img_path).convert("RGB"))
        img_gray = F.rgb_to_grayscale(img_RGB, num_output_channels=1)  # tensor → (1,H,W)
        mos = torch.tensor([mos], dtype=torch.float32)
        return img_RGB,img_gray, mos

#---------------------------------------训练-----------------------------------------------------------------------
# ---------- 训练一个 epoch ----------
def train_one_epoch(model, loader, criterion, optim, device):
    model.train()
    epoch_loss = 0.0
    for img_RGB,img_gray, mos in loader:
        img_RGB, img_gray, mos = img_RGB.to(device),img_gray.to(device), mos.to(device)        # shape: img[B,3,256,265]  mos[B,1]

        optim.zero_grad()
        pred  = model(img_RGB,img_gray)                               # forward
        loss  = criterion(pred, mos)                     # MSE
        loss.backward()                                  # backward
        optim.step()                                     # update

        epoch_loss += loss.item() * img_RGB.size(0)

    return epoch_loss / len(loader.dataset)

# ═══ 5. ✨ 主函数 ✨ =======================================================
def main():

    # 5-1 数据
    train_ds = IQADataset(DATA_ROOT, MOS_FILE)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)

    # 5-2 模型 / 损失 / 优化器
    model     = FnIQA().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    # 5-3 训练循环
    for ep in range(EPOCHS):

        # —— 第 FREEZE_EPS 轮时解冻 VAE —— -----------------------------
        if ep < FREEZE_EPS:
            # 这里通常什么也不用做；保险起见可再次确认
            for p in model.Fsn.backbone.VAE.parameters():
                p.requires_grad_(False)
        elif ep == FREEZE_EPS:
            for p in model.Fsn.backbone.VAE.parameters():
                p.requires_grad_(True)
            # 将新参数加入优化器；保留已有动量
            optimizer.add_param_group({'params': model.Fsn.backbone.VAE.parameters()})
            print(f"🔓  Epoch {ep}: VAE 已解冻，参与联合训练")

        loss = train_one_epoch(model, train_dl, criterion, optimizer, DEVICE)
        print(f"{datetime.now().strftime('%H:%M:%S')}  "
              f"Epoch {ep+1:02d}/{EPOCHS}  MSE = {loss:.4f}")

    # 5-4 保存
    torch.save(model.state_dict(), SAVE_Fsn_PATH)
    print("🎉  训练完成，模型权重已保存 {SAVE_Fsn_PATH}")

# ═══ 6. ✨ 入口 ✨ =========================================================
# if __name__ == "__main__":
#     main()