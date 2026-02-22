import torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from datetime import datetime
from VAE import SimpleCAE
from blocks1 import UncertaintyHead
from blocks1 import noise_space
from blocks1 import RFeatureFusion
"""
构建各个特征提取子网络
"""
# ==================== 超参 ====================
IMG_SIZE      = 224   # 输入分辨率
BATCH_SIZE    = 5    # 4090 建议 32；显存不足可调小
EPOCHS        = 5     # 预训练 5–10 个 epoch 已够
LR            = 2e-4  # AdamW 初始学习率
DATA_ROOT     = r'F:\image_data\UWIQA_train\UWIQA_data'  # 数据根目录，子文件夹随便命名
MOS_FILE     = r"F:\image_data\UWIQA_train\UWIQA_data\mos_result\mos.xlsx"
#VAE_WEIGHT_PATH = r"F:\image_data\VAE\VAE_train_data\simple_cae_epoch8.pth"
SAVE_Fsn_PATH     = r'F:\image_data\UWIQA_train\simple_Fsn_epoch5.pth' # 权重保存路径
FREEZE_EPS   = 2                            # VAE 冻结轮数
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------------------ Noise branch ---------------------------------------------------------------------
class NoiseExtractionBranch_S(nn.Module):
    """
        RGB → 自编码器提取可能存在噪声的残差图并压缩至64维度R → 提取U且扩展到64维 → 拼接R与U → 多尺度膨胀卷积并拼接压缩至128维 → 利用R与U生成Ms → 融合
    """
    def __init__(self):
        super().__init__()

        # ───────── ① VAE及 R1、R2、R3 ────────────────────────
        self.VAE = SimpleCAE()
        self.R = RFeatureFusion()
        # ───────── ① 计算U ────────────────────────
        self.U = UncertaintyHead()
        # ───────── ① 提取空域散斑噪声 ────────────────────────
        self.Fsn= noise_space()

    def forward(self, batch_x_RGB: torch.Tensor):
        f1, f2, f3, I1, I2, I3 = self.VAE(batch_x_RGB)
        R1 = I1 - f1
        R2 = I2 - f2
        R3 = I3 - f3
        R = self.R(R1, R2, R3)
        U = self.U(f1,batch_x_RGB)
        Fsn = self.Fsn(R,U)

        return Fsn
#--------简单的MLP预测头
class QualityHead(nn.Module):
    """
    将 Fsn(B,C,H,W) → 全局池化 → MLP → 1 维预测
    """
    def __init__(self, in_channels: int = 128, hidden: int = 32):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # B,C,1,1
            nn.Flatten(),             # B,C
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid()              # 输出 0-1，便于和归一化 MOS 对齐
        )

    def forward(self, x):
        return self.head(x)

class NoiseIQA(nn.Module):
    """
    包装：图片 -> Fsn -> 分数
    """
    def __init__(self, fsn_channels: int = 128, vae_weight: str | None = None):
    #def __init__(self, fsn_channels: int = 128, vae_weight=VAE_WEIGHT_PATH):
        super().__init__()
        self.backbone  = NoiseExtractionBranch_S()   # 你的噪声分支
        self.regressor = QualityHead(in_channels=fsn_channels)

        # ① 加载 VAE 预训练
        if vae_weight is not None:
            self.backbone.VAE.load_state_dict(torch.load(vae_weight, map_location="cpu"))
            print(f"✅ 载入 VAE 预训练权重：{vae_weight}")

        # ② 默认先冻结
        for p in self.backbone.VAE.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        fsn  = self.backbone(x)      # [B,C,H,W]
        pred = self.regressor(fsn)   # [B,1]  range 0-1
        return pred

class NoiseIQA_S(nn.Module):
    """
    包装：图片 -> Fsn -> 分数
    """
    def __init__(self, fsn_channels, vae_weight):
        super().__init__()
        self.backbone  = NoiseExtractionBranch_S()   # 你的噪声分支
        self.regressor = QualityHead(in_channels=fsn_channels)

        # ① 加载 VAE 预训练
        if vae_weight is not None:
            self.backbone.VAE.load_state_dict(torch.load(vae_weight, map_location="cpu"))
            print(f"✅ 载入 VAE 预训练权重：{vae_weight}")

        # ② 默认先冻结
        for p in self.backbone.VAE.parameters():
            p.requires_grad_(False)

        ## 后续因SGUIQA所需而加入的部分，单独测试时注释以下结构即可
        # —— 唯一可信的定位器（以后 VAE 搬家，只改这里）——
    def _get_vae(self) -> nn.Module:
        return self.backbone.VAE

    # —— 训练脚本要用的两个接口 ——
    def freeze_vae(self, freeze: bool = True):
        vae = self._get_vae()
        for p in vae.parameters():
            p.requires_grad_(not freeze)
        return vae

    def vae_params(self):
        return list(self._get_vae().parameters())
    ## 后续因SGUIQA所需而加入的部分，单独测试时注释以上结构即可

    def forward(self, x):
        fsn  = self.backbone(x)      # [B,C,H,W]
        return fsn

# ---------数据集
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
        img = self.tf(Image.open(img_path).convert("RGB"))
        mos = torch.tensor([mos], dtype=torch.float32)
        return img, mos


# ---------- 训练一个 epoch ----------
def train_one_epoch(model, loader, criterion, optim, device):
    model.train()
    epoch_loss = 0.0
    for img, mos in loader:
        img, mos = img.to(device), mos.to(device)        # shape: img[B,3,256,265]  mos[B,1]

        optim.zero_grad()
        pred  = model(img)                               # forward
        loss  = criterion(pred, mos)                     # MSE
        loss.backward()                                  # backward
        optim.step()                                     # update

        epoch_loss += loss.item() * img.size(0)

    return epoch_loss / len(loader.dataset)

# ═══ 5. ✨ 主函数 ✨ =======================================================
def main():

    # 5-1 数据
    train_ds = IQADataset(DATA_ROOT, MOS_FILE)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)

    # 5-2 模型 / 损失 / 优化器
    model     = NoiseIQA().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    # 5-3 训练循环
    for ep in range(EPOCHS):

        # —— 第 FREEZE_EPS 轮时解冻 VAE —— -----------------------------
        if ep == FREEZE_EPS:
            for p in model.backbone.VAE.parameters():
                p.requires_grad_(True)
            # 将新参数加入优化器；保留已有动量
            optimizer.add_param_group({'params': model.backbone.VAE.parameters()})
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










