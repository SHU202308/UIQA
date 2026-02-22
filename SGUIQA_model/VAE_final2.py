from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import time
import torch
from torchvision import datasets, transforms
from pytorch_msssim import ms_ssim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split


# ==================== 超参 ====================
IMG_SIZE      = 224   # 输入分辨率
BATCH_SIZE    = 5    # 4090 建议 32；显存不足可调小
EPOCHS        = 8     # 预训练 5–10 个 epoch 已够
LR            = 2e-4  # AdamW 初始学习率
DATA_ROOT1     = r'F:\image_data\VAE\VAE_train_data\raw2'  # 数据根目录，子文件夹随便命名
DATA_ROOT2     = r'F:\image_data\VAE\VAE_train_data\UID'  # 数据根目录，子文件夹随便命名
DATA_ROOT3     = r'F:\image_data\VAE\VAE_train_data'  # 数据根目录，子文件夹随便命名
SAVE_PATH1     = r'F:\image_data\VAE\VAE_train_data\cae_UWIQA.pth' # 权重保存路径
SAVE_PATH2     = r'F:\image_data\VAE\VAE_train_data\cae_UID.pth' # 权重保存路径
SAVE_PATH3     = r'F:\image_data\VAE\VAE_train_data\cae_UWIQA.pth' # 权重保存路径


# 自编码器
class BasicBlock(nn.Sequential):
    """
    Conv-BN-ReLU
    nn.Sequential 自带 forward()，这个父类已经实现了“把输入按定义顺序依次喂给内部子模块”的 forward()，我们只是在 __init__ 里把 3 个层（Conv → BN → ReLU）按顺序塞进去。
    因此 BasicBlock(x) 本质上就是 ReLU(BN(Conv(x)))，直接调用即可，无需再手写 forward()。
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )

class ConvAlign(nn.Module):
    def __init__(self, stride_total):
        super().__init__()
        if stride_total == 1:          # I1 → 3 ch
            self.align = nn.Sequential(
                nn.Conv2d(3, 3, 3, 1, 1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True))
        elif stride_total == 2:        # I2
            self.align = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1, bias=False),  # ↓2
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        elif stride_total == 4:        # I3
            self.align = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1, bias=False),  # ↓2
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 2, 1, bias=False),  # 再 ↓2
                nn.BatchNorm2d(128), nn.ReLU(inplace=True))
    def forward(self, x): return self.align(x)


class SimpleCAE(nn.Module):
    def __init__(self, in_ch=3, base=64, latent=256):
        super().__init__()
        # ---------- Encoder ----------
        self.g1 = BasicBlock(in_ch, base, 3, 2)          # 1/2
        self.g2 = BasicBlock(base, base*2, 3, 2)         # 1/4
        self.g3 = BasicBlock(base*2, base*4, 3, 2)       # 1/8
        self.to_z = nn.Conv2d(base*4, latent, 1)         # 压到 latent dim

        # ---------- Decoder ----------
        self.up3 = nn.ConvTranspose2d(latent, base*4, 4, 2, 1)  # 1/4
        self.f3 = BasicBlock(base*4, base*2)                    # f3 (大尺度)
        self.up2 = nn.ConvTranspose2d(base*2, base*2, 4, 2, 1)  # 1/2
        self.f2 = BasicBlock(base*2, base)                      # f2
        self.up1 = nn.ConvTranspose2d(base, base, 4, 2, 1)      # 1/1
        self.f1 = BasicBlock(base, in_ch)                       # f1 (与输入同分辨率)

        # ----------ConvAlign-------------
        self.align1 = ConvAlign(1)  # 256→256   C=3
        self.align2 = ConvAlign(2)  # 256→128   C=64
        self.align3 = ConvAlign(4)  # 256→64    C=128

    def forward(self, x):
        g1 = self.g1(x)
        g2 = self.g2(g1)
        g3 = self.g3(g2)
        z  = self.to_z(g3)

        d3 = self.up3(z)
        f3 = self.f3(d3)  # 128

        d2 = self.up2(f3)
        f2 = self.f2(d2)  # 64

        d1 = self.up1(f2)
        f1 = self.f1(d1)  # 3

        I1 = self.align1(x) # 3
        I2 = self.align2(x) # 64
        I3 = self.align3(x) # 128

        # 若仅需重建，可直接输出 f1；但子网络中要把 f1-f3 作为残差对齐
        return f1, f2, f3, I1, I2, I3



# ==================== 损失函数 =================
mse = nn.MSELoss()
def recon_loss(f1 ,f2, f3, I1, I2,I3):
    # --------- MSE 多尺度 ----------
    MSE_loss = mse(f1, I1)
    MSE_loss += 0.5 * mse(f2, I2)  # 64 通道 ←→ 64 通道
    MSE_loss += 0.25 * mse(f3, I3)  # 128 通道 ←→128 通道
    # -------------- MS-SSIM 部分 -------------
    #SSIM_loss = 1. - ms_ssim(f1, I1, data_range=1, size_average=True)
    SSIM_loss = 1. - ms_ssim(f1.clamp(0, 1), I1.clamp(0, 1), data_range=1, size_average=True)
    # -------------- 总损失 -------------
    loss = MSE_loss + 0.1*SSIM_loss

    return loss

# ===============================数据集=================
train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomApply([
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip()
    ], p=0.5),
    transforms.ToTensor()
])
val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor()
])

# 基础数据集（不带 transform，便于对子集套不同变换）
base_ds = datasets.ImageFolder(DATA_ROOT2, transform=None)

# 按 9:1 划分训练/验证
N = len(base_ds)
val_size = int(0.10 * N)        # 用数据集中的10%进行验证
train_size = N - val_size
train_subset, val_subset = random_split(base_ds, [train_size, val_size])

# 给子集套各自的 transform
class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, i):
        img, _ = self.subset.dataset[self.subset.indices[i]]  # PIL, label
        img = img.convert('RGB')
        return self.transform(img), 0

train_ds = SubsetWithTransform(train_subset, train_tfm)
val_ds   = SubsetWithTransform(val_subset,   val_tfm)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

# ds      = datasets.ImageFolder(DATA_ROOT, transform=tfm)
# loader  = DataLoader(ds, batch_size=BATCH_SIZE,
#                      shuffle=True, num_workers=4, pin_memory=True)

#=================================================================================================================
@torch.no_grad()
def evaluate_val(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for x,_ in loader:
        x = x.to(device)
        f1,f2,f3,I1,I2,I3 = model(x)
        loss = recon_loss(f1,f2,f3,I1,I2,I3)
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
    return total / n


# #---------------------------------------------定义训练主函数 --------------------------------------------------------
# def main():
#     # ================== 模型 & 优化器 ==============
#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     #DEVICE = torch.device('cpu')
#     cae = SimpleCAE().to(DEVICE)
#     opt = torch.optim.Adam(cae.parameters(), lr=LR)
#     writer = SummaryWriter(log_dir=r'F:\image_data\VAE_train\cae_pretrain')
#     # ================== 训练循环 ===================
#     for ep in range(EPOCHS):
#         cae.train()
#         epoch_loss = 0
#         t0 = time.time()
#         for x, _ in loader:
#             x = x.to(DEVICE)  # RGB ∈[0,1]
#             f1, f2, f3, I1, I2, I3 = cae(x)  # 前向
#             loss = recon_loss(f1, f2, f3, I1, I2, I3)  # 计算损失
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#             epoch_loss += loss.item()
#         avg_loss = epoch_loss / len(loader)
#         print(f"[Ep {ep + 1:02d}/{EPOCHS}]  L={avg_loss:.4f}  "
#               f"t={time.time() - t0:.1f}s")
#         writer.add_scalar("Loss/train", avg_loss, ep)
#
#     torch.save(cae.state_dict(), SAVE_PATH)
#     print(f"✓ 预训练完成，权重保存为 {SAVE_PATH}")
#     writer.close()


def main():
    # ============== 超参（把固定 EPOCHS 改为“上限 20 + 早停”） ==============
    MAX_EPOCHS     = 20
    MIN_DELTA      = 0.003   # 相对改善阈值 0.3%
    PATIENCE       = 6       # 连续 6 个 epoch 改善不足则早停
    LR_PATIENCE    = 2       # Plateau 连续 2 次降 LR 后仍无明显改进则允许早停

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cae = SimpleCAE().to(DEVICE)
    opt = torch.optim.AdamW(cae.parameters(), lr=LR, weight_decay=1e-4)
    # 用验证损失驱动的自适应 lr 调度（不强制，但能更稳）
    sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2, verbose=True)

    writer = SummaryWriter(log_dir=r'F:\image_data\VAE_train\cae_pretrain')

    best_val = float('inf')
    bad_epochs = 0
    lr_drops = 0

    for ep in range(1, MAX_EPOCHS + 1):
        cae.train()
        epoch_loss, n_train = 0.0, 0
        t0 = time.time()

        for x, _ in train_loader:
            x = x.to(DEVICE)
            f1, f2, f3, I1, I2, I3 = cae(x)
            loss = recon_loss(f1, f2, f3, I1, I2, I3)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cae.parameters(), 5.0)
            opt.step()

            bs = x.size(0)
            epoch_loss += loss.item() * bs
            n_train += bs

        train_avg = epoch_loss / max(n_train, 1)

        # ---------- 验证 ----------
        val_loss = evaluate_val(cae, val_loader, DEVICE)

        # 调 scheduler（基于验证损失）
        prev_lr = opt.param_groups[0]['lr']
        sched.step(val_loss)
        new_lr = opt.param_groups[0]['lr']
        if new_lr < prev_lr:
            lr_drops += 1

        # 记录日志
        dt = time.time() - t0
        print(f"[Ep {ep:02d}/{MAX_EPOCHS}] train={train_avg:.4f}  val={val_loss:.4f}  "
              f"lr={new_lr:.2e}  t={dt:.1f}s")
        writer.add_scalar("Loss/train", train_avg, ep)
        writer.add_scalar("Loss/val", val_loss, ep)
        writer.add_scalar("LR", new_lr, ep)

        # ---------- 早停判定 ----------
        # 相对改善是否超过 min_delta（例如 0.3%）
        improved = (best_val - val_loss) > (best_val * MIN_DELTA)
        if improved or best_val == float('inf'):
            best_val = val_loss
            bad_epochs = 0
            torch.save(cae.state_dict(), SAVE_PATH2)  # 只保存验证最优
        else:
            bad_epochs += 1

        # 同时满足：耐心已用尽 或 连续多次降 lr 仍无改进  → 停
        if bad_epochs >= PATIENCE or lr_drops >= LR_PATIENCE:
            print(f"Early stopped at epoch {ep}. Best val={best_val:.4f}")
            break

    N = len(base_ds)
    print(f"Total images: {N}")
    print(f"✓ 预训练完成，最优权重已保存：{SAVE_PATH2}")
    writer.close()



if __name__ == "__main__":
    main()

# 画图
# tensorboard --logdir F:\image_data\VAE_train\cae_pretrain --port 6006