import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
# from torch.utils.data import Dataset
from skimage import io, color, transform
from datetime import datetime
from block_Fc import F_GAT
from block_Fc import F_AB
from block_Fc import FusionBlockRich
"""
构建各个特征提取子网络
"""
# ==================== 超参 ====================
IMG_SIZE      = 224   # 输入分辨率
BATCH_SIZE    = 2    # 4090 建议 32；显存不足可调小
EPOCHS        = 2     # 预训练 5–10 个 epoch 已够
LR            = 2e-4  # AdamW 初始学习率
DATA_ROOT     = r'F:\image_data\UWIQA_train\UWIQA_test'  # 数据根目录，子文件夹随便命名
MOS_FILE     = r"F:\image_data\UWIQA_train\UWIQA_test\mos_result\mos.xlsx"
SAVE_Fsn_PATH     = r'F:\image_data\UWIQA_train\Color_epoch5.pth' # 权重保存路径
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###测试超参数
fc_args = {
    'f_ab_args': {
        'in_channels': 2,
        'out_channels': 8,
        'kernel_size1': 3,
        'kernel_size2': 5,
        'act_layer': nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #'act_kwargs': {'negative_slope': 0.2, 'inplace': True}  # 传参
    },

    'f_gat_args': {
        'd_in': 16,
        'edge_dim': 3,
        'hidden': 16,
        'd_out':16,
        'heads':[4, 4, 4],
        'dropout':0.6,
        'use_edge_all': True
    },

    'fusion_block_args': {
        'in_c': 16,
        'kernel_size1': 5 ,
        'kernel_size2': 1 ,
        'reduction': 16
    }
}
















# ------------------------------------ Color branch ---------------------------------------------------------------------
# 1.提取色度特征
class F_Color(nn.Module):
    def __init__(self):
        super().__init__()
        self.F_CNN = F_AB()
        self.F_GAT = F_GAT()
        self.Fc = FusionBlockRich()

    def forward(self,lab_tensor, lab_norm_ts):

        # 1.为SLIC及GAT的计算进行数据准备
        LAB_images = lab_tensor.permute(0, 2, 3, 1).cpu().numpy()  # 把数据转回[B,H,W,C]，且定为numpy
        LAB_images_Fedge = lab_tensor  # tensor数据，B,C,H,W
        # 这里的lab_norm_ts是归一化的批量LAB数据，因为卷积只取AB，所以还需要只提取出AB通道
        lab_norm = lab_norm_ts[:, 1:3, :, :]

        # 2.特征计算
        F_CNN = self.F_CNN(lab_norm)
        F_GAT = self.F_GAT(LAB_images, LAB_images_Fedge, F_CNN)

        # 3.特征融合
        Fc = self.Fc(F_CNN,F_GAT)
        print("Fc.shape:",Fc.shape)

        return Fc

# 2.测试色度分支能否运行的简单预测头
class QualityHead(nn.Module):
    """
    将 Fsn(B,C,H,W) → 全局池化 → MLP → 1 维预测
    """
    def __init__(self, in_channels: int = 128, hidden: int = 4):
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

# 3.由1和2构成色度分支测试模型
class ColorIQA(nn.Module):
    """
    包装：图片 -> Fsn -> 分数
    """
    def __init__(self):
        super().__init__()
        self.backbone  = F_Color2()   # 颜色分支
        self.regressor = QualityHead()


    def forward(self, lab_tensor, lab_norm_ts):
        Fcn  = self.backbone(lab_tensor, lab_norm_ts)      # [B,C,H,W]
        pred = self.regressor(Fcn)   # [B,1]  range 0-1
        print("pred:",pred)
        return pred


# 3.2  SGUIQA运行测试所用的模型FcIQA_2
class F_Color2(nn.Module):
    #def __init__(self, f_ab_args=None, f_gat_args=None, fusion_block_args=None):
    def __init__(self, f_ab_args=fc_args['f_ab_args'], f_gat_args=fc_args['f_gat_args'], fusion_block_args=fc_args['fusion_block_args']):
        super().__init__()
        self.F_CNN = F_AB(**f_ab_args)
        self.F_GAT = F_GAT(**f_gat_args)
        self.Fc = FusionBlockRich(**fusion_block_args)

    def forward(self,lab_tensor, lab_norm_ts, shards_dir=None,keys=None ):

        # 1.为SLIC及GAT的计算进行数据准备
        LAB_images = lab_tensor.permute(0, 2, 3, 1).cpu().numpy()  # 把数据转回[B,H,W,C]，且定为numpy
        LAB_images_Fedge = lab_tensor  # tensor数据，B,C,H,W
        # 这里的lab_norm_ts是归一化的批量LAB数据，因为卷积只取AB，所以还需要只提取出AB通道
        lab_norm = lab_norm_ts[:, 1:3, :, :]

        # 2.特征计算
        F_CNN = self.F_CNN(lab_norm)
        #F_GAT = self.F_GAT(LAB_images, LAB_images_Fedge, F_CNN)  #原始利用LAB_image
        #F_GAT = self.F_GAT( LAB_images_Fedge, F_CNN, npz_paths=npz_paths)
        F_GAT = self.F_GAT(LAB_images_Fedge, F_CNN,shards_dir=shards_dir,keys=keys)

        # 3.特征融合
        Fc = self.Fc(F_CNN,F_GAT)
        #print("Fc.shape:",Fc.shape)

        return Fc
# class FcIQA_2(nn.Module):
#     def __init__(self, f_ab_args=None, f_gat_args=None,fusion_block_args=None):
#         super().__init__()
#         self.backbone  = F_Color2(f_ab_args=f_ab_args, f_gat_args=f_gat_args, fusion_block_args=fusion_block_args)   # 颜色分支
#         self.regressor = QualityHead()
#
#     def forward(self, lab_tensor, lab_norm_ts):
#         Fc  = self.backbone(lab_tensor, lab_norm_ts)      # [B,C,H,W]
#         pred = self.regressor(Fc)  # [B,1]  range 0-1
#         print("pred:", pred)
#         return pred

class FcIQA_2(nn.Module):
    def __init__(self, f_ab_args=None, f_gat_args=None,fusion_block_args=None):
        super().__init__()
        if fc_args is None:
            raise ValueError("fc_args 参数是必需的")
        self.backbone  = F_Color2(f_ab_args=f_ab_args, f_gat_args=f_gat_args, fusion_block_args=fusion_block_args)   # 颜色分支

    def forward(self, lab_tensor, lab_norm_ts, shards_dir=None,keys=None ):
        Fc  = self.backbone(lab_tensor, lab_norm_ts,shards_dir=shards_dir,keys=keys )      # [B,C,H,W]
        return Fc


# 4.数据集
class RGBLabDataset(Dataset):
    """
    读一张图 -> 返回
        rgb_tensor :   [3,H,W]  float32 归一化 (供 CNN / ViT)
        lab_tensor :   [3,H,W]  float32 L0-100, a/b-128~127 (供 SLIC 或其他 Lab 处理)
        path       :   原图路径
    """
    def __init__(self, DATA_ROOT:str,MOS_FILE:str, size=(IMG_SIZE, IMG_SIZE)):

        # 列出所有图的路径
        self.paths = sorted([
            os.path.join(DATA_ROOT, f)
            for f in os.listdir(DATA_ROOT)
            if os.path.isfile(os.path.join(DATA_ROOT, f))
        ])
        self.size = size

        # 读 MOS 表格，建立 filename -> mos 的映射
        df = pd.read_excel(MOS_FILE)  # 或者 .read_csv
        # 假设 df 有两列：NAME（图名，含后缀） 和 MOS（原始分值）
        self.mos_map = {row.NAME: float(row.MOS)
                        for _, row in df.iterrows()}


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        filename = os.path.basename(path)

        # ---------- 1. 读图 (RGB uint8, HWC) ----------
        rgb_uint8 = io.imread(path)            # skimage.io -> RGB order

        rgb = transform.resize(
            rgb_uint8,
            output_shape=(*self.size, 3),
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.uint8)  # 仍然 uint8 0-255

        # ---------- 2. 生成归一化后的 RGB Tensor ----------
        # skimage 返回 ndarray，需要转 PIL 再走 ToTensor/Normalize
        rgb_float = torch.from_numpy(rgb.astype(np.float32) / 255.0)   #   HWC
        rgb_tensor = rgb_float.permute(2, 0, 1)        # [3,H,W] float32 已归一化  CHW

        # ---------- 3. 生成 Lab float32 ----------
        # rgb_float = rgb.astype(np.float32) / 255.0      # 归一到 0-1
        lab_float = color.rgb2lab(rgb_float).astype(np.float32)  # L0-100, a/b-128~127
        lab_tensor = torch.from_numpy(lab_float).permute(2, 0, 1)  # HWC→CHW   在进行SLIC的时候需要转回HWC

        # ----------4. 生成归一化后的 LAB Tensor----------------------
        lab_norm_np = lab_float.copy()
        lab_norm_np[..., 0] = lab_norm_np[..., 0] / 50.0 - 1.0  # L: 0-100 → -1~1
        lab_norm_np[..., 1:] /= 128.0  # a,b: -1~1
        lab_norm_ts = torch.from_numpy(lab_norm_np).permute(2, 0, 1)  # CHW

        # ---------- 5. Lookup MOS，转成 Tensor ----------
        mos = self.mos_map.get(filename, None)
        if mos is None:
            raise KeyError(f"文件 {filename} 在 MOS 表里找不到对应分值")
        mos_tensor = torch.tensor([mos], dtype=torch.float32)  # shape=(1,)

        return rgb_tensor, lab_tensor, lab_norm_ts, path, mos_tensor


# ------------------------------------测试数据集读取-----------------------------------------------------------------------
#
# # 创建 Dataset 实例
# dataset = RGBLabDataset(DATA_ROOT, MOS_FILE)
# # 创建 DataLoader 实例
# dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=0)
#
# # 遍历 DataLoader 加载图像
# for i, (rgb_tensor, lab_tensor, lab_norm_ts,filenames, mos_tensor) in enumerate(dataloader):
#     if i == 1:  # 获取第二个批次的数据
#         rgb_tensor_1 = rgb_tensor
#         print('rgb_tensor_1 shape:', rgb_tensor_1.shape)
#         lab_tensor_1 = lab_tensor
#         print('lab_tensor_1 shape:', lab_tensor_1.shape)
#         lab_norm_ts_1 = lab_norm_ts
#         print('lab_norm_ts_1 shape:', lab_norm_ts_1.shape)
#         MOS = mos_tensor
#         print('MOS shape:', MOS.shape)
#         # 2) 打印这个 batch 中的所有图像路径
#         print(f"第 {i + 1} 个 batch 的图像路径：")
#         for p in filenames:
#             print("  ", p)
#         break  # 获取到第二个批次后，退出循环
#
# print("RGB数据:",rgb_tensor_1 )
# print("SLIC用的LAB数量:",lab_tensor_1 )
# print("网络用的归一化LAB数量:", lab_norm_ts_1)
# print("MOS数值:", MOS)



#---------------------------------测试Fc分支是否能运行通--------------------------------------------------------------------

# ---------- 训练一个 epoch ----------
def train_one_epoch(model, loader, criterion, optim, device):
    model.train()
    epoch_loss = 0.0
    for rgb_tensor, lab_tensor, lab_norm_ts,filenames, mos_tensor in loader:
        lab_tensor, lab_norm_ts,mos =lab_tensor.to(device), lab_norm_ts.to(device) , mos_tensor.to(device)       # shape: img[B,3,256,265]  mos[B,1]

        optim.zero_grad()
        pred  = model(lab_tensor, lab_norm_ts)                               # forward
        loss  = criterion(pred, mos)                     # MSE
        loss.backward()                                  # backward
        optim.step()                                     # update

        epoch_loss += loss.item() * lab_tensor.size(0)

    return epoch_loss / len(loader.dataset)

# ═══ 5. ✨ 主函数 ✨ =======================================================
def main():

    # 5-1 数据
    train_ds = RGBLabDataset(DATA_ROOT, MOS_FILE)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)

    # 5-2 模型 / 损失 / 优化器
    model     =ColorIQA().to(DEVICE)
    # model = FcIQA_2(f_ab_args=fc_args.get('f_ab_args'),
    #                       f_gat_args = fc_args.get('f_gat_args'),
    #                       fusion_block_args=fc_args.get('fusion_block_args')).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    # 5-3 训练循环
    for ep in range(EPOCHS):
        loss = train_one_epoch(model, train_dl, criterion, optimizer, DEVICE)
        print(f"{datetime.now().strftime('%H:%M:%S')}  "
              f"Epoch {ep+1:02d}/{EPOCHS}  MSE = {loss:.4f}")

    # 5-4 保存
    torch.save(model.state_dict(), SAVE_Fsn_PATH)
    print("🎉  训练完成，模型权重已保存 {SAVE_Fsn_PATH}")

#═══ 6. ✨ 入口 ✨ =========================================================
# if __name__ == "__main__":
#     main()