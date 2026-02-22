import torch
import torch.nn as nn
import numpy as np
from branch_FL_LTEST import FlIQA_1
from branch_Fc1 import FcIQA_2
from branch_Fn import FnIQA_3
from branch_Fusion import Swin_CSA_FusionIQA_Hard4
from torch.utils.data import Dataset, DataLoader
from skimage import io, color, transform
import os,json
import pandas as pd
import torchvision.transforms as T
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from dataclasses import dataclass




# ==================== 超参 ====================
IMG_SIZE      = 224   # 输入分辨率
BATCH_SIZE    = 2    # 4090 建议 32；显存不足可调小
EPOCHS        = 4     # 预训练 5–10 个 epoch 已够
#LR            = 2e-4  # AdamW 初始学习率
DATA_ROOT     = r'F:\image_data\UWIQA_train\UWIQA_test'  # 数据根目录，子文件夹随便命名
#NPZ_ROOR = r'F:\image_data\UWIQA_train\UWIQA_test_NPZ'
shards_dir=r'F:\image_data\UWIQA_train\UWIQA_test_FENPIAN'
Npy_file_path = r'F:\image_data\UWIQA_train\UWIQA_test\UWIQA_test1_L.npy'
MOS_FILE     = r"F:\image_data\UWIQA_train\UWIQA_test\mos_result\mos.xlsx"
VAE_WEIGHT_PATH = r"F:\image_data\VAE\VAE_train_data\simple_cae_epoch8.pth"
SWIN_WEIGHT_PATH = r"F:\pycharm_practice\SGUIQA\Swin_Transform_model\swin_tiny_patch4_window7_224.pth"
OUT_XLSX = r"F:\image_data\underwater_image_data\HUAFEN\split_UWIQA_test1.xlsx"
SAVE_SGUIQA_PATH     = r'F:\image_data\UWIQA_train\SGUIQA4.pth' # 权重保存路径

DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FREEZE_EPS_VAE   = 1
FREEZE_EPS_SWIN   = 2

BASE_LR   = 2e-4
SWIN_LR   = 2e-5
WD_HEAD   = 0.01
WD_SWIN   = 0.05
VAE_LR    = 1e-4   # 例子：VAE 的 LR 比 head 小一点
WD_VAE    = 0.01

# ---------------- logistic 拟合（为 RMSE/SRCC 对齐分布，常见做法） ----------------
def logistic_5p(x, b1, b2, b3, b4, b5):
    z = np.clip(b2 * (x - b3), -60, 60)
    return b1 * (0.5 - 1.0 / (1 + np.exp(z))) + b4 * x + b5

def fit_logistic(y_pred, y_true):
    try:
        popt, _ = curve_fit(
            logistic_5p, y_pred, y_true,
            bounds=([-1e3, -10, -1e3, -1e3, -1e3], [1e3, 10, 1e3, 1e3, 1e3]),
            maxfev=10000
        )
        return logistic_5p(y_pred, *popt)
    except Exception:
        return y_pred


# =============================================各分支参数===========================================================
#=== 1.亮度分支
fl_args = {
    'mid_channels': 32,
    'feat_channels_down': 128,
    'feat_channels_up': 96,
    'feat_channels': 128,
    'reflect_ch': 32,
    'heads': 4,
    'win_patch': 8,
}

#=== 2.色度分支
fc_args = {
    'f_ab_args': {
        'in_channels': 2,
        'out_channels': 8,
        'kernel_size1': 3,
        'kernel_size2': 5,
        'act_layer': nn.LeakyReLU(negative_slope=0.2, inplace=True)
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

fusion_args = {
    "which_backbone": "small",          # 'small'
    "swin_ckpt": SWIN_WEIGHT_PATH,                  # Swin下载好的参数地址
    "freeze_mode": "stage34",           # 'frozen' | 'none' | 'stage34'
    "fusion_channels": 128,             # 三路特征通道数
    "beta_hidden": 16,                  # β-MLP 隐层
    "reg_hidden": 64,                   # 回归头 MLP 隐层
    "csa_refine": True,                 # CSA_TDP 是否细化
    "auto_imagenet_norm": True,         # 是否给 Swin 分支做 IN 归一化
}


#=======================  L1、2正则化  =============================

def l1_regularization(model, l1_lambda=1e-4) -> torch.Tensor:
    l1_loss = 0
    # 遍历模型的所有参数
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))  # 对每个参数计算 L1 范数
    return l1_lambda * l1_loss  # 乘以正则化系数


def l2_regularization(model, l2_lambda=1e-4):
    l2_loss = 0.0
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.sum(param ** 2)
    return l2_lambda * l2_loss


#---------------------------------------  Dataset  -----------------------------------------------------------------

# ---------------- 读取 MOS 表（支持 xlsx/csv/txt） ----------------
def load_mos_table(mos_path, name_col='NAME', mos_col='MOS'):
    ext = os.path.splitext(mos_path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(mos_path, engine='openpyxl')
    elif ext in ['.csv']:
        df = pd.read_csv(mos_path)
    else:
        # 假设 txt 格式类似：image_name mos（空格/逗号分隔）
        df = pd.read_csv(mos_path, sep=None, engine='python', header=None, names=[name_col, mos_col])

    df.columns = [c.strip() for c in df.columns]
    assert name_col in df.columns and mos_col in df.columns, \
        f'MOS 文件必须包含列: {name_col} 和 {mos_col}'

    df = df[[name_col, mos_col]].copy()
    df[name_col] = df[name_col].astype(str).str.strip()
    df[mos_col]  = df[mos_col].astype(float)
    return df.rename(columns={name_col: 'NAME', mos_col: 'MOS'})

## ==== 对数据集进行划分并导出excel表格 ====
def export_split_excel(
    data_root: str,
    mos_file: str,
    out_xlsx: str,
    name_col: str = 'NAME',   # 如果你的表头是 ImageName 就改成 'ImageName'
    mos_col: str  = 'MOS',
    test_ratio: float = 0.2,
    seed: int = 42,
    recursive: bool = True,
):
    """
    读取 MOS → 和 data_root 取交集 → 划分 train/val → 导出 Excel
    返回: mos_map_train, mos_map_val, split_df, stats_df, missing_df
    """
    # 1) 读 MOS 表（会把列名统一成 NAME / MOS）
    df = load_mos_table(mos_file, name_col=name_col, mos_col=mos_col)  # 你已有的函数

    # 2) 枚举磁盘上所有文件（basename -> 绝对路径）
    all_paths = []
    if recursive:
        for d, _, fs in os.walk(data_root):
            for f in fs:
                all_paths.append(os.path.join(d, f))
    else:
        all_paths = [os.path.join(data_root, f)
                     for f in os.listdir(data_root)
                     if os.path.isfile(os.path.join(data_root, f))]
    basename2path = {os.path.basename(p): p for p in all_paths}

    # 3) 和磁盘取交集；记录缺失
    df['abs_path'] = df['NAME'].map(basename2path)
    missing_df = df[df['abs_path'].isna()].copy()
    df_exist   = df[df['abs_path'].notna()].reset_index(drop=True)
    if df_exist.empty:
        raise RuntimeError("在数据目录中没有找到与 MOS 表匹配的图片，请检查文件名/后缀是否一致。")

    # 4) 划分
    names = df_exist['NAME'].values
    mos   = df_exist['MOS'].values
    n_train, n_val, y_train, y_val = train_test_split(
        names, mos, test_size=test_ratio, random_state=seed, shuffle=True
    )

    # 5) 组合 split 表 & 统计
    train_df = pd.DataFrame({'NAME': n_train, 'MOS': y_train, 'split': 'train'})
    val_df   = pd.DataFrame({'NAME': n_val,   'MOS': y_val,   'split': 'val'})
    split_df = pd.concat([train_df, val_df], ignore_index=True)
    split_df['abs_path'] = split_df['NAME'].map(basename2path)

    stats_df = split_df.groupby('split')['MOS'].agg(['count','mean','std']).reset_index()

    # 6) 导出 Excel
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
        split_df.to_excel(w, sheet_name='split', index=False)
        stats_df.to_excel(w, sheet_name='stats', index=False)
        if not missing_df.empty:
            missing_df.to_excel(w, sheet_name='missing_files', index=False)

    print(f"✅ 划分完成并导出：{out_xlsx}")
    print(stats_df)

    # 7) 返回两个映射字典 + 三个 DataFrame（可选）
    mos_map_train = dict(zip(n_train, y_train))
    mos_map_val   = dict(zip(n_val,   y_val))
    return mos_map_train, mos_map_val, split_df, stats_df, missing_df

class IQADataset2(Dataset):
    """
    读一张图 -> 返回 6 个量：
      rgb_tensor  : [3,H,W] float32, 0~1
      lab_tensor  : [3,H,W] float32, L:0~100, a/b:-128~127
      lab_norm_ts : [3,H,W] float32, L∈[-1,1], a/b∈[-1,1]
      gray_tensor : [1,H,W] float32, 0~1
      path        : str
      mos_tensor  : [1] float32
    """
    def __init__(self, data_root: str, mos_map: dict, size=(IMG_SIZE, IMG_SIZE), recursive: bool = True,
                 shards_dir: str | None = None, basename_to_key: dict[str, str] | None = None,npy_file_path: str = None):
        self.size = size
        self.mos_map = mos_map  # filename -> MOS
        self.shards_dir = shards_dir
        self.basename_to_key = basename_to_key
        self.npy_file_path = npy_file_path

        # 1) 枚举图片路径：递归 or 单层
        all_paths = []
        if recursive:
            for d, _, files in os.walk(data_root):
                for f in files:
                    all_paths.append(os.path.join(d, f))
        else:
            all_paths = [
                os.path.join(data_root, f)
                for f in os.listdir(data_root)
                if os.path.isfile(os.path.join(data_root, f))
            ]

        # 2) 仅保留 MOS 表中存在的文件（按 basename 对齐）
        mos_names = set(self.mos_map.keys())
        self.paths = [p for p in sorted(all_paths) if os.path.basename(p) in mos_names]

        if len(self.paths) == 0:
            raise RuntimeError("在数据目录中没有找到与 MOS 表匹配的图片。请检查文件名是否一致（含后缀）。")

        # 加载npy文件
        if self.npy_file_path:
            self.l_norm_dict = np.load(self.npy_file_path, allow_pickle=True).item()
        else:
            self.l_norm_dict = {}

        # 3) 若给了 shards_dir，则准备 basename->key 解析器
        self._basename2key = None
        if self.shards_dir is not None:
            index_path = os.path.join(self.shards_dir, "index.json")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"未找到 index.json：{index_path}")
            with open(index_path, "r", encoding="utf-8") as f:
                key2shard = json.load(f)["key2shard"]

            if self.basename_to_key is not None:
                    # 用户自定义映射优先
                self._basename2key = dict(self.basename_to_key)
            else:
                # 自动从 index.json 构造 basename -> key（要求唯一）
                tmp = {}
                dup = {}
                for k in key2shard.keys():
                    b = os.path.basename(k)
                    if b in tmp and tmp[b] != k:
                        dup.setdefault(b, set()).update({tmp[b], k})
                    else:
                        tmp[b] = k
                if dup:
                     # 有重名，提示用户传入 basename_to_key
                    msgs = [f"{b}: {sorted(list(v))}" for b, v in dup.items()]
                    raise ValueError(
                        "index.json 中存在重名 basename，无法唯一确定 key。"
                        "请传入 basename_to_key 参数来消除歧义。\n" + "\n".join(msgs)
                    )
                self._basename2key = tmp  # 现在是一一对应：basename -> key

    def __len__(self):
        return len(self.paths)

    def _fname_to_key(self, fname: str) -> str | None:
        if self.shards_dir is None:
            return None
        # 依据 basename 找 key（默认把后缀换成 .npz，再去 _basename2key 查）
        base_npz = os.path.splitext(fname)[0] + ".npz"
        if self._basename2key is None:
            return None
        key = self._basename2key.get(base_npz)
        if key is None:
            # 有时分片里的 key 不是 .npz 后缀（极少见），退一步直接用原 basename 查
            key = self._basename2key.get(fname)
        return key

    def __getitem__(self, idx):
        path = self.paths[idx]
        fname = os.path.basename(path)

        # ---------- 1) 读图 (H,W,3) uint8 ----------
        rgb_uint8 = io.imread(path)  # skimage 读出的就是 RGB
        # 若是灰图，升维成 3 通道
        if rgb_uint8.ndim == 2:
            rgb_uint8 = np.stack([rgb_uint8]*3, axis=-1)
        elif rgb_uint8.shape[-1] == 4:  # RGBA → RGB
            rgb_uint8 = rgb_uint8[..., :3]

        # ---------- 2) resize 并归一化到 0~1（numpy，HWC）----------
        rgb_resized = transform.resize(
            rgb_uint8, output_shape=(*self.size, 3),
            preserve_range=True, anti_aliasing=True
        ).astype(np.float32) / 255.0                      # HWC, float32, 0~1

        # ---------- 3) 转 torch RGB（CHW, 0~1） & 灰度 ----------
        rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1)      # [3,H,W]
        gray_tensor = T.functional.rgb_to_grayscale(rgb_tensor, num_output_channels=1)  # [1,H,W]

        # ---------- 4) 生成 Lab（注意：rgb2lab 期望 numpy、且输入范围[0,1]）----------
        lab_np = color.rgb2lab(rgb_resized).astype(np.float32)           # HWC
        lab_tensor = torch.from_numpy(lab_np).permute(2, 0, 1)           # [3,H,W]

        # ---------- 5) 归一化后的 Lab ----------
        lab_norm_np = lab_np.copy()
        lab_norm_np[..., 0] = lab_norm_np[..., 0] / 50.0 - 1.0   # L: 0~100 → [-1,1]
        lab_norm_np[..., 1:] /= 128.0                            # a/b: [-128,127] → ~[-1,1]
        lab_norm_ts = torch.from_numpy(lab_norm_np).permute(2, 0, 1)

        # ---------- 6) MOS ----------
        mos = self.mos_map.get(fname)
        if mos is None:
            raise KeyError(f"文件 {fname} 在 MOS 表里找不到分值")
        mos_tensor = torch.tensor([mos], dtype=torch.float32)

        # 7) 生成 key（用于从分片读取图结构）
        key = self._fname_to_key(fname)  # 若没给 shards_dir，则为 None
        if self.shards_dir is not None and key is None:
            # 只有设置了 shards_dir 才强制要求有 key
            raise KeyError(
                f"[IQADataset2] 无法为 {fname} 找到分片 key。\n"
                f"请确认 {os.path.join(self.shards_dir, 'index.json')} 中存在"
                f" {os.path.splitext(fname)[0] + '.npz'} 对应的键，或传入 basename_to_key。"
            )

        # ---------- 8) 获取对应图像的亮度异常度 L_norm ----------
        L_norm = self.l_norm_dict.get(fname, None)  # 使用文件名从字典中获取对应的亮度异常度
        print(L_norm.keys())
        if L_norm is None:
            raise KeyError(f"L_norm data not found for image {fname}")
        #L_norm = L_norm.astype(np.float32, copy=False)
        # 将 L_norm 数据扩展为 [1,1, H, W]
        if L_norm.ndim == 2:
            L_norm_tensor = L_norm.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        return rgb_tensor, lab_tensor, lab_norm_ts, gray_tensor, path, mos_tensor, L_norm_tensor, key





#-------------------------------------   SGUIQA model   --------------------------------------------------------------
class SGUIQA(nn.Module):
    def __init__(self, fl_args=None, fc_args=None,fusion_args=None,shards_dir=shards_dir):
        super().__init__()
        self.shards_dir = shards_dir

        # 1.亮度分支：如果 fl_args 为 None，将抛出错误，强制开发人员提供该参数
        if fl_args is None:
            raise ValueError("fl_args 参数是必需的")
        self.FL = FlIQA_1(fl_args=fl_args)

        # 2.色度分支，fc_args 也不使用 `or {}`，强制检查
        if fc_args is None:
            raise ValueError("fc_args 参数是必需的")
        self.FC = FcIQA_2(f_ab_args=fc_args.get('f_ab_args'),
                          f_gat_args=fc_args.get('f_gat_args'),
                          fusion_block_args=fc_args.get('fusion_block_args'))


        self.FN = FnIQA_3(fsn_channels=128, vae_weight=VAE_WEIGHT_PATH)
        #self.Fusion = Swin_CSA_FusionIQA_Hard()
        if fusion_args is None:
            raise ValueError("fusion_args 参数是必需的")  # 用默认
        else:
            self.Fusion = Swin_CSA_FusionIQA_Hard4(**fusion_args)  # 用你的配置

    def freeze_vae(self, freeze: bool = True):
        return self.FN.freeze_vae(freeze)

    def vae_params(self):
        return self.FN.vae_params()

    # 便捷转发：训练时直接 model.set_swin_freeze_mode('stage4') 调用
    def set_swin_freeze_mode(self, mode: str):
        self.Fusion.set_swin_freeze_mode(mode)

    # 便捷：返回当前可训练参数（重建优化器用）
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, lab_tensor, lab_norm_ts, rgb_tensor, gray_tensor,L_norm_tensor,keys=None,return_aux: bool = False ):
        #FL = self.FL(lab_tensor, lab_norm_ts)

        if return_aux:
            FL_out = self.FL(lab_tensor, lab_norm_ts, return_aux=True)
            FL, fl_aux = FL_out  # fl_aux 里应包含 {'A_ref','A_pred', ...}
        else:
            FL = self.FL(lab_tensor, lab_norm_ts, return_aux=False)
            fl_aux = {}

        FC = self.FC(lab_tensor, lab_norm_ts,L_norm_tensor,shards_dir=self.shards_dir,keys=keys)
        FN = self.FN(rgb_tensor, gray_tensor)
        F_fusion, beta, score, S = self.Fusion(rgb_tensor, FL, FC, FN)
        print("score:", score)

        # if return_aux:
        #     return score, {"beta": beta, "S": S}  # 训练用
        if return_aux:
            aux = {"beta": beta, "S": S}
            aux.update(fl_aux)  # 合并亮度分支的 {'A_ref','A_pred',...}
            return score, aux

        return score  # 验证/推理用


#----------------------------------------- 训练 ------------------------------------------------------------------------

#==== 构建优化器（顶层定义）====
def build_optimizer(model):
    head_params, swin_params, vae_params = [], [], []

    # 先拿到一份 VAE 的参数集合（对象级别）
    vae_param_set = set(model.vae_params())  # 需要你在 SGUIQA 里实现 .vae_params()

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if "Fusion.swin" in n:
            swin_params.append(p)
        elif p in vae_param_set:
            vae_params.append(p)
        else:
            head_params.append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": BASE_LR, "weight_decay": WD_HEAD})
    if swin_params:
        groups.append({"params": swin_params, "lr": SWIN_LR, "weight_decay": WD_SWIN})
    if vae_params:
        groups.append({"params": vae_params, "lr": VAE_LR, "weight_decay": WD_VAE})

    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


# ====  构建四项评价指标 ====
@torch.no_grad()
def evaluate(model, loader, device, use_logistic=True):
    """
    返回: srcc, plcc, krcc, rmse
    - srcc/krcc: 用原始预测
    - plcc/rmse: 可选择是否 logistic 映射（默认 True）
    """
    model.eval()
    y_pred, y_true = [], []

    for rgb_tensor, lab_tensor, lab_norm_ts, gray_tensor, filenames, mos_tensor, keys, L_norm_tensor in loader:
        lab_tensor   = lab_tensor.to(device)
        lab_norm_ts  = lab_norm_ts.to(device)
        rgb_tensor   = rgb_tensor.to(device)
        gray_tensor  = gray_tensor.to(device)
        L_norm_tensor = L_norm_tensor.to(device)
        mos          = mos_tensor.to(device)

        pred = model(lab_tensor, lab_norm_ts, rgb_tensor, gray_tensor,L_norm_tensor,keys=keys, return_aux=False)  # [B,1] or [B]
        pred = pred.squeeze().detach().cpu().numpy()
        mos  = mos.squeeze().detach().cpu().numpy()

        y_pred.append(pred)
        y_true.append(mos)

    # 拼接为一维
    y_pred = np.concatenate(y_pred, axis=0).astype(np.float64)
    y_true = np.concatenate(y_true, axis=0).astype(np.float64)

    # 去除 NaN / Inf
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred, y_true = y_pred[mask], y_true[mask]

    if y_true.size < 2:
        raise RuntimeError("验证样本过少，无法计算相关系数。")

    # SRCC / KRCC：原始预测
    srcc = spearmanr(y_pred, y_true)[0]
    krcc = kendalltau(y_pred, y_true)[0]

    # PLCC / RMSE：是否做 logistic 单调映射
    if use_logistic:
        y_pred_m = fit_logistic(y_pred, y_true)
    else:
        y_pred_m = y_pred

    plcc = pearsonr(y_pred_m, y_true)[0]
    rmse = np.sqrt(mean_squared_error(y_pred_m, y_true))

    return srcc, plcc, krcc, rmse

# ============ 构建损失函数 ===============
@dataclass
class IqaLossCfg:
    delta: float = 1.0          # Huber/ SmoothL1 的 delta
    rank_eps: float = 1e-3      # 排序时跳过 |yi - yj| < eps 的近似相等对
    rank_max_pairs: int = 4096  # 排序子采样上限
    lambda_beta: float = 1e-3   # β 稀疏正则系数
    lambda_tv: float = 1e-4     # S 的 TV 正则系数
    init_s1: float = 0.0        # s1 初值（0 => 初始权重=1）
    init_s2: float = 0.0        # s2 初值

class CombinedIqaLoss(nn.Module):
    """
    L_total = exp(-s1)*L_huber + exp(-s2)*L_rank + (s1+s2)
              + lambda_beta*R_beta + lambda_tv*R_tv
    需要前向提供: pred(score) [B], mos[B], 以及可选 beta[B,C,H,W] 或 [B,K], S[B,1,H,W]
    """
    def __init__(self, cfg: IqaLossCfg = IqaLossCfg()):
        super().__init__()
        self.cfg = cfg
        # 可学习的不确定性参数
        self.s1 = nn.Parameter(torch.tensor(cfg.init_s1, dtype=torch.float32))
        self.s2 = nn.Parameter(torch.tensor(cfg.init_s2, dtype=torch.float32))

    # --- 组件：Huber（等价 SmoothL1Loss(beta=delta)） ---
    def huber_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # PyTorch 自带: nn.SmoothL1Loss(beta=delta, reduction='mean')
        return F.smooth_l1_loss(pred, target, beta=self.cfg.delta, reduction='mean')

    # --- 组件：pairwise 排序损失（softplus(log(1+exp()))）---
    #@torch.cuda.amp.autocast(enabled=False)
    def pairwise_rank_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred, y_true: [B] (float)
        L_rank = mean_{(i,j)} softplus( - s_ij * (ŷ_i - ŷ_j) ),
        其中 s_ij = sign(y_i - y_j), 跳过 |y_i - y_j| < eps，且最多采样 rank_max_pairs 对。
        """
        y_pred = y_pred.float()
        y_true = y_true.float()
        B = y_true.shape[0]
        if B < 2:
            return y_pred.new_tensor(0.0)

        # 构造上三角索引 (i<j)
        idx_i, idx_j = torch.triu_indices(B, B, offset=1)
        dy_true = (y_true[idx_i] - y_true[idx_j])
        mask = dy_true.abs() >= self.cfg.rank_eps
        if mask.sum() == 0:
            return y_pred.new_tensor(0.0)

        idx_i = idx_i[mask]
        idx_j = idx_j[mask]
        # 子采样
        if idx_i.numel() > self.cfg.rank_max_pairs:
            perm = torch.randperm(idx_i.numel(), device=idx_i.device)[: self.cfg.rank_max_pairs]
            idx_i = idx_i[perm]
            idx_j = idx_j[perm]
            dy_true = dy_true[mask][perm]

        sij = torch.sign(dy_true)                      # +1 / -1
        margin = (y_pred[idx_i] - y_pred[idx_j])      # 预测差
        # softplus(- s_ij * margin)
        loss_ij = F.softplus(-sij * margin)           # = log(1+exp(.))
        return loss_ij.mean()

    # --- 组件：β 稀疏正则（L1） ---
    def beta_reg(self, beta: torch.Tensor | None) -> torch.Tensor:
        if beta is None:
            return self.s1.new_tensor(0.0)
        return beta.abs().mean()

    # --- 组件：S 的总变差（各向同性 L1 TV） ---
    def tv_reg(self, S: torch.Tensor | None) -> torch.Tensor:
        if S is None:
            return self.s1.new_tensor(0.0)
        # 允许 S 为 [B,1,H,W] 或 [B,H,W]
        if S.dim() == 3:
            S = S.unsqueeze(1)
        dx = S[:, :, :, :-1] - S[:, :, :, 1:]
        dy = S[:, :, :-1, :] - S[:, :, 1:, :]
        return (dx.abs().mean() + dy.abs().mean())

    def forward(
        self,
        pred: torch.Tensor,       # [B] or [B,1]
        mos: torch.Tensor,        # [B] or [B,1]
        beta: torch.Tensor | None = None,
        S: torch.Tensor | None = None,
    ):
        pred = pred.view(-1)
        mos  = mos.view(-1)

        L_huber = self.huber_loss(pred, mos)
        L_rank  = self.pairwise_rank_loss(pred, mos)

        # 不确定性加权
        w1 = torch.exp(-self.s1)
        w2 = torch.exp(-self.s2)
        multi_task = w1 * L_huber + w2 * L_rank + (self.s1 + self.s2)

        # 正则
        R_beta = self.beta_reg(beta)
        R_tv   = self.tv_reg(S)

        loss = multi_task + self.cfg.lambda_beta * R_beta + self.cfg.lambda_tv * R_tv

        # 便于日志记录
        components = {
            "loss": loss,
            "L_huber": L_huber.detach(),
            "L_rank":  L_rank.detach(),
            "R_beta":  R_beta.detach(),
            "R_tv":    R_tv.detach(),
            "s1":      self.s1.detach(),
            "s2":      self.s2.detach(),
            "w1":      w1.detach(),
            "w2":      w2.detach(),
        }
        return loss, components

# ============================================== 亮度分支正则化 ==================================================
def tv_loss(x: torch.Tensor) -> torch.Tensor:
    # L1 各向同性 TV
    dx = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dy = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dx + dy

def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + eps)

@dataclass
class LuminanceRegCfg:
    # 分支内三项配比（你要的：TV=1.0, ZM=0.3, ALIGN=1.0）
    w_tv: float = 1.0
    w_zm: float = 0.3
    w_align: float = 1.0
    # 亮度正则总门控（外层系数）
    gamma_lumi: float = 0.10
    # 是否使用自适应目标占比（可选：让 gamma_lumi 维持约 10%）
    use_target_ratio: bool = False
    target_ratio: float = 0.10
    ratio_smooth_pow: float = 0.1   # 慢调幂次（0.05~0.2都可）

class LuminanceReg(nn.Module):
    """
    亮度分支正则（独立模块）：
      L_lumi = w_tv*TV(A_pred) + w_zm*mean(A_pred)^2 + w_align*ALIGN
      ALIGN：优先使用 alpha（[B,H]）；若无，则退化用 beta 中“亮度分支权重”。
    最后在训练脚本里： L_total = L_main + gamma_lumi * L_lumi
    """
    def __init__(self, cfg: LuminanceRegCfg = LuminanceRegCfg()):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def _global_from_A(self, A_pred: torch.Tensor) -> torch.Tensor:
        # A_pred 的全局强度（均值），做 z-score；detach 以免“把先验拉歪”
        g = A_pred.detach().mean(dim=(1, 2, 3))  # [B]
        return zscore(g)

    @staticmethod
    def _alpha_reduce(alpha: torch.Tensor) -> torch.Tensor:
        """
        alpha: [B,H] or [B,1] —— 每头/全局标量门控
        统一压成 [B]（对 head 取均值），再做 z-score
        """
        if alpha.dim() == 2:
            a = alpha.mean(dim=1)   # [B]
        elif alpha.dim() == 1:
            a = alpha
        else:
            # [B,1] 或其他情况
            a = alpha.view(alpha.size(0), -1).mean(dim=1)
        return zscore(a)

    @staticmethod
    def _extract_beta_lumi(beta: torch.Tensor) -> torch.Tensor:
        """
        从 beta 中取“亮度分支”的全局权重，形状压成 [B]。
        你需要按自己的 Fusion 输出格式实现这一函数。
        下面是一个常见的占位实现（假设 beta[B,K]，第0维是亮度）：
        """
        if beta.dim() == 2:
            beta_L = beta[:, 0]  # [B] —— 假设第0列是亮度分支
        else:
            # 例如 beta[B,C,H,W]，你可以改成：先空域均值 -> 通道索引
            beta_L = beta[:, 0].mean(dim=(1, 2))  # 占位写法
        return zscore(beta_L)

    def forward(
        self,
        A_pred: torch.Tensor,              # [B,1,H,W]，校准后的亮度异常图（参与反传）
        alpha: torch.Tensor | None = None, # [B,H]（若有）
        beta:  torch.Tensor | None = None, # 融合层权重（若无 alpha，则用它做全局对齐）
    ):
        # --- 形态项 ---
        L_tv = tv_loss(A_pred)
        L_zm = (A_pred.mean())**2

        # --- “使用一致性”：全局对齐（优先 alpha，其次 beta）
        if alpha is not None:
            a_n = self._alpha_reduce(alpha)       # [B] z-score
            g_n = self._global_from_A(A_pred)     # [B] z-score (detach)
            L_align = ((a_n - g_n)**2).mean()
        elif beta is not None:
            b_n = self._extract_beta_lumi(beta)   # [B] z-score
            g_n = self._global_from_A(A_pred)     # [B] z-score (detach)
            L_align = ((b_n - g_n)**2).mean()
        else:
            L_align = A_pred.new_tensor(0.0)

        # 分支内配比
        L_lumi = (self.cfg.w_tv    * L_tv +
                  self.cfg.w_zm    * L_zm +
                  self.cfg.w_align * L_align)

        # 总门控（可自适应目标占比）
        gamma = self.cfg.gamma_lumi
        if self.cfg.use_target_ratio:
            # 需要外部提供 L_main（见下文示例），这里仅返回 L_lumi 与一个闭包式的缩放器
            return L_lumi, {'L_tv': L_tv.detach(), 'L_zm': L_zm.detach(),
                            'L_align': L_align.detach(), 'gamma': gamma}
        else:
            L_reg = gamma * L_lumi
            logs  = {'L_tv': L_tv.detach(), 'L_zm': L_zm.detach(),
                     'L_align': L_align.detach(), 'L_lumi': L_lumi.detach(),
                     'gamma': torch.tensor(gamma, device=A_pred.device)}
            return L_reg, logs

# ========= 训练一次 =========
def train_once(
    model: nn.Module,
    device,
    train_loader,
    val_loader,
    num_epochs: int = EPOCHS,
    save_best_path: str | None = SAVE_SGUIQA_PATH,
    unfreeze_plan: dict[int, str] | None = None,   # 例如 {100: 'stage4', 160: 'stage34'}
    use_logistic_eval: bool = True,                # 验证时 PLCC/RMSE 是否做 5p logistic
    loss_cfg: IqaLossCfg = IqaLossCfg(),           # 我们定义的损失配置
    use_amp: bool = True,                          # 混合精度（显存友好）
    grad_clip_norm: float | None = 1.0,            # 梯度裁剪，可设 None 关闭
    vae_unfreeze_at: int | None = None,
    swin_init_mode = "frozen"
):
    """
    每个 epoch：训练一次 -> 在 val 上评估（SRCC/PLCC/KRCC/RMSE）
    可选：按 unfreeze_plan 动态解冻 Swin 并重建优化器
    以 SRCC 最优为准保存 best 权重
    """
    model.to(device)

    if swin_init_mode is not None:
        model.set_swin_freeze_mode(swin_init_mode)  # 'frozen' / 'stage4' / 'stage34' / 'none'

    if vae_unfreeze_at is not None:
        model.freeze_vae(True)

    # 1) 主损失初始化（原有）
    criterion = CombinedIqaLoss(loss_cfg).to(device)
    # 2) 亮度分支的正则化
    lumi_reg_cfg = LuminanceRegCfg(
        w_tv=1.0, w_zm=0.3, w_align=1.0,
        gamma_lumi=0.10,  # 固定总门控；想自适应就把 use_target_ratio=True
        use_target_ratio=False,
        target_ratio=0.10,
        ratio_smooth_pow=0.1
    )
    lumi_reg = LuminanceReg(lumi_reg_cfg).to(device)

    optimizer = build_optimizer(model)

    use_amp = use_amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    #scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history   = []
    best_srcc = -1.0
    best_path = None
    best_ep   = -1

    for ep in range(1, num_epochs + 1):
        # --- (A) Swin 的阶段性解冻----
        if unfreeze_plan and ep in unfreeze_plan:
            mode = unfreeze_plan[ep]                     # 'stage4' / 'stage34' / 'none' ...
            model.set_swin_freeze_mode(mode)
            optimizer = build_optimizer(model)           # 可训练集变化，重建优化器
            print(f"🔓 Epoch {ep}: set Swin freeze mode -> {mode} (rebuilt optimizer)")

        # --- (B) VAE 的解冻（本次新增）----
        if (vae_unfreeze_at is not None) and (ep == vae_unfreeze_at):
            model.freeze_vae(False)  # 解冻 VAE
            optimizer = build_optimizer(model)  # 现在会多一个 VAE 的 param group（带你的专属 lr/wd）
            print(f"🔓 Epoch {ep}: unfreeze VAE (rebuilt optimizer)")

        # --- 训练 ---
        model.train()
        running = 0.0
        seen    = 0

        for rgb, lab, lab_norm, gray, paths, mos, L_norm_tensor,keys  in train_loader:
            rgb, lab, lab_norm, gray = rgb.to(device), lab.to(device), lab_norm.to(device), gray.to(device)
            mos = mos.to(device).view(-1)                # [B]

            optimizer.zero_grad(set_to_none=True)
            # with torch.cuda.amp.autocast(enabled=use_amp):
            #     # ⬅️ 训练时拿到 score + {beta, S}
            #     pred, aux = model(lab, lab_norm, rgb, gray, return_aux=True)
            #     pred = pred.view(-1)                     # [B]
            #     loss = criterion(pred, mos, aux=aux)     # 计算 L_total

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred, aux = model(lab, lab_norm, rgb, gray, L_norm_tensor,keys=keys, return_aux=True)
                pred = pred.view(-1)
                beta = aux.get("beta") if isinstance(aux, dict) else None
                S = aux.get("S") if isinstance(aux, dict) else None
                A_pred = aux.get("A_pred")  # 亮度分支需透出
                alpha = aux.get("alpha")  # 若你把 Q_wight 的输出也透出了

                # 1）主损失+融合模块的正则化
                loss_main, logs_main = criterion(pred, mos, beta=beta, S=S)  # 注意损失返回 (loss, logs)
                # 2）先取未乘 gamma 的亮度分支正则化（模块会返回 L_lumi 和日志）
                L_lumi_raw, logs_lumi = lumi_reg(A_pred=A_pred, alpha=alpha, beta=beta)
                # 自适应调整 gamma_lumi（保持亮度正则约占主损失 10%）
                with torch.no_grad():
                    ratio = (lumi_reg_cfg.gamma_lumi * L_lumi_raw.detach()) / (loss_main.detach() + 1e-8)
                    lumi_reg_cfg.gamma_lumi = float(
                        torch.clamp(
                            torch.tensor(lumi_reg_cfg.gamma_lumi) *
                            ((lumi_reg_cfg.target_ratio / (ratio + 1e-8)) ** lumi_reg_cfg.ratio_smooth_pow),
                            1e-3, 10.0
                        )
                    )

                # L2 正则化（色度分支的 L2 正则化项）
                FC_loss = l2_regularization(model.FC, l2_lambda=1e-4)  # L2 正则化，λ=1e-4
                # L1正则化 （噪声分支的L1正则化，把空域噪声和频域噪声分别进行）
                Fsn_loss = l1_regularization(model.FN.Fsn)
                Ffn_loss = l1_regularization(model.FN.Ffn)

                loss = loss_main + lumi_reg_cfg.gamma_lumi * L_lumi_raw+FC_loss+Fsn_loss+Ffn_loss



            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            bs = lab.size(0)
            running += loss.item() * bs
            seen    += bs

        train_loss = running / max(seen, 1)

        # --- 验证 ---
        srcc, plcc, krcc, rmse = evaluate(model, val_loader, device, use_logistic=use_logistic_eval)
        print(f"[Epoch {ep:03d}/{num_epochs}] "
              f"train_loss={train_loss:.4f}  SRCC={srcc:.4f}  PLCC={plcc:.4f}  KRCC={krcc:.4f}  RMSE={rmse:.4f}")

        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "srcc": srcc, "plcc": plcc, "krcc": krcc, "rmse": rmse
        })

        # --- 保存最优（按 SRCC）---
        if srcc is not None and srcc > best_srcc:
            best_srcc = srcc
            best_ep   = ep
            if save_best_path:
                torch.save(model.state_dict(), save_best_path)
                best_path = save_best_path
                print(f"💾  Save best (SRCC {best_srcc:.4f}) -> {best_path}")

    best = {"best_epoch": best_ep, "best_srcc": best_srcc, "best_path": best_path}
    return history, best


def main(
    test_ratio: float = 0.5,
    seed: int = 42,                 # 固定 seed -> 固定划分
    do_train: bool = True,
    runs: int = 2,                 # 连续训练次数
    num_workers: int = 4,
    epochs_per_run: int = 4      # 每次训练轮数
):
    # === 1) 固定划分（只做一次）并导出 Excel ===
    mos_map_train, mos_map_val, split_df, stats_df, missing_df = export_split_excel(
        data_root=DATA_ROOT,
        mos_file=MOS_FILE,
        out_xlsx=OUT_XLSX,
        name_col='NAME',       # 你的表头是 ImageName（非常重要）
        mos_col='MOS',
        test_ratio=test_ratio,
        seed=seed,                  # 固定 seed -> 固定 train/val 列表
        recursive=True
    )
    print("\n=== 固定划分统计 ===")
    print(stats_df)

    # === 2) 用同一划分构建一次 Dataset/DataLoader（后面 10 次都复用） ===
    train_ds = IQADataset2(DATA_ROOT, mos_map_train, size=(IMG_SIZE, IMG_SIZE),shards_dir=shards_dir,npy_file_path =Npy_file_path )
    val_ds   = IQADataset2(DATA_ROOT, mos_map_val,   size=(IMG_SIZE, IMG_SIZE),shards_dir=shards_dir,npy_file_path =Npy_file_path )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    # 只看划分就退出
    if not do_train:
        print("\n仅做一次固定划分+导表，未启动训练。")
        return

    # === 3) 连续训练 runs 次（每次 1000 轮），划分不变 ===
    results = []
    for run in range(1, runs + 1):
        print(f"\n\n================= RUN {run}/{runs} =================")

        # （可选）给权重初始化一个不同的随机种子，但划分不变
        torch.manual_seed(1000 + run)
        np.random.seed(1000 + run)

        # 3.1 构建模型
        model = SGUIQA(fl_args=fl_args, fc_args=fc_args, fusion_args=fusion_args).to(DEVICE)

        # 3.2 设置 Swin/VAE 的冻结-解冻策略（和你之前一致）
        unfreeze_plan   = {FREEZE_EPS_SWIN: "stage4"}   # 需要更细粒度可继续加键值
        vae_unfreeze_at = FREEZE_EPS_VAE               # 不想解冻就设 None
        save_path_this  = SAVE_SGUIQA_PATH.replace(".pth", f"_run{run}.pth")

        # 3.3 开始训练（一次训练 = 多个 epoch）
        history, best = train_once(
            model=model,
            device=DEVICE,
            train_loader=train_dl,
            val_loader=val_dl,
            num_epochs=epochs_per_run,          # 每次训练 1000 轮
            save_best_path=save_path_this,
            unfreeze_plan=unfreeze_plan,
            use_logistic_eval=True,
            loss_cfg=IqaLossCfg(),              # 需要改超参就传 IqaLossCfg(...)

            # 下面两项按需
            use_amp=True,
            grad_clip_norm=1.0,

            # 冻结/解冻
            vae_unfreeze_at=vae_unfreeze_at,
            swin_init_mode="frozen"             # 开局完全冻结 Swin；也可换 "stage4"/"stage34"/"none"
        )

        print(f"✅ RUN {run} 完成：best SRCC={best['best_srcc']:.4f} @ epoch {best['best_epoch']}, weight={best['best_path']}")
        results.append({"run": run, **best})

    # （可选）汇总 10 次最佳结果
    try:
        import pandas as pd
        df_res = pd.DataFrame(results)
        print("\n=== 10 次训练最佳结果汇总 ===")
        print(df_res)
        # 想保存成表格就解除注释：
        # df_res.to_excel(OUT_XLSX.replace('.xlsx', '_train10_summary.xlsx'), index=False)
    except Exception:
        pass

    if len(results) > 0:
        overall = max(results, key=lambda x: x["best_srcc"])
        print(f"\n🌟 Overall best across {len(results)} runs: "
              f"SRCC={overall['best_srcc']:.4f} @ run={overall['run']}, epoch={overall['best_epoch']}")
        print(f"   → weight file: {overall['best_path']}")
    else:
        print("No results to summarize.")


# ═══ 6. ✨ 入口 ✨ =========================================================
if __name__ == "__main__":
    main()

