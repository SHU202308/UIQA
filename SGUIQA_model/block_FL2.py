import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
#from torchvision.transforms import functional as F
import torch.nn.functional as FT
from typing import Optional
from typing import Dict
from L_new_anomaly import Luminance_anomaly
from typing import Tuple




"""
构建各个特征提取子网络
"""
# ==================== 超参 ====================
IMG_SIZE      = 256   # 输入分辨率
BATCH_SIZE    = 5    # 4090 建议 32；显存不足可调小
EPOCHS        = 5     # 预训练 5–10 个 epoch 已够
LR            = 2e-4  # AdamW 初始学习率
DATA_ROOT     = r'F:\image_data\UWIQA_train\UWIQA_data'  # 数据根目录，子文件夹随便命名
MOS_FILE     = r"F:\image_data\UWIQA_train\UWIQA_data\mos_result\mos.xlsx"
SAVE_Fsn_PATH     = r'F:\image_data\UWIQA_train\Fn_epoch5.pth' # 权重保存路径
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FREEZE_EPS   = 2

# ------------------------------------ Luminance branch ---------------------------------------------------------------------

# 1. LFE
class LFE(nn.Module):
    """
    对输入特征进行多尺度并行卷积且进行1*1卷积融合
    """
    def __init__(self,in_channels,mid_channels,out_channels):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(in_channels, mid_channels, kernel_size=5, stride=1,padding=2)
        self.conv7 = nn.Conv2d(in_channels, mid_channels, kernel_size=7, stride=1,padding=3)
        self.conv1 = nn.Conv2d(3*mid_channels, out_channels, kernel_size=1,stride=1,padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.conv3(x)
        f2 = self.conv5(x)
        f3 = self.conv7(x)
        f = torch.cat([f1, f2, f3], dim=1)
        f_out = self.conv1(f)
        return f_out

# 2. 求解图像整体亮度异常情况，以便更新Q
class Q_wight(nn.Module):
    def __init__(self, in_channels:int, out_channels:int,heads:int,leak: float = 0.1):
        '''
        nn.LeakyReLU(negative_slope: float = 0.1, inplace: bool = False) 默认是0.01，但是因为
        在极亮/极暗的亮度分支里，负输入较多；把它调到 0.1 能保留更多负向梯度，避免死神经元——这正是之前建议改用 LeakyReLU(0.1) 的原因。
        '''
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(leak, inplace=True),
        )

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # GAP 对每个通道进行池化   变为B*C*1*1
            nn.Flatten(1),                    # 去掉 1×1 维度，只保 (B, C)
            nn.Linear(out_channels, out_channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 2, heads, bias=False),
            nn.Sigmoid()
        )

    def forward(self,L,A):
        x = torch.cat([L, A] , dim=1)
        feat = self.backbone(x)
        alpha = self.mlp(feat)  # (B, heads)
        return alpha

# 3.基于Retinex理论，利用卷积得到I和R
class RetinexDecompose(nn.Module):
    """
    输入  L  (B,1,H,W)
    输出 Ĩ  (B,1,H,W)   &   R̂  (B,1,H,W)
    """
    def __init__(self, eps: float = 1e-3,
                 use_dilation: bool = True):
        super().__init__()
        # depth-wise 3×3 近似 Retinex 平滑
        if use_dilation:
            self.conv_i = nn.Conv2d(
                1, 1, kernel_size=3, padding=5,
                dilation=5, groups=1, bias=False)
        else:
            self.conv_i = nn.Conv2d(
                1, 1, kernel_size=1, bias=False)  # 1×1 也可
        self.act = nn.Sigmoid()
        self.eps = eps

    def forward(self, L: torch.Tensor):
        I = self.act(self.conv_i(L))           # 照度
        R = L / (I + self.eps)                # 反射率
        return I, R


class ReflectanceEmbed(nn.Module):
    """
    把单通道 R̂ 嵌入到 C_R 维，方便与亮度纹理拼接
    """
    def __init__(self, C_R: int ):
        super().__init__()
        self.embed = nn.Conv2d(1, C_R, 1, bias=False)

    def forward(self, R):
        return self.embed(R)                  # (B,C_R,H,W)

# # 5. 多头交叉注意力机制
# class LuminanceAttention(nn.Module):
#     """
#     Multi-Head Attention for the luminance branch
#       Q  ← from tensor A  (B, C, H, W)
#       K,V← from tensor B  (B, C, H, W)
#       α  ← (B, 1) or (B, heads)
#     """
#     def __init__(self, channelsQ: int , channelsKV: int ,out_channels:int,heads: int ):
#         super().__init__()
#         assert out_channels % heads == 0, "C must be divisible by heads"
#         self.C      = out_channels
#         self.H      = heads
#         self.d_k    = out_channels // heads
#
#         self.w_q = nn.Linear(channelsQ, out_channels, bias=False)
#         self.w_k = nn.Linear(channelsKV, out_channels, bias=False)
#         self.w_v = nn.Linear(channelsKV, out_channels, bias=False)
#
#         self.scale = self.d_k ** -0.5     # 1/sqrt(d_k)
#
#     def _tokenise(self, x: torch.Tensor) -> torch.Tensor:
#         # 把数据展开 (B, C, H, W) -> (B, N, C)
#         B, C, H, W = x.shape
#         N = H * W
#         return x.flatten(2).transpose(1, 2)   # (B, N, C)
#
#     def forward(
#         self,
#         F_LAR: torch.Tensor,        # (B, C, H, W)  -- Query source
#         F_L: torch.Tensor,        # (B, C, H, W)  -- Key/Value source
#         alpha: Optional[torch.Tensor] = None  # (B, 1) or (B, heads)
#     ) -> torch.Tensor:
#         """
#         return F_L : (B, C, H, W)
#         """
#         Bsz, C, H, W = F_LAR.shape
#         N = H * W
#
#         # 1. Tokenise
#         q_tokens = self._tokenise(F_LAR)          # (B, N, C)
#         kv_tokens= self._tokenise(F_L)          # (B, N, C)
#
#         # 2. Linear projections
#         Q = self.w_q(q_tokens)                # (B, N, C)
#         K = self.w_k(kv_tokens)
#         V = self.w_v(kv_tokens)
#
#         # 3. Split heads  -> (B, heads, N, d_k)
#         def split_heads(x):
#             return x.view(Bsz, N, self.H, self.d_k).permute(0, 2, 1, 3)
#         Q = split_heads(Q)
#         K = split_heads(K)
#         V = split_heads(V)
#
#         # 4. Gate α  (broadcast到 (B, heads, 1, 1))
#         if alpha is None:
#             alpha = torch.full((Bsz, self.H), 0.5, device=F_LAR.device)  # 默认 0.5
#         if alpha.dim() == 2:                             # 若 alpha 已是 (B, H) 向量，给末尾 再加两个维度 → (B, H, 1, 1)，这样可与 Q/K 的形状 (B, H, N, d) 逐元素相乘
#             alpha = alpha.unsqueeze(-1).unsqueeze(-1)                # (B,H,1,1)
#         elif alpha.dim() == 1:                                       # 标量 α
#             alpha = alpha.view(Bsz, 1, 1, 1)
#         else:
#             raise ValueError("alpha shape must be (B,) or (B,H)")
#
#         Q_adj = alpha * Q + (1.0 - alpha) * K                        # (B,H,N,d)
#
#         # 5. Scaled Dot-Product Attention
#         attn = torch.matmul(Q_adj, K.transpose(-2, -1)) * self.scale # (B,H,N,N)
#         attn = FT.softmax(attn, dim=-1)
#         out  = torch.matmul(attn, V)                                 # (B,H,N,d)
#
#         # 6. Merge heads
#         out = out.permute(0, 2, 1, 3).contiguous()  # (B,N,H,d)
#         out = out.view(Bsz, N, self.C)              # (B,N,C)
#
#         # 7. Reshape back to (B,C,H,W)
#         out = out.transpose(1, 2).view(Bsz, C, H, W)
#         return out

#--------------------------窗口化的多头交叉注意力-----------------------------------------------------------------------
# ---------- ① 助手函数：窗口切分 / 还原 ----------
def window_partition(x: torch.Tensor, win: int) -> torch.Tensor:
    """
    (B,C,H,W) -> (B*n_win, C, win, win)
    n_win = (H//win)*(W//win)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // win, win, W // win, win)          # (B,C,Hp,win,Wp,win)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()            # (B,Hp,Wp,C,win,win)
    return x.view(-1, C, win, win)                          # (B*n_win,C,win,win)


def window_reverse(x_win: torch.Tensor, H: int, W: int,
                   win: int, C: int) -> torch.Tensor:
    """
    (B*n_win, C, win, win) -> (B,C,H,W)
    """
    Bn = x_win.size(0)
    B = Bn // (H // win * W // win)
    x = x_win.view(B, H // win, W // win, C, win, win)      # (B,Hp,Wp,C,win,win)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()            # (B,C,Hp,win,Wp,win)
    return x.view(B, C, H, W)                               # (B,C,H,W)

# ---------- ② 完整的窗口 Multi-Head 注意力 ----------
class LuminanceAttention(nn.Module):
    """
    窗口化(MSA)的多头交叉注意力：
      Q ←  F_LAR  (B,C,H,W)  —— Query 源
      K,V← F_L    (B,C,H,W)  —— Key/Value 源
      α ←  (B,1) or (B,H)    —— 门控权重
    """
    def __init__(self,
                 channelsQ: int,
                 channelsKV: int,
                 out_channels: int,
                 heads: int,
                 win_patch: int = 8):
        super().__init__()
        assert out_channels % heads == 0, "out_channels 必须能被 heads 整除"
        self.C   = out_channels
        self.H   = heads
        self.d_k = out_channels // heads
        self.win_patch = win_patch                       # 窗口大小 8 或 16

        self.w_q = nn.Linear(channelsQ,  out_channels, bias=False)
        self.w_k = nn.Linear(channelsKV, out_channels, bias=False)
        self.w_v = nn.Linear(channelsKV, out_channels, bias=False)

        self.scale = self.d_k ** -0.5         # 缩放因子

    # ------ img⇄token 变换 ------
    @staticmethod
    def _tokenise(x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)   # (B, N, C)

    # ------ 主 forward ------
    def forward(self,
                F_LAR: torch.Tensor,
                F_L:   torch.Tensor,
                alpha: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        """
        输入/输出尺寸均为 (B,C,H,W)
        """
        B, C, H, W = F_LAR.shape
        win = self.win_patch
        assert H % win == 0 and W % win == 0, "H,W 必须被窗口大小整除"

        # 1) 分窗口 → (B*n_win, C, win, win)
        q_patch = window_partition(F_LAR, win)
        k_patch = window_partition(F_L,   win)

        # 如果 alpha=(B,H)，需 broadcast 到每个窗口
        if alpha is not None and alpha.dim() == 2:
            repeats = q_patch.size(0) // B
            alpha   = alpha.repeat_interleave(repeats, dim=0)  # (B*n_win, heads)

        # 2) 窗口内视作“迷你 batch”做原生注意力
        Bn, _, _, _ = q_patch.shape
        Nw = win * win                                     # 每窗口 token 数

        def tok(x):  # (Bn,C,win,win)->(Bn,Nw,C)
            return x.flatten(2).transpose(1, 2)

        Q = self.w_q(tok(q_patch))
        K = self.w_k(tok(k_patch))
        V = self.w_v(tok(k_patch))

        # split heads: (Bn,heads,Nw,d_k)
        def split(x):
            return x.view(Bn, Nw, self.H, self.d_k).permute(0, 2, 1, 3)

        Q = split(Q); K = split(K); V = split(V)

        # 3) α 门控
        if alpha is None:
            alpha = torch.full((Bn, self.H), 0.5, device=F_LAR.device)
        if alpha.dim() == 2:
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)         # (Bn,H,1,1)
        Q_adj = alpha * Q + (1.0 - alpha) * K                 # (Bn,H,Nw,d)

        # 4) Scaled Dot-Product
        attn = torch.matmul(Q_adj, K.transpose(-2, -1)) * self.scale  # (Bn,H,Nw,Nw)
        attn = FT.softmax(attn, dim=-1)
        out  = torch.matmul(attn, V)                           # (Bn,H,Nw,d_k)

        # 5) 合并 heads→(Bn,Nw,C)
        out = out.permute(0, 2, 1, 3).contiguous().view(Bn, Nw, self.C)

        # 6) token→image 还原窗口 (Bn,C,win,win)
        out_patch = out.transpose(1, 2).view(Bn, self.C, win, win)

        # 7) 窗口逆变换 → (B,C,H,W)
        out_full = window_reverse(out_patch, H, W, win, self.C)
        return out_full

class LumiFeat(nn.Module):
    """
    LumiFeat: 亮度异常度 & 去材质照度批量计算器

    输入：
        batch_lab_raw: torch.Tensor, shape (B,3,H,W),
                       原始 Lab (L in [0,100], a,b in [-128,127])
    输出：
        L_perp_batch      : torch.Tensor, shape (B,1,H,W)
        L_anom_norm_batch : torch.Tensor, shape (B,1,H,W)
    """
    def __init__(self,
                 win:   int   = 15,
                 bins:  int   = 64,
                 alpha: float = 0.9,
                 lam1:  float = 0.5,
                 lam2:  float = 0.5):
        super().__init__()
        self.win   = win
        self.bins  = bins
        self.alpha = alpha
        self.lam1  = lam1
        self.lam2  = lam2

        # 明确只取这两个键
        self._want = ['L_perp', 'L_anom_norm']

    @torch.no_grad()
    def forward(self,
                batch_lab_raw: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = batch_lab_raw.shape

        Lp_list      = []
        Lanom_list   = []

        for i in range(B):
            # (3,H,W) -> (H,W,3) numpy
            lab_np = batch_lab_raw[i].permute(1,2,0).cpu().numpy()
            # 逐图调用你的 NumPy 版亮度异常度计算
            res_np = Luminance_anomaly(
                        img_path=None,
                        LAB_raw = lab_np,
                        win     = self.win,
                        bins    = self.bins,
                        alpha   = self.alpha,
                        lam1    = self.lam1,
                        lam2    = self.lam2)
            # 提取并转回 (1,H,W) Tensor
            Lp_list.append(
                torch.from_numpy(res_np['L_perp']).unsqueeze(0))
            Lanom_list.append(
                torch.from_numpy(res_np['L_anom_norm']).unsqueeze(0))

        # (B,1,H,W)
        L_perp_batch      = torch.cat(Lp_list,    dim=0).float()
        L_anom_norm_batch = torch.cat(Lanom_list, dim=0).float()
        if L_anom_norm_batch.dim() == 3:
            L_anom_norm_batch = L_anom_norm_batch.unsqueeze(1)  # -> (B,1,H,W)
        if L_perp_batch.dim() == 3:
            L_perp_batch = L_perp_batch.unsqueeze(1)  # -> (B,1,H,W)

        return L_perp_batch, L_anom_norm_batch


#========================================== 因为CPU计算缓慢，以下是GPU版本 ==================================================

def box_avg(x: torch.Tensor, win: int | tuple[int,int]):
    """
    x: [B,C,H,W]  CUDA/Tensor
    win: 窗口大小（奇数更好，比如 15）。也可传 (kh, kw)
    """
    if isinstance(win, int):
        kh = kw = win
    else:
        kh, kw = win
    ph, pw = kh // 2, kw // 2
    # count_include_pad=False -> 用有效像素平均，边界不会被零填充值拉低
    return FT.avg_pool2d(x, kernel_size=(kh, kw), stride=1,
                        padding=(ph, pw), count_include_pad=False)

class LumiFeat_GPU(nn.Module):
    """
    亮度异常度 & 去相关度批量计算（纯 torch 版，支持 CUDA）
    输入: batch_lab_raw: [B,3,H,W]  (L:0~100, a,b:-128~127)
    输出: L_perp_batch, L_anom_norm_batch: [B,1,H,W]
    """
    def __init__(self, win=15, bins=32, alpha=0.9, lam1=0.5, lam2=0.5, gmm_down=64):
        super().__init__()
        self.win   = int(win)
        self.bins  = int(bins)
        self.alpha = float(alpha)
        self.lam1  = float(lam1)
        self.lam2  = float(lam2)
        self.gmm_down = int(gmm_down)     # 下采样到 gmm_down×gmm_down 再做 GMM

        # 预置 Sobel 卷积核（放到 buffer，随模块移到 CUDA）
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sobel_x', kx)
        self.register_buffer('sobel_y', ky)

    @torch.no_grad()
    def forward(self, batch_lab_raw: torch.Tensor):
        device = batch_lab_raw.device
        B, _, H, W = batch_lab_raw.shape
        L = batch_lab_raw[:, :1, :, :].contiguous()                 # [B,1,H,W]
        a = batch_lab_raw[:, 1:2, :, :].contiguous()
        b = batch_lab_raw[:, 2:3, :, :].contiguous()

        # --- 1) E,S 特征（GPU） ---
        gx = FT.conv2d(L, self.sobel_x, padding=1)
        gy = FT.conv2d(L, self.sobel_y, padding=1)
        # 结构张量近似 trace：用高斯平滑可加，简化：均值平滑近似
        J11 = FT.avg_pool2d(gx*gx, kernel_size=3, stride=1, padding=1)
        J22 = FT.avg_pool2d(gy*gy, kernel_size=3, stride=1, padding=1)
        trace = J11 + J22
        E = trace                                                      # [B,1,H,W] 作为结构强度
        S = torch.sqrt(a*a + b*b + 1e-8)                               # [B,1,H,W]

        # --- 2) L_perp（批量最小二乘） ---
        X = torch.cat([E, S], dim=1)                                   # [B,2,H,W]
        Xf = X.flatten(2).transpose(1,2)                                # [B,N,2]
        yf = L.flatten(2).transpose(1,2)                                # [B,N,1]
        XtX = torch.matmul(Xf.transpose(1,2), Xf)                       # [B,2,2]
        XtY = torch.matmul(Xf.transpose(1,2), yf)                       # [B,2,1]
        I2  = torch.eye(2, device=device).expand(B,2,2)
        beta = torch.linalg.solve(XtX + 1e-6*I2, XtY).squeeze(-1)       # [B,2]
        pred = (beta[:,0:1,None]*E + beta[:,1:2,None]*S)                # [B,1,H,W]
        L_perp = L - pred                                               # [B,1,H,W]

        # --- 3) 组装 X 并白化（MAD + 协方差） ---
        X3 = torch.cat([L_perp, E, S], dim=1)                           # [B,3,H,W]
        X3f = X3.flatten(2).transpose(1,2)                              # [B,N,3]
        # robust z
        med = X3f.median(dim=1, keepdim=True).values
        mad = (X3f - med).abs().median(dim=1, keepdim=True).values * 1.4826 + 1e-8
        Xz  = (X3f - med) / mad
        # 白化
        diff = Xz - Xz.mean(dim=1, keepdim=True)
        cov  = torch.matmul(diff.transpose(1,2), diff) / Xz.shape[1]    # [B,3,3]
        e,V  = torch.linalg.eigh(cov.clamp_min(1e-8))
        Wwh  = V @ torch.diag_embed(e.rsqrt()) @ V.transpose(1,2)
        Xw   = (Xz - Xz.mean(dim=1, keepdim=True)) @ Wwh                # [B,N,3]

        # --- 4) 投影追踪（GPU 随机向量） ---
        n_try = 128
        a_rand = torch.randn(B, n_try, 3, device=device)
        a_rand = a_rand / (a_rand.norm(dim=-1, keepdim=True) + 1e-12)
        # z = Xw @ a，每个 a 一次性算
        z = torch.matmul(Xw, a_rand.transpose(1,2))                     # [B,N,n_try]
        # 峰度
        mean = z.mean(dim=1, keepdim=True)
        m2   = ((z-mean)**2).mean(dim=1) + 1e-12
        m4   = ((z-mean)**4).mean(dim=1)
        kurt = m4 / (m2**2) - 3.0                                       # [B,n_try]
        best_idx = kurt.abs().argmax(dim=1)                              # [B]
        a_star = a_rand[torch.arange(B, device=device), best_idx]        # [B,3]
        z_best = torch.matmul(Xw, a_star[:, :, None]).squeeze(-1)        # [B,N]

        # --- 5) GMM：建议路线上先用“下采样+sklearn”，然后回 GPU 算责任度 ---
        # 这里示例用最简单的 K=2 EM（GPU），你也可替换为下采样+sklearn 参数回填
        K = 2
        # 初始化：KMeans++ 可加；这里用分位数初始化
        Q25, Q75 = z_best.quantile(0.25, dim=1, keepdim=True), z_best.quantile(0.75, dim=1, keepdim=True)
        mu = torch.stack([Q25.squeeze(1), Q75.squeeze(1)], dim=1)       # [B,K]
        mu = mu[..., None].expand(-1,-1,1)                               # [B,K,1] 与 D=1 对齐
        var = torch.ones(B,K,1, device=device)
        pi  = torch.full((B,K,1), 0.5, device=device)

        x1 = z_best[..., None]                                           # [B,N,1]
        for _ in range(10):  # EM 迭代
            # E-step
            logp = -0.5 * ((x1 - mu)**2 / (var+1e-6)) - 0.5*torch.log(var+1e-6) + torch.log(pi+1e-8) # [B,N,K]
            logp = logp - torch.logsumexp(logp, dim=2, keepdim=True)
            R = logp.exp()                                              # 责任度 [B,N,K]
            Nk = R.sum(dim=1, keepdim=True)                             # [B,1,K]
            # M-step
            mu = (R.transpose(1,2) @ x1) / (Nk.transpose(1,2) + 1e-8)   # [B,K,1]
            var = (R.transpose(1,2) @ ((x1 - mu)**2)) / (Nk.transpose(1,2) + 1e-8) + 1e-6
            pi  = Nk.transpose(1,2) / x1.shape[1]

        # 温度化
        R_alpha = (R ** self.alpha)
        R_alpha = R_alpha / (R_alpha.sum(dim=2, keepdim=True) + 1e-8)    # [B,N,K]
        rk_maps = R_alpha.transpose(1,2).reshape(B, K, 1, H, W)          # [B,K,1,H,W]

        # --- 6) D_M（马氏距离软融合） ---
        # 这里用简单的各簇均值/协方差（在原 3 维特征空间），完全 GPU
        X3f = X3f = torch.cat([L_perp, E, S], dim=1).flatten(2).transpose(1,2)  # [B,N,3]
        Dk = []
        for k in range(K):
            w = R_alpha[..., k]                                          # [B,N]
            Wsum = w.sum(dim=1, keepdim=True) + 1e-8
            mu_k = (w.unsqueeze(-1) * X3f).sum(dim=1, keepdim=True) / Wsum.unsqueeze(-1)   # [B,1,3]
            Xm   = X3f - mu_k
            cov_k = torch.matmul((w.unsqueeze(-1)*Xm).transpose(1,2), Xm) / Wsum.unsqueeze(-1) # [B,3,3]
            cov_k = cov_k + 1e-6*torch.eye(3, device=device)[None]
            inv_k = torch.linalg.inv(cov_k)
            md2 = torch.einsum('bij,bjk,bik->bi', Xm, inv_k, Xm)         # [B,N]
            Dk.append(torch.sqrt(md2 + 1e-8))
        Dk = torch.stack(Dk, dim=2)                                      # [B,N,K]
        Rtemp = FT.softmax(torch.log(R_alpha+1e-8), dim=2)                # 温度化/归一
        D_M  = (Rtemp * Dk).sum(dim=2).reshape(B,1,H,W)                  # [B,1,H,W]

        # --- 7) LLA（GPU） ---
        # 加权均值/方差：用 avg_pool2d
        muB = []; sigB = []
        for k in range(K):
            rk = rk_maps[:, k]       # [B,1,H,W]
            den = box_avg(rk, self.win) + 1e-8
            num1 = box_avg(rk * L_perp, self.win)
            num2 = box_avg(rk * (L_perp**2), self.win)
            mu_k = num1 / den
            var_k = (num2/den - mu_k**2).clamp_min(1e-8)
            muB.append(mu_k); sigB.append(var_k.sqrt())
        LLA = [((L_perp - m).abs() / (s + 1e-8)) for m,s in zip(muB,sigB)]   # K 个 [B,1,H,W]
        # 簇内 z-score（近似，无权）：按图像维度做中位/MAD
        Z_LLA = []
        for k in range(K):
            v = LLA[k].flatten(2)
            med = v.median(dim=2, keepdim=True).values
            mad = (v - med).abs().median(dim=2, keepdim=True).values * 1.4826 + 1e-8
            z  = ((v - med) / mad).reshape(B,1,H,W)
            Z_LLA.append(z)

        # --- 8) GLD（GPU，无逐 bin 循环） ---
        bins = self.bins
        Lmin = L_perp.amin(dim=(2,3), keepdim=True)
        Lmax = L_perp.amax(dim=(2,3), keepdim=True)
        edges = torch.linspace(0, 1, bins+1, device=device)[None,None,:,None,None]
        Lnorm = ((L_perp - Lmin) / (Lmax - Lmin + 1e-8)).clamp(0,1)     # [B,1,H,W]
        idx = torch.clamp((Lnorm * bins).long(), 0, bins-1)              # [B,1,H,W]
        onehot = FT.one_hot(idx.squeeze(1), num_classes=bins).permute(0,3,1,2).float()  # [B,bins,H,W]

        Z_GLD = []
        with torch.autocast(device_type='cuda', enabled=False):
            for k in range(K):
                rk = rk_maps[:, k]                                       # [B,1,H,W]
                w_hist = onehot * rk                                      # [B,bins,H,W]
                # 全局 CDF
                g_hist = w_hist.float().flatten(2).sum(-1)               # [B,bins]
                g_hist = g_hist / (g_hist.sum(dim=1, keepdim=True) + 1e-8)
                g_cdf  = g_hist.cumsum(dim=1)                            # [B,bins]

                # 局部 CDF：对每个 bin 平面做 avg_pool2d，再在 bin 维 cumsum
                loc_hist = box_avg(w_hist, self.win)                      # [B,bins,H,W]
                loc_cdf  = loc_hist.cumsum(dim=1) / (box_avg(rk, self.win)+1e-8)
                # Wasserstein-1
                diff = (loc_cdf - g_cdf[:, :, None, None])               # broadcast
                GLD  = diff.abs().sum(dim=1) * (1.0 / bins)              # [B,H,W]

                # z-score
                v = GLD.flatten(1)
                med = v.median(dim=1, keepdim=True).values
                mad = (v - med).abs().median(dim=1, keepdim=True).values * 1.4826 + 1e-8
                Z_GLD.append(((GLD - med[:,None,None]) / mad[:,None,None]).unsqueeze(1))  # [B,1,H,W]

        # --- 9) gamma 与可靠度 R（GPU） ---
        softplus = lambda x: torch.where(x>=0, x + (x.abs().neg().exp()+1).log(), (x.abs().neg().exp()+1).log())
        gamma = torch.ones_like(L_perp)
        for k in range(K):
            term = self.lam1*softplus(Z_LLA[k]) + self.lam2*softplus(Z_GLD[k])
            gamma = gamma + rk_maps[:,k] * term

        # 可靠度 R：用 E,S 的鲁棒马氏距离（简化近似）
        ES = torch.cat([E,S], dim=1).flatten(2).transpose(1,2)           # [B,N,2]
        med = ES.median(dim=1, keepdim=True).values
        mad = (ES - med).abs().median(dim=1, keepdim=True).values*1.4826 + 1e-8
        ESz = (ES - med)/mad
        diff = ESz - ESz.mean(dim=1, keepdim=True)
        cov  = torch.matmul(diff.transpose(1,2), diff) / ESz.shape[1]
        cov  = cov + 1e-6*torch.eye(2, device=device)[None]
        inv  = torch.linalg.inv(cov)
        d2   = torch.einsum('bij,bjk,bik->bi', diff, inv, diff).reshape(B,1,H,W)
        R_struct = torch.exp(-0.5 * d2)

        # --- 10) L_anom + 归一化 ---
        L_anom = (D_M * gamma * (1 - R_struct)).clamp_min(0)
        q1 = L_anom.flatten(1).quantile(0.01, dim=1, keepdim=True).view(B,1,1,1)
        q9 = L_anom.flatten(1).quantile(0.99, dim=1, keepdim=True).view(B,1,1,1)
        L_anom_norm = ((L_anom - q1) / (q9 - q1 + 1e-8)).clamp(0,1)

        return L_perp, L_anom_norm

### ===================================== GPU版本结束  ==================================================================




# ----------------------------------FL_modele-------------------------------------------------------------------------
class Luminance_Net(nn.Module):
    def __init__(self,
                 mid_channels:   int = 32,
                 feat_channels_down:  int = 128,
                 feat_channels_up: int = 96,
                 feat_channels: int = 128,
                 reflect_ch:     int = 32,
                 heads:          int = 4,
                 win_patch: int = 8,
                 # 以下超参传给 LumiFeat
                 win:   int   = 15,
                 bins:  int   = 64,
                 alpha: float = 0.9,
                 lam1:  float = 0.5,
                 lam2:  float = 0.5):
        super().__init__()

        # ———— 亮度先验计算 （CPU no-grad） ——————————
        self.lumi_feat = LumiFeat(win=win, bins=bins,
                                  alpha=alpha, lam1=lam1, lam2=lam2)

        # ———— Retinex 分解 & 反射率 Embed ——————————
        self.ret_decomp   = RetinexDecompose(eps=1e-3, use_dilation=True)
        self.reflect_emb = ReflectanceEmbed(C_R=reflect_ch)

        # ———— 上下支 LFE —————————————————————————
        # 上支输入通道 = L(1) + A(1) + L⊥(1) = 3
        self.lfe_up   = LFE(in_channels=2,
                            mid_channels=mid_channels,
                            out_channels=feat_channels_up)
        # 下支输入通道 = L⊥(1)
        self.lfe_down = LFE(in_channels=1,
                            mid_channels=mid_channels,
                            out_channels=feat_channels_down)

        self.fuse_1x1 = nn.Sequential(
            nn.Conv2d( feat_channels_up+reflect_ch ,feat_channels , kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # ———— α 预测 —————————————————————————————
        # in_channels = L(1) + A(1) = 2
        self.q_weight = Q_wight(in_channels=2,
                                out_channels=mid_channels,
                                heads=heads,
                                leak=0.1)

        # ———— 多头交叉注意力 ————————————————————————
        # Query 源渠道 = 上支输出 feat_channels + reflect_ch
        # self.attn = LuminanceAttention(
        #     channelsQ   = feat_channels_up + reflect_ch,
        #     channelsKV  = feat_channels,
        #     out_channels= feat_channels,
        #     heads       = heads
        # )
        self.attn = LuminanceAttention(
            channelsQ=feat_channels_up + reflect_ch,  # Q 源通道数
            channelsKV=feat_channels,  # K/V 源通道数
            out_channels=feat_channels,  # 注意力输出通道
            heads=heads,  # 128 % 8 == 0
            win_patch=win_patch  # 或 16
        )

        # ———— 最后一个 1×1 投影（可选残差） —————————
        self.proj = nn.Conv2d(feat_channels, feat_channels, 1, bias=False)

    @torch.no_grad()
    def forward(self, lab_raw: torch.Tensor,lab_norm_ts: torch.Tensor) -> torch.Tensor:
        """
        lab_raw: (B,3,H,W) float32
                 L∈[0,100], a,b∈[-128,127] (未归一化)
        return:  feat (B, feat_channels, H, W)
        """
        # 1. 亮度先验
        L_perp_batch, L_anom_norm_batch = self.lumi_feat(lab_raw)
        # 形状都 (B,1,H,W)

        # 2. 准备单通道 L
        L = lab_raw[:, :1, ...]   # (B,1,H,W)
        L_ts = lab_norm_ts[:, :1, ...]   # (B,1,H,W)

        # 3. Retinex 得到 R̂ & 嵌入
        I_hat, R_hat = self.ret_decomp(L_ts)          # (B,1,H,W)×2
        R_emb = self.reflect_emb(R_hat)  # (B,reflect_ch,H,W)

        # 4. 上支路特征提取
        x_up  = torch.cat([L_ts, L_anom_norm_batch], dim=1)  # (B,4,H,W)
        f_up  = self.lfe_up(x_up)                           # (B,feat_ch,H,W)

        # 5. 下支路特征提取
        f_down = self.lfe_down(L_ts)                      # (B,feat_ch,H,W)

        # 6. α 权重
        alpha = self.q_weight(L_ts, L_anom_norm_batch)               # (B, heads)

        # 7. 组装 Query + Attention
        Q_src = torch.cat([f_up, R_emb], dim=1)             # (B,feat_ch+reflect_ch,H,W)
        Q_src2 =  self.fuse_1x1(Q_src)
        out   = self.attn(Q_src2, f_down, alpha)             # (B,feat_ch,H,W)

        # 8. 可选残差融合
        feat  = self.proj(out) + f_down                     # (B,feat_ch,H,W)
        return feat


#====================================== 利用CPU计算亮度异常度且不加校正模块 =======================================================
class Luminance_Net1(nn.Module):
    def __init__(self,
                 mid_channels,
                 feat_channels_down,
                 feat_channels_up,
                 feat_channels,
                 reflect_ch,
                 heads,
                 win_patch,
                 # 以下超参传给 LumiFeat
                 win,
                 bins,
                 alpha,
                 lam1,
                 lam2):
        super().__init__()

        # ———— 亮度先验计算 （CPU no-grad） ——————————
        self.lumi_feat = LumiFeat(win=win, bins=bins,
                                  alpha=alpha, lam1=lam1, lam2=lam2)


        # ———— Retinex 分解 & 反射率 Embed ——————————
        self.ret_decomp   = RetinexDecompose(eps=1e-3, use_dilation=True)
        self.reflect_emb = ReflectanceEmbed(C_R=reflect_ch)

        # ———— 上下支 LFE —————————————————————————
        # 上支输入通道 = L(1) + A(1) + L⊥(1) = 3
        self.lfe_up   = LFE(in_channels=2,
                            mid_channels=mid_channels,
                            out_channels=feat_channels_up)
        # 下支输入通道 = L⊥(1)
        self.lfe_down = LFE(in_channels=1,
                            mid_channels=mid_channels,
                            out_channels=feat_channels_down)

        self.fuse_1x1 = nn.Sequential(
            nn.Conv2d( feat_channels_up+reflect_ch ,feat_channels , kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # ———— α 预测 —————————————————————————————
        # in_channels = L(1) + A(1) = 2
        self.q_weight = Q_wight(in_channels=2,
                                out_channels=mid_channels,
                                heads=heads,
                                leak=0.1)

        # ———— 多头交叉注意力 ————————————————————————
        # Query 源渠道 = 上支输出 feat_channels + reflect_ch
        # self.attn = LuminanceAttention(
        #     channelsQ   = feat_channels_up + reflect_ch,
        #     channelsKV  = feat_channels,
        #     out_channels= feat_channels,
        #     heads       = heads
        # )
        self.attn = LuminanceAttention(
            channelsQ=feat_channels_up + reflect_ch,  # Q 源通道数
            channelsKV=feat_channels,  # K/V 源通道数
            out_channels=feat_channels,  # 注意力输出通道
            heads=heads,  # 128 % 8 == 0
            win_patch=win_patch  # 或 16
        )

        # ———— 最后一个 1×1 投影（可选残差） —————————
        self.proj = nn.Conv2d(feat_channels, feat_channels, 1, bias=False)

    @torch.no_grad()
    def forward(self, lab_raw: torch.Tensor,lab_norm_ts: torch.Tensor) -> torch.Tensor:
        """
        lab_raw: (B,3,H,W) float32
                 L∈[0,100], a,b∈[-128,127] (未归一化)
        return:  feat (B, feat_channels, H, W)
        """
        # 1. 亮度先验
        L_perp_batch, L_anom_norm_batch = self.lumi_feat(lab_raw)
        # 形状都 (B,1,H,W)

        # 2. 准备单通道 L
        L = lab_raw[:, :1, ...]   # (B,1,H,W)
        L_ts = lab_norm_ts[:, :1, ...]   # (B,1,H,W)

        # 3. Retinex 得到 R̂ & 嵌入
        I_hat, R_hat = self.ret_decomp(L_ts)          # (B,1,H,W)×2
        R_emb = self.reflect_emb(R_hat)  # (B,reflect_ch,H,W)

        # 4. 上支路特征提取
        #x_up  = torch.cat([L_ts, L_anom_norm_batch], dim=1)  # (B,2,H,W)
        x_up = torch.cat([L_ts, L_anom_norm_batch], dim=1)  # (B,2,H,W)  # 利用CPU计算亮度异常同时加入校正模块
        f_up  = self.lfe_up(x_up)                           # (B,feat_ch,H,W)

        # 5. 下支路特征提取
        f_down = self.lfe_down(L_ts)                      # (B,feat_ch,H,W)

        # 6. α 权重
        alpha = self.q_weight(L_ts, L_anom_norm_batch)               # (B, heads)

        # 7. 组装 Query + Attention
        Q_src = torch.cat([f_up, R_emb], dim=1)             # (B,feat_ch+reflect_ch,H,W)
        Q_src2 =  self.fuse_1x1(Q_src)
        out   = self.attn(Q_src2, f_down, alpha)             # (B,feat_ch,H,W)

        # 8. 可选残差融合
        feat  = self.proj(out) + f_down                     # (B,feat_ch,H,W)
        return feat



#================================= 利用CPU计算亮度异常度且加校正模块并暴露中间量亮度异常值和校正后的异常值 ===========================
class Luminance_Net1(nn.Module):
    def __init__(self,
                 mid_channels,
                 feat_channels_down,
                 feat_channels_up,
                 feat_channels,
                 reflect_ch,
                 heads,
                 win_patch,
                 # 以下超参传给 LumiFeat
                 win,
                 bins,
                 alpha,
                 lam1,
                 lam2):
        super().__init__()

        # ———— 亮度先验计算 （CPU no-grad） ——————————
        self.lumi_feat = LumiFeat(win=win, bins=bins,
                                  alpha=alpha, lam1=lam1, lam2=lam2)

        self.JZ = LumiPriorCalib()  #加入的异常度校正模块

        # ———— Retinex 分解 & 反射率 Embed ——————————
        self.ret_decomp   = RetinexDecompose(eps=1e-3, use_dilation=True)
        self.reflect_emb = ReflectanceEmbed(C_R=reflect_ch)

        # ———— 上下支 LFE —————————————————————————
        # 上支输入通道 = L(1) + A(1) + L⊥(1) = 3
        self.lfe_up   = LFE(in_channels=2,
                            mid_channels=mid_channels,
                            out_channels=feat_channels_up)
        # 下支输入通道 = L⊥(1)
        self.lfe_down = LFE(in_channels=1,
                            mid_channels=mid_channels,
                            out_channels=feat_channels_down)

        self.fuse_1x1 = nn.Sequential(
            nn.Conv2d( feat_channels_up+reflect_ch ,feat_channels , kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # ———— α 预测 —————————————————————————————
        # in_channels = L(1) + A(1) = 2
        self.q_weight = Q_wight(in_channels=2,
                                out_channels=mid_channels,
                                heads=heads,
                                leak=0.1)

        # ———— 多头交叉注意力 ————————————————————————
        # Query 源渠道 = 上支输出 feat_channels + reflect_ch
        # self.attn = LuminanceAttention(
        #     channelsQ   = feat_channels_up + reflect_ch,
        #     channelsKV  = feat_channels,
        #     out_channels= feat_channels,
        #     heads       = heads
        # )
        self.attn = LuminanceAttention(
            channelsQ=feat_channels_up + reflect_ch,  # Q 源通道数
            channelsKV=feat_channels,  # K/V 源通道数
            out_channels=feat_channels,  # 注意力输出通道
            heads=heads,  # 128 % 8 == 0
            win_patch=win_patch  # 或 16
        )

        # ———— 最后一个 1×1 投影（可选残差） —————————
        self.proj = nn.Conv2d(feat_channels, feat_channels, 1, bias=False)

    @torch.no_grad()
    def forward(self, lab_raw: torch.Tensor,lab_norm_ts: torch.Tensor,return_aux: bool = False) -> torch.Tensor:
        """
        lab_raw: (B,3,H,W) float32
                 L∈[0,100], a,b∈[-128,127] (未归一化)
        return:  feat (B, feat_channels, H, W)
        """
        # 1. 亮度先验
        with torch.no_grad():
            L_perp_ref, A_ref = self.lumi_feat(lab_raw)  # [B,1,H,W] in [0,1]
            #A_ref = A_ref.to(device)
        #L_perp_batch, L_anom_norm_batch = self.lumi_feat(lab_raw)
        A_pred = self.JZ(A_ref)
        # 形状都 (B,1,H,W)

        # 2. 准备单通道 L
        L = lab_raw[:, :1, ...]   # (B,1,H,W)
        L_ts = lab_norm_ts[:, :1, ...]   # (B,1,H,W)

        # 3. Retinex 得到 R̂ & 嵌入
        I_hat, R_hat = self.ret_decomp(L_ts)          # (B,1,H,W)×2
        R_emb = self.reflect_emb(R_hat)  # (B,reflect_ch,H,W)

        # 4. 上支路特征提取
        #x_up  = torch.cat([L_ts, L_anom_norm_batch], dim=1)  # (B,2,H,W)
        x_up = torch.cat([L_ts, A_pred], dim=1)  # (B,2,H,W)  # 利用CPU计算亮度异常同时加入校正模块
        f_up  = self.lfe_up(x_up)                           # (B,feat_ch,H,W)

        # 5. 下支路特征提取
        f_down = self.lfe_down(L_ts)                      # (B,feat_ch,H,W)

        # 6. α 权重
        alpha = self.q_weight(L_ts, A_pred)               # (B, heads)

        # 7. 组装 Query + Attention
        Q_src = torch.cat([f_up, R_emb], dim=1)             # (B,feat_ch+reflect_ch,H,W)
        Q_src2 =  self.fuse_1x1(Q_src)
        out   = self.attn(Q_src2, f_down, alpha)             # (B,feat_ch,H,W)

        # 8. 可选残差融合
        feat  = self.proj(out) + f_down                     # (B,feat_ch,H,W)

        if return_aux:
            aux = {'A_ref': A_ref, 'A_pred': A_pred}  # 有 Q_adj/门控也可以加进来
            return feat, aux

        return feat



#==================================   加入残差校正模块   =============================================================
class LumiPriorCalib(nn.Module):
    """
    受限残差校准器（单通道版）
    输入:  A_ref ∈ [B,1,H,W] (stop-grad 产生的亮度异常度图)
    输出:  A_pred ∈ [B,1,H,W] (微调后, 仍在[0,1])
    公式:  A_pred = clip( A_ref + eps * tanh( PW(GELU(DW3x3(A_ref))) ), 0, 1)
    """
    def __init__(self, eps: float = 0.10, use_logit: bool = False):
        super().__init__()
        self.eps = float(eps)
        self.use_logit = bool(use_logit)
        # 单通道时，DW=普通3x3；保留写法清晰
        self.dw = nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1, bias=False)
        self.pw = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        # 身份初始化：起步 A_pred = A_ref
        nn.init.zeros_(self.dw.weight)
        nn.init.zeros_(self.pw.weight)

    def forward(self, A_ref: torch.Tensor) -> torch.Tensor:
        # A_ref ∈ [0,1]
        if not self.use_logit:
            r = self.pw(FT.gelu(self.dw(A_ref)))          # 残差原始值
            A_pred = torch.clamp(A_ref + self.eps * torch.tanh(r), 0.0, 1.0)
            return A_pred
        else:
            # 处处可导的替代：sigmoid(logit(A_ref) + eps * net(A_ref))
            x = A_ref.clamp(1e-6, 1-1e-6)
            logit = torch.log(x) - torch.log(1 - x)
            r = self.pw(FT.gelu(self.dw(x)))
            return torch.sigmoid(logit + self.eps * r)

class Luminance_Net2(nn.Module):
    def __init__(self,
                 mid_channels,
                 feat_channels_down,
                 feat_channels_up,
                 feat_channels,
                 reflect_ch,
                 heads,
                 win_patch,
                 # 以下超参传给 LumiFeat
                 win,
                 bins,
                 alpha,
                 lam1,
                 lam2):
        super().__init__()

        # ———— 亮度先验计算 （GPU no-grad） ——————————
        self.lumi_feat = LumiFeat_GPU(win=win, bins=bins,
                                  alpha=alpha, lam1=lam1, lam2=lam2)

        self.JZ  = LumiPriorCalib()

        # ———— Retinex 分解 & 反射率 Embed ——————————
        self.ret_decomp   = RetinexDecompose(eps=1e-3, use_dilation=True)
        self.reflect_emb = ReflectanceEmbed(C_R=reflect_ch)

        # ———— 上下支 LFE —————————————————————————
        # 上支输入通道 = L(1) + A(1) + L⊥(1) = 3
        self.lfe_up   = LFE(in_channels=2,
                            mid_channels=mid_channels,
                            out_channels=feat_channels_up)
        # 下支输入通道 = L⊥(1)
        self.lfe_down = LFE(in_channels=1,
                            mid_channels=mid_channels,
                            out_channels=feat_channels_down)

        self.fuse_1x1 = nn.Sequential(
            nn.Conv2d( feat_channels_up+reflect_ch ,feat_channels , kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # ———— α 预测 —————————————————————————————
        # in_channels = L(1) + A(1) = 2
        self.q_weight = Q_wight(in_channels=2,
                                out_channels=mid_channels,
                                heads=heads,
                                leak=0.1)

        # ———— 多头交叉注意力 ————————————————————————
        # Query 源渠道 = 上支输出 feat_channels + reflect_ch
        # self.attn = LuminanceAttention(
        #     channelsQ   = feat_channels_up + reflect_ch,
        #     channelsKV  = feat_channels,
        #     out_channels= feat_channels,
        #     heads       = heads
        # )
        self.attn = LuminanceAttention(
            channelsQ=feat_channels_up + reflect_ch,  # Q 源通道数
            channelsKV=feat_channels,  # K/V 源通道数
            out_channels=feat_channels,  # 注意力输出通道
            heads=heads,  # 128 % 8 == 0
            win_patch=win_patch  # 或 16
        )

        # ———— 最后一个 1×1 投影（可选残差） —————————
        self.proj = nn.Conv2d(feat_channels, feat_channels, 1, bias=False)

    @torch.no_grad()
    def forward(self, lab_raw: torch.Tensor,lab_norm_ts: torch.Tensor) -> torch.Tensor:
        """
        lab_raw: (B,3,H,W) float32
                 L∈[0,100], a,b∈[-128,127] (未归一化)
        return:  feat (B, feat_channels, H, W)
        """
        # 1. 亮度先验
        L_perp_batch, L_anom_norm_batch = self.lumi_feat(lab_raw)
        L_anom_norm_batch_JZ = self.JZ(L_anom_norm_batch)
        # 形状都 (B,1,H,W)

        # 2. 准备单通道 L
        L = lab_raw[:, :1, ...]   # (B,1,H,W)
        L_ts = lab_norm_ts[:, :1, ...]   # (B,1,H,W)

        # 3. Retinex 得到 R̂ & 嵌入
        I_hat, R_hat = self.ret_decomp(L_ts)          # (B,1,H,W)×2
        R_emb = self.reflect_emb(R_hat)  # (B,reflect_ch,H,W)

        # 4. 上支路特征提取
        x_up  = torch.cat([L_ts, L_anom_norm_batch_JZ], dim=1)  # (B,2,H,W)
        f_up  = self.lfe_up(x_up)                           # (B,feat_ch,H,W)

        # 5. 下支路特征提取
        f_down = self.lfe_down(L_ts)                      # (B,feat_ch,H,W)

        # 6. α 权重
        alpha = self.q_weight(L_ts, L_anom_norm_batch)               # (B, heads)

        # 7. 组装 Query + Attention
        Q_src = torch.cat([f_up, R_emb], dim=1)             # (B,feat_ch+reflect_ch,H,W)
        Q_src2 =  self.fuse_1x1(Q_src)
        out   = self.attn(Q_src2, f_down, alpha)             # (B,feat_ch,H,W)

        # 8. 可选残差融合
        feat  = self.proj(out) + f_down                     # (B,feat_ch,H,W)
        return feat