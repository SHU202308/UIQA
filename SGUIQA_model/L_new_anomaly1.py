import numpy as np
import cv2
from skimage import color, filters
from sklearn.mixture import GaussianMixture
from sklearn.covariance import MinCovDet
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import os
from tqdm import tqdm
import torch





# 0. 工具函数：鲁棒统计、卷积辅助
def mad(x):
    """Median Absolute Deviation."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))

def robust_zscore(x):
    """鲁棒 z-score：z = (x - median)/MAD."""
    med = np.median(x)
    m = mad(x) + 1e-8
    return (x - med) / m, med, m

# def softplus(x):
#     return np.log1p(np.exp(x))  # ln(1+e^x)
def softplus(x):
    return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))

def box_filter(img, ksize):
    """均值滤波（可当做加权分子/分母时的卷积）。"""
    return uniform_filter(img, size=ksize, mode='reflect')


# 1. 预处理：读图、缩放、转 Lab
def load_resize_to_lab(img_path, size=256):
    """读取图像 -> resize -> 转 Lab (range: L in [0,100], a,b in [-128,127])"""
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(img_path)
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lab = color.rgb2lab(rgb)  # skimage
    L = lab[..., 0]  # 0~100
    a = lab[..., 1]
    b = lab[..., 2]
    # L_raw = lab[..., 0]  # 0~100
    # a_raw = lab[..., 1]
    # b_raw = lab[..., 2]
    # L =  L_raw / 100.0
    # a = (a_raw + 128.0) / 255.0-1
    # b = (b_raw + 128.0) / 255.0 - 1
    return L, a, b, rgb

# 2. 特征提取 E（结构张量）与 S（色度）
def compute_texture_E(L, sigma=1.5):
    """结构张量 + 特征值和：E = λ1+λ2 = trace(J)."""
    # 一阶梯度
    gx = filters.sobel_h(L)
    gy = filters.sobel_v(L)
    # 二阶项
    J11 = filters.gaussian(gx*gx, sigma=sigma)
    J22 = filters.gaussian(gy*gy, sigma=sigma)
    J12 = filters.gaussian(gx*gy, sigma=sigma)
    # 特征值闭式：trace±sqrt(trace^2-4det) / 2
    trace = J11 + J22
    det = J11*J22 - J12**2
    # 为了稳定，有时det可能略<0，clip一下
    tmp = np.maximum(trace**2 - 4*det, 0)
    lam1 = 0.5*(trace + np.sqrt(tmp))
    lam2 = 0.5*(trace - np.sqrt(tmp))
    E = lam1 + lam2  # 或 lam1 (最大特征值) 也可
    return E

def compute_chroma_S(a, b):
    """色度幅值 S = sqrt(a^2+b^2). 可再做归一化到[0,1]"""
    S = np.sqrt(a**2 + b**2)
    return S

# 3. 亮度去相关 L⊥
def compute_L_perp(L, E, S):
    """L_perp = L - [E,S] beta，beta 用最小二乘或鲁棒回归"""
    H, W = L.shape
    X = np.stack([E.ravel(), S.ravel()], axis=1)  # N×2
    y = L.ravel()
    # 普通最小二乘（也可用 Huber/Tukey）
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    L_perp = L - (beta[0]*E + beta[1]*S)
    return L_perp, beta


# 4. 鲁棒标准化 & 白化
def robust_whitening(X):
    """
    X: (N, D) feature matrix. (Here D=3)
    返回：white_X, robust_mean, robust_cov_inv_sqrt
    """
    # 每维鲁棒z-score
    Xz = np.zeros_like(X)
    med = np.zeros(X.shape[1])
    mads = np.zeros(X.shape[1])
    for d in range(X.shape[1]):
        Xz[:, d], med[d], mads[d] = robust_zscore(X[:, d])

    # 用 MinCovDet 做鲁棒协方差估计
    mcd = MinCovDet().fit(Xz)
    cov = mcd.covariance_
    mean = mcd.location_

    # 白化：W = Sigma^{-1/2}
    # eigh 保证对称矩阵的精确分解
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-8, None)
    inv_sqrt = vecs @ np.diag(1.0/np.sqrt(vals)) @ vecs.T

    Xw = (Xz - mean) @ inv_sqrt
    return Xw, (med, mads, mean, inv_sqrt)

#5. 投影追踪（峰度最大化） & GMM+BIC
def kurtosis(x):
    # 传统无偏峰度估计也可以，这里用简单版本
    m2 = np.mean((x - np.mean(x))**2) + 1e-12
    m4 = np.mean((x - np.mean(x))**4)
    return m4 / (m2**2) - 3

def projection_pursuit_kurtosis(Xw, n_try=200):
    """
    简单随机搜索单位向量 a，使 |kurtosis(a^T Xw)| 最大.
    Xw: (N, D)
    """
    N, D = Xw.shape
    best_k = -np.inf
    best_a = None
    for _ in range(n_try):
        a = np.random.randn(D)
        a /= np.linalg.norm(a) + 1e-12
        z = Xw @ a
        k = abs(kurtosis(z))
        if k > best_k:
            best_k = k
            best_a = a
    # 再做一次局部细化可选，这里省略
    z_best = Xw @ best_a
    return z_best, best_a

def fit_gmm_bic(z, max_components=3, n_init=5):
    """在 z 上试 K=1..max_components, 用BIC选最优"""
    z = z.reshape(-1, 1)
    best_bic = np.inf
    best_gmm = None
    for K in range(1, max_components+1):
        gmm = GaussianMixture(
            n_components=K, covariance_type='full',
            n_init=n_init, reg_covar=1e-6, random_state=0
        )
        gmm.fit(z)
        bic = gmm.bic(z)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
    r = best_gmm.predict_proba(z)  # N×K
    return best_gmm, r

#6. 簇内鲁棒 MD（软权重融合）
def weighted_cov_mean(X, w):
    """
    X: (N,D), w: (N,)
    返回 weighted mean & covariance.
    """
    w = w.reshape(-1, 1)
    W = np.sum(w)
    mu = np.sum(w * X, axis=0) / W
    Xm = X - mu
    cov = (w * Xm).T @ Xm / W
    # 稳定性增强
    cov += 1e-8 * np.eye(X.shape[1])
    return mu, cov

def mahalanobis_distance(X, mu, cov_inv):
    Xm = X - mu
    # MD = sqrt(diag(Xm*cov_inv*Xm^T))
    return np.sqrt(np.sum(Xm @ cov_inv * Xm, axis=1))

def cluster_md_soft(X, R, alpha=0.9):
    """
    X: (N,D)
    R: (N,K) responsibility
    返回 D_M(p)  (N,)
    """
    N, D = X.shape
    K = R.shape[1]
    # 先算每个簇的 MD
    Dk_list = []
    for k in range(K):
        w = R[:, k]
        mu_k, cov_k = weighted_cov_mean(X, w)
        cov_inv = np.linalg.inv(cov_k)
        Dk = mahalanobis_distance(X, mu_k, cov_inv)
        Dk_list.append(Dk)
    Dk_arr = np.stack(Dk_list, axis=1)  # N×K
    # 温度化权重
    R_alpha = R**alpha
    R_alpha /= np.sum(R_alpha, axis=1, keepdims=True)
    D_M = np.sum(R_alpha * Dk_arr, axis=1)
    return D_M, Dk_arr, R_alpha

# 7. 加权窗口统计 (LLA) & 加权CDF (GLD)
def weighted_local_stats(Lp, rk, win=8):
    """
    计算簇k下窗口内的加权均值/方差。
    Lp, rk: H×W
    返回: mu_k_block, sigma_k_block (H×W)
    """
    # 分子：sum r_k * Lp, sum r_k * Lp^2
    num1 = box_filter(rk * Lp, win)
    num2 = box_filter(rk * (Lp**2), win)
    den = box_filter(rk, win) + 1e-8
    mu = num1 / den
    var = num2 / den - mu**2
    var = np.clip(var, 1e-8, None)
    sigma = np.sqrt(var)
    return mu, sigma

def compute_LLA_z(Lp, R_alpha, rk_list, win=8):
    """
    对每个簇k求 LLA^(k)(p) -> Z-score(簇内)。
    输入:
        Lp: L_perp (H×W)
        R_alpha: (N,K) 温度化权重
        rk_list: list of H×W maps for each k
    返回:
        Z_LLA_list: list of H×W
    """
    H, W = Lp.shape
    N, K = R_alpha.shape
    Z_LLA_list = []
    for k in range(K):
        rk = rk_list[k]
        mu_kB, sig_kB = weighted_local_stats(Lp, rk, win)
        LLA_k = np.abs(Lp - mu_kB) / (sig_kB + 1e-8)

        # 簇内 z-score (用像素责任度做权重)
        vec = LLA_k.ravel()
        w = R_alpha[:, k]  # N,
        # 权重版 median/MAD：简单做法是无权 median/MAD 或加权approx
        # 这里先用无权近似，严格加权MAD可自行实现
        med = np.median(vec)
        m = mad(vec) + 1e-8
        Z_LLA_k = (LLA_k - med)/m
        Z_LLA_list.append(Z_LLA_k)
    return Z_LLA_list

# 7.2 GLD (Wasserstein-1 / EMD)
def compute_GLD_z(Lp, rk_list, R_alpha, bins=64, win=8):
    """
    计算 GLD^(k)(p): 窗口 vs. 簇内全局 CDF 的加权 Wasserstein-1。
    返回 Z-score 之后的列表。
    """
    H, W = Lp.shape
    N, K = R_alpha.shape
    Z_GLD_list = []

    L_min, L_max = Lp.min(), Lp.max()
    edges = np.linspace(L_min, L_max, bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    delta = centers[1]-centers[0]

    # 先为每个簇计算全局CDF
    global_cdfs = []
    for k in range(K):
        rk = rk_list[k].ravel()
        hist, _ = np.histogram(Lp.ravel(), bins=edges, weights=rk, density=False)
        hist = hist.astype(np.float64)
        hist /= (hist.sum() + 1e-8)
        cdf = np.cumsum(hist)
        global_cdfs.append(cdf)

    # 为局部CDF做准备：对每个 bin j，生成 indicator 并做加权卷积 (rk * 1[L<=edge_j])。
    # 直接逐bin卷积很慢，这里优化思路：把 Lp 量化成 bin index，再用累积和。
    # 简便起见，这里展示直接逐bin滑窗(学习/论文版可这么写；工程版需加速)。

    # Precompute for local sums: r_k and r_k*indicator
    # indicator_j(q) = 1 if Lp(q)<=edge_j else 0
    L_flat = Lp.ravel()
    bin_idx = np.digitize(L_flat, edges) - 1  # 0..bins-1
    bin_idx = np.clip(bin_idx, 0, bins-1).reshape(H, W)

    for k in range(K):
        rk = rk_list[k]
        # 构造每个 bin 的局部累计权重（使用逐bin累积 + box_filter）
        # local_cdf[p,j] = sum_{q in B(p)} rk(q)*1[L(q)<=edge_j] / sum rk(q)
        den = box_filter(rk, win) + 1e-8
        local_cdf_stack = np.zeros((H, W, bins), dtype=np.float32)

        # 逐bin累积：为了效率其实可以做前缀和，这里简写成loop
        # 每一层: mask = (bin_idx <= j)
        # 用 r_k * mask 做卷积 / den
        # 优化思路：先排序，然后累积，这里为清晰性保持简单写法
        for j in range(bins):
            mask = (bin_idx <= j).astype(np.float32)
            num = box_filter(rk * mask, win)
            local_cdf_stack[..., j] = num / den

        # Wasserstein-1: sum |local_cdf - global_cdf| * delta
        global_cdf = global_cdfs[k]
        diff = np.abs(local_cdf_stack - global_cdf[None, None, :])
        GLD_k = np.sum(diff, axis=2) * delta

        # 簇内 z-score
        vec = GLD_k.ravel()
        # 同上，简单用无权 median/MAD
        med = np.median(vec)
        m = mad(vec) + 1e-8
        Z_GLD_k = (GLD_k - med)/m

        Z_GLD_list.append(Z_GLD_k)

    return Z_GLD_list

# 8. 亮度先验增强项 γ 和 结构/色度可靠度 R
def compute_gamma(Z_LLA_list, Z_GLD_list, R_alpha, lam1=0.5, lam2=0.5):
    """
    Z_LLA_list / Z_GLD_list: list of H×W maps, length=K.
    R_alpha: (N,K)
    """
    K = R_alpha.shape[1]
    H, W = Z_LLA_list[0].shape
    gamma = np.ones((H, W), dtype=np.float32)
    for k in range(K):
        term = lam1*softplus(Z_LLA_list[k]) + lam2*softplus(Z_GLD_list[k])
        # R_alpha 是 N×K，要 reshape 回 H×W
        w_k = R_alpha[:, k].reshape(H, W)
        gamma += w_k * term
    return gamma

def compute_reliability(E, S):
    """
    结构/色度可靠度 R(p) = exp(-1/2 * d_ES^2),
    其中 d_ES^2 用 E,S 的鲁棒 z-score 做马氏距离
    """
    H, W = E.shape
    y = np.stack([E.ravel(), S.ravel()], axis=1)

    # 鲁棒 z-score
    Ez, Em, Emad = robust_zscore(y[:,0])
    Sz, Sm, Smad = robust_zscore(y[:,1])
    yz = np.stack([Ez, Sz], axis=1)

    mcd = MinCovDet().fit(yz)
    mu = mcd.location_
    cov = mcd.covariance_
    cov_inv = np.linalg.inv(cov)

    d2 = mahalanobis_distance(yz, mu, cov_inv)**2
    R = np.exp(-0.5 * d2).reshape(H, W)
    return R


# 9. 总控函数：整合一切
def compute_luminance_anomaly(img_path,
                              size=224,
                              win=15,
                              bins=64,
                              alpha=0.9,
                              lam1=0.5,
                              lam2=0.5):
    # 0. 读图 & Lab
    L, a, b, rgb = load_resize_to_lab(img_path, size)
    H, W = L.shape

    # 1. 特征
    E = compute_texture_E(L)
    S = compute_chroma_S(a, b)

    # 2. L_perp
    L_perp, beta = compute_L_perp(L, E, S)

    # 3. 组装X并鲁棒白化
    X = np.stack([L_perp.ravel(), E.ravel(), S.ravel()], axis=1)
    Xw, white_params = robust_whitening(X)

    # 4. 投影追踪 & GMM
    z, a_star = projection_pursuit_kurtosis(Xw, n_try=200)
    gmm, R = fit_gmm_bic(z, max_components=3, n_init=5)
    N, K = R.shape
    # 温度化
    R_alpha = R**alpha
    R_alpha /= np.sum(R_alpha, axis=1, keepdims=True)

    # 5. 簇内 MD
    D_M, Dk_arr, R_alpha_norm = cluster_md_soft(X, R_alpha, alpha=1.0)  # 这里再用一次alpha=1也行
    D_M_map = D_M.reshape(H, W)

    # 6. LLA/GLD 软式
    # 每个簇的 r_k map
    rk_maps = [R_alpha[:, k].reshape(H, W) for k in range(K)]
    Z_LLA_list = compute_LLA_z(L_perp, R_alpha, rk_maps, win=win)
    Z_GLD_list = compute_GLD_z(L_perp, rk_maps, R_alpha, bins=bins, win=win)

    # 7. gamma
    gamma = compute_gamma(Z_LLA_list, Z_GLD_list, R_alpha, lam1=lam1, lam2=lam2)

    # 8. 可靠度 R
    R_struct = compute_reliability(E, S)

    # 9. 最终 L_anom
    L_anom = D_M_map * gamma * (1 - R_struct)

    # 10. 可选归一化
    q1, q99 = np.percentile(L_anom, [1, 99])
    L_anom_norm = np.clip((L_anom - q1) / (q99 - q1 + 1e-8), 0, 1)

    results = {
        'rgb' : rgb,
        'L_perp': L_perp,
        'E': E,
        'S': S,
        'D_M': D_M_map,
        'gamma': gamma,
        'R_struct': R_struct,
        'L_anom': L_anom,
        'L_anom_norm': L_anom_norm,
        'Z_LLA_list': Z_LLA_list,
        'Z_GLD_list': Z_GLD_list,
        'R_alpha': R_alpha.reshape(H,W,K),
        'beta': beta,
        'a_star': a_star
    }
    return results

##-------------------------------L_channle 调用的亮度异常度计算函数----------------------------------------------------
def Luminance_anomaly(
                              LAB_raw,
                              win=15,
                              bins=64,
                              alpha=0.9,
                              lam1=0.5,
                              lam2=0.5):
    # 0. 读图 & Lab
    L = LAB_raw[..., 0].astype('float32')
    a = LAB_raw[..., 1].astype('float32')
    b = LAB_raw[..., 2].astype('float32')

    H, W = L.shape

    # 1. 特征
    E = compute_texture_E(L)
    S = compute_chroma_S(a, b)

    # 2. L_perp
    L_perp, beta = compute_L_perp(L, E, S)

    # 3. 组装X并鲁棒白化
    X = np.stack([L_perp.ravel(), E.ravel(), S.ravel()], axis=1)
    Xw, white_params = robust_whitening(X)

    # 4. 投影追踪 & GMM
    z, a_star = projection_pursuit_kurtosis(Xw, n_try=200)
    gmm, R = fit_gmm_bic(z, max_components=3, n_init=5)
    N, K = R.shape
    # 温度化
    R_alpha = R**alpha
    R_alpha /= np.sum(R_alpha, axis=1, keepdims=True)

    # 5. 簇内 MD
    D_M, Dk_arr, R_alpha_norm = cluster_md_soft(X, R_alpha, alpha=1.0)  # 这里再用一次alpha=1也行
    D_M_map = D_M.reshape(H, W)

    # 6. LLA/GLD 软式
    # 每个簇的 r_k map
    rk_maps = [R_alpha[:, k].reshape(H, W) for k in range(K)]
    Z_LLA_list = compute_LLA_z(L_perp, R_alpha, rk_maps, win=win)
    Z_GLD_list = compute_GLD_z(L_perp, rk_maps, R_alpha, bins=bins, win=win)

    # 7. gamma
    gamma = compute_gamma(Z_LLA_list, Z_GLD_list, R_alpha, lam1=lam1, lam2=lam2)

    # 8. 可靠度 R
    R_struct = compute_reliability(E, S)

    # 9. 最终 L_anom
    L_anom = D_M_map * gamma * (1 - R_struct)

    # 10. 可选归一化
    q1, q99 = np.percentile(L_anom, [1, 99])
    L_anom_norm = np.clip((L_anom - q1) / (q99 - q1 + 1e-8), 0, 1)

    results = {
        'L_perp': L_perp,
        'E': E,
        'S': S,
        'D_M': D_M_map,
        'gamma': gamma,
        'R_struct': R_struct,
        'L_anom': L_anom,
        'L_anom_norm': L_anom_norm,
        'Z_LLA_list': Z_LLA_list,
        'Z_GLD_list': Z_GLD_list,
        'R_alpha': R_alpha.reshape(H,W,K),
        'beta': beta,
        'a_star': a_star
    }
    return results



# # 方案 A：在同一窗口里添加红色热力图子图
#
# if __name__ == "__main__":
#     img_path = r'F:\image_data\underwater_image_data\UIF\original_image\raw\988.bmp'
#     res = compute_luminance_anomaly(img_path)
#
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import LinearSegmentedColormap
#     import numpy as np
#
#     # 自定义 黑→红→黄 的 colormap（也可以用 plt.cm.hot / plt.cm.magma）
#     red_map = LinearSegmentedColormap.from_list(
#         'red_map',
#         [(0.0, (0, 0, 0)),      # 黑
#          (0.6, (1, 0, 0)),      # 红
#          (1.0, (1, 1, 0))],     # 黄
#         N=256
#     )
#     rgb = res['rgb']
#     L_anom      = res['L_anom']
#     L_anom_norm = res['L_anom_norm']
#     print("L_anom.shape:",L_anom.shape)
#     print("L_anom_norm.shape:", L_anom_norm.shape)
#     print('L_anom      - min:', L_anom.min(), ' max:', L_anom.max())
#     print('L_anom_norm - min:', L_anom_norm.min(), ' max:', L_anom_norm.max())

# #####################   1.画图
#     plt.figure(figsize=(15, 5))
#     # # 左：原始灰度
#     # plt.subplot(1, 3, 1)
#     # plt.title("L_anom")
#     # plt.imshow(L_anom, cmap='gray')
#     # plt.axis('off')
#
#     # 左：原始灰度
#     plt.subplot(1, 3, 1)
#     plt.title("rgb")
#     plt.imshow(rgb)
#     plt.axis('off')
#
#     # 中：归一化灰度
#     plt.subplot(1, 3, 2)
#     plt.title("L_anom_norm")
#     plt.imshow(L_anom_norm, cmap='gray', vmin=0, vmax=1)
#     plt.axis('off')
#
#     # 右：热力图
#     plt.subplot(1, 3, 3)
#     plt.title("L_anom_norm (Red Heatmap)")
#     im = plt.imshow(L_anom_norm, cmap=red_map, vmin=0, vmax=1)
#     plt.axis('off')
#     plt.colorbar(im, fraction=0.046, pad=0.04, label='Anomaly Level')
#
#     plt.tight_layout()
#     plt.show()

# ##########333332. 画图
# fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
#
# # (1) 原图
# axes[0].imshow(rgb)
# axes[0].set_title('rgb')
# axes[0].axis('off')
#
# # (2) 归一化灰度
# axes[1].imshow(L_anom_norm, cmap='gray', vmin=0, vmax=1)
# axes[1].set_title('L_anom_norm')
# axes[1].axis('off')
#
# # (3) 热力图
# im = axes[2].imshow(L_anom_norm, cmap=red_map, vmin=0, vmax=1)
# axes[2].set_title('L_anom_norm (Red Heatmap)')
# axes[2].axis('off')
#
# # ---- 关键：把色标挂到整行子图右侧，而不侵占第 3 块 ----
# fig.colorbar(im, ax=axes.ravel().tolist(),           # 对齐整行
#              fraction=0.025, pad=0.02,               # 大小/间距可微调
#              label='Anomaly Level')
#
# plt.show()



# if __name__ == "__main__":
#     img_path1 = r'F:\image_data\underwater_image_data\UIF\original_image\raw\224.bmp'
#     img_path2 = r'F:\image_data\underwater_image_data\UIF\original_image\raw\188.bmp'
#     img_path3 = r'F:\image_data\underwater_image_data\UIF\original_image\raw\988.bmp'
#     res = compute_luminance_anomaly(img_path)
#
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import LinearSegmentedColormap
#     import numpy as np
#
#     # 自定义 黑→红→黄 的 colormap（也可以用 plt.cm.hot / plt.cm.magma）
#     red_map = LinearSegmentedColormap.from_list(
#         'red_map',
#         [(0.0, (0, 0, 0)),      # 黑
#          (0.6, (1, 0, 0)),      # 红
#          (1.0, (1, 1, 0))],     # 黄
#         N=256
#     )
#
#     L_anom      = res['L_anom']
#     L_anom_norm = res['L_anom_norm']
#     print("L_anom.shape:",L_anom.shape)
#     print("L_anom_norm.shape:", L_anom_norm.shape)
#     print('L_anom      - min:', L_anom.min(), ' max:', L_anom.max())
#     print('L_anom_norm - min:', L_anom_norm.min(), ' max:', L_anom_norm.max())
#
#
#     plt.figure(figsize=(15, 5))
#     # 左：原始灰度
#     plt.subplot(1, 3, 1)
#     plt.title("L_anom")
#     plt.imshow(L_anom, cmap='gray')
#     plt.axis('off')
#
#     # 中：归一化灰度
#     plt.subplot(1, 3, 2)
#     plt.title("L_anom_norm")
#     plt.imshow(L_anom_norm, cmap='gray', vmin=0, vmax=1)
#     plt.axis('off')
#
#     # 右：热力图
#     plt.subplot(1, 3, 3)
#     plt.title("L_anom_norm (Red Heatmap)")
#     im = plt.imshow(L_anom_norm, cmap=red_map, vmin=0, vmax=1)
#     plt.axis('off')
#     plt.colorbar(im, fraction=0.046, pad=0.04, label='Anomaly Level')
#
#     plt.tight_layout()
#     plt.show()

################################# 出图  ######################################################################################
#------- 1. 三张图片路径 -------
# img_paths = [
#     r'/home/sunhao/SUNHAO/UIQA/Dataset/224.bmp',
#     r'/home/sunhao/SUNHAO/UIQA/Dataset/188.bmp',
#     r'/home/sunhao/SUNHAO/UIQA/Dataset/988.bmp' ]

# # ------- 2. 自定义黑→红→黄 colormap -------
# red_map = LinearSegmentedColormap.from_list(
#     'red_map',
#     [(0.0, (0, 0, 0)),      # 黑
#      (0.6, (1, 0, 0)),      # 红
#      (1.0, (1, 1, 0))],     # 黄
#     N=256
# )
#
# # ------- 3. 创建 3×3 画布 -------
# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15),
#                          constrained_layout=True)

# ------- 4. 逐张图片可视化 -------
# for row, path in enumerate(img_paths):
#     res = compute_luminance_anomaly(path)
#     rgb          = res['rgb']
#     L_anom_norm  = res['L_anom_norm']
#
#     # (1) RGB
#     axes[row, 0].imshow(rgb)
#     axes[row, 0].set_title(f'RGB {row+1}')
#     axes[row, 0].axis('off')
#
#     # (2) 灰度异常
#     axes[row, 1].imshow(L_anom_norm, cmap='gray', vmin=0, vmax=1)
#     axes[row, 1].set_title('L_anom_norm')
#     axes[row, 1].axis('off')
#
#     # (3) 热力图（记录 handle 以便后面挂色标）
#     im = axes[row, 2].imshow(L_anom_norm, cmap=red_map, vmin=0, vmax=1)
#     axes[row, 2].set_title('Heatmap')
#     axes[row, 2].axis('off')
#
# # ------- 5. 统一色标放最右侧 -------
# #   ax=axes[:, 2]  →  只沿着 3 个热力图所在的那一列对齐
# # fig.colorbar(im, ax=axes[:, 2], fraction=0.025, pad=0.02,
# #              label='Anomaly Level')
# #
# # plt.show()
#
# fig, axes = plt.subplots(
#     nrows=3, ncols=3, figsize=(9, 9),
#     gridspec_kw=dict(wspace=0.01,   # ← 列间距   (0~1，越小越贴近)
#                      hspace=0.03)   # ← 行间距
# )
# plt.show()


##################################  第一种图标 ##################################################################
# fig, axes = plt.subplots(
#     nrows=3, ncols=3, figsize=(9, 9),
#     gridspec_kw={'wspace': 0.03, 'hspace': 0.08}  # ← 列/行间距
# )
#
# # ------- 2. 逐张图片可视化 -------
# for row, path in enumerate(img_paths):
#     res = compute_luminance_anomaly(path)
#     rgb, L_norm = res['rgb'], res['L_anom_norm']
#
#     axes[row, 0].imshow(rgb)
#     axes[row, 0].set_title(f'RGB {row+1}')
#     axes[row, 0].axis('off')
#
#     axes[row, 1].imshow(L_norm, cmap='gray', vmin=0, vmax=1)
#     axes[row, 1].set_title('L_anom_norm')
#     axes[row, 1].axis('off')
#
#     im = axes[row, 2].imshow(L_norm, cmap=red_map, vmin=0, vmax=1)
#     axes[row, 2].set_title('Heatmap')
#     axes[row, 2].axis('off')
#
# # ------- 3. 全局色标挂在最右侧 -------
# fig.colorbar(im, ax=axes[:, 2], fraction=0.02, pad=0.01,
#              label='Anomaly Level')
#
# # （可选）再把整张图四周白边稍收一下
# fig.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.02)
#
# plt.show()

##################################  第2种图标 ##################################################################

# fig, axes = plt.subplots(
#     nrows=3, ncols=3, figsize=(9, 9),  # 保留3行3列
#     gridspec_kw={'wspace': 0.02, 'hspace': 0.08}  # 调整子图之间的间距
# )

# start_time = time.time()
# # ------- 逐张图片可视化 -------
# for row, path in enumerate(img_paths):
#     res = compute_luminance_anomaly(path)
#     rgb, L_norm = res['rgb'], res['L_anom_norm']

#     # 第一行图像的标题
#     if row == 0:
#         axes[row, 0].imshow(rgb)
#         axes[row, 0].set_title('RGB', fontsize=12)  # 只保留“RGB”作为标题
#         axes[row, 0].axis('off')

#         axes[row, 1].imshow(L_norm, cmap='gray', vmin=0, vmax=1)
#         axes[row, 1].set_title('L_anom_norm', fontsize=12)
#         axes[row, 1].axis('off')

#         im = axes[row, 2].imshow(L_norm, cmap=red_map, vmin=0, vmax=1)
#         axes[row, 2].set_title('Heatmap', fontsize=12)
#         axes[row, 2].axis('off')

#     # 其他行不加标题
#     else:
#         axes[row, 0].imshow(rgb)
#         axes[row, 0].axis('off')

#         axes[row, 1].imshow(L_norm, cmap='gray', vmin=0, vmax=1)
#         axes[row, 1].axis('off')

#         im = axes[row, 2].imshow(L_norm, cmap=red_map, vmin=0, vmax=1)
#         axes[row, 2].axis('off')

# # 计算时间
# end_time = time.time()
# elapsed_time = end_time - start_time

# # 输出计算时间
# print(f"Time taken for computation: {elapsed_time:.4f} seconds")

# # ------- 全局色标挂在最右侧 -------
# fig.colorbar(im, ax=axes[:, 2], fraction=0.02, pad=0.01,
#              label='Anomaly Level')

# # 调整图框的边距，确保所有图像都能完全显示
# fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.02)

# plt.show()






#=========================================================================================================================================================

# # 假设数据集A的路径
# dataset_a_path = r'/home/sunhao/SUNHAO/UIQA/Dataset/data_TEST/UWIQA_test'

# # 获取数据集A中的所有图像路径
# img_paths = [os.path.join(dataset_a_path, fname) for fname in os.listdir(dataset_a_path) if fname.endswith('.png')]

# # 存储图像路径与 L_norm 数据的映射
# l_norm_dict = {}


# start_time = time.time()
# # 逐张图像计算 L_norm
# for img_path in tqdm(img_paths, desc="Processing images"): 
#     res = compute_luminance_anomaly(img_path)  # 获取亮度异常度
#     file_name = os.path.basename(img_path)
#     L_norm = res['L_anom_norm']  # 获取亮度异常度图
#     l_norm_dict[file_name] = L_norm  # 使用图像路径作为键保存 L_norm

# # 保存字典为 .npy 文件
# save_path = r'/home/sunhao/SUNHAO/UIQA/Dataset/L_anom_nrom/UWIQA_test_L'  # 修改为你希望保存的位置
# np.save(save_path, l_norm_dict)

# print(f"L_norm data saved to {save_path}")

# # 计算时间
# end_time = time.time()
# elapsed_time = end_time - start_time

# # 输出计算时间
# print(f"Time taken for computation: {elapsed_time:.4f} seconds")





# red_map = LinearSegmentedColormap.from_list(
#     'red_map',
#     [(0.0, (0, 0, 0)),      # 黑
#      (0.6, (1, 0, 0)),      # 红
#      (1.0, (1, 1, 0))],     # 黄
#     N=256
# )
# #loaded_data = np.load(r'/home/sunhao/SUNHAO/UIQA/Dataset/L_TEST_RESULT.npy', allow_pickle=True).item()
loaded_data = np.load(r'/home/sunhao/SUNHAO/UIQA/Dataset/L_anom_nrom/UWIQA_test_L.npy', allow_pickle=True).item()


# num_entries = len(loaded_data)

# # 打印数据量
# print(f"The .npy file contains {num_entries} entries.")
#print(loaded_data.keys())  # 输出所有图像名

# 获取字典中的第一个路径（图像路径）
# first_img_path = list(loaded_data.keys())[0]  # 获取第一个图像路径
# print("path:",first_img_path)
# 获取对应的 L_norm 数据
#L_norm = loaded_data[first_img_path]

image_name = '0010.png'
L_norm = loaded_data.get(image_name)
print("L_norm.shape:",L_norm.shape)
# if torch.is_tensor(L_norm):
#     print("类型: torch.Tensor")
# elif isinstance(L_norm, np.ndarray):
#     print("类型: numpy.ndarray")
# else:
#     print(type(L_norm))

# print("dtype:", L_norm.dtype)              # 例如 float32
# print("shape:", L_norm.shape)



# # 打印出 L_norm 数据的规模
# #print(f"L_norm shape for the first image ({first_img_path}): {L_norm.shape}")
# print(f"L_norm shape for the first image ({image_name}): {L_norm.shape}")

# # 绘制热图
# plt.imshow(L_norm, cmap=red_map, vmin=0, vmax=1)
# plt.colorbar()
# #plt.title(f"Luminance Anomaly Heatmap for {first_img_path}")
# plt.show()







# loaded_data = np.load(r'/home/sunhao/SUNHAO/UIQA/Dataset/L_anom_nrom/L_UWIQA.npy', allow_pickle=True).item()

# new_data = {}

# # 遍历原始字典，将键转换为文件名
# for key, value in loaded_data.items():
#     file_name = os.path.basename(key)  # 提取文件名
#     new_data[file_name] = value  # 使用文件名作为新的键

# save_path = r'/home/sunhao/SUNHAO/UIQA/Dataset/L_anom_nrom/UWIQA_L'  # 修改为你希望保存的位置
# np.save(save_path, new_data)









# dataset_a_path = r'/home/sunhao/SUNHAO/UIQA/Dataset/SAUD2.0/SAUD2.0_dataset'

# # 获取数据集A中的所有图像路径（递归遍历子文件夹）
# img_paths = []
# for root, _, files in os.walk(dataset_a_path):
#     for file in files:
#         if file.endswith('.png'):
#             img_paths.append(os.path.join(root, file))

# # 存储图像名与 L_norm 数据的映射
# l_norm_dict = {}

# start_time = time.time()

# # 逐张图像计算 L_norm
# for img_path in tqdm(img_paths, desc="Processing images"): 
#     res = compute_luminance_anomaly(img_path)  # 获取亮度异常度
#     file_name = os.path.basename(img_path)  # 获取文件名
#     L_norm = res['L_anom_norm']  # 获取亮度异常度图
#     l_norm_dict[file_name] = L_norm  # 使用图像名作为键保存 L_norm

# # 保存字典为 .npy 文件
# save_path = r'/home/sunhao/SUNHAO/UIQA/Dataset/L_anom_nrom/SAUD_L.npy'  # 修改为你希望保存的位置
# np.save(save_path, l_norm_dict)

# print(f"L_norm data saved to {save_path}")

# # 计算时间
# end_time = time.time()
# elapsed_time = end_time - start_time

# # 输出计算时间
# print(f"Time taken for computation: {elapsed_time:.4f} seconds")