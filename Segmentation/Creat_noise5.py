import os
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
from pytorch_msssim import ms_ssim as MS_SSIM

# ========== 路径与参数 ==========
JSON_PATH = r'F:\pycharm_practice\WaterMask-master\dataset\UIIS\UIIS\UDW\annotations\train.json'
IMG_DIR   = r'F:\pycharm_practice\WaterMask-master\dataset\UIIS\UIIS\UDW\train'
SAVE_DIR = r'F:\pycharm_practice\WaterMask-master\dataset\UIIS\UIIS\UDW'
IMG_NAME  = 'XL_3126.jpg'      # 只写文件名
TARGET_CAT_ID = 1              # 目标类别(你的 1: fish)

# ========= 实验参数（按需改）=========
CAT_FISH      = 1           # 鱼
CAT_REEF      = 2           # 珊瑚/礁体 reefs
CAT_SEAFLOOR  = 7           # sea-floor（若该图没有7，会用“非鱼非珊瑚”的其余区域或补集作为背景）
N_PIX         = 45000        # 每个 ROI 中加噪像素数（等面积）
TARGET_PSNR   = 10.0        # 三张加噪图在各自 ROI 内对齐到的 PSNR(dB)，强度一致
FEATHER       = 1.5         # 加噪块的边缘羽化sigma(像素)，0=硬边
RNG_SEED      = 0           # 随机种子，改为 None 则每次不同
MAX_W, MAX_H  = 1280, 800   # 预览窗口最大尺寸（只缩小不放大）

# ========= 工具函数 =========
def imread_color(path):
    """更稳的读图（支持中文路径）"""
    data = np.fromfile(path, dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise FileNotFoundError(f'无法读取图像: {path}')
    return img

def ann_to_mask(ann, h, w):
    """annotation -> bool 掩膜；兼容 RLE / 多段多边形"""
    seg = ann['segmentation']
    if isinstance(seg, dict) and 'counts' in seg:     # RLE
        m = maskUtils.decode(seg).astype(bool)
    else:                                             # 多段多边形
        rles = maskUtils.frPyObjects(seg, h, w)
        m    = maskUtils.decode(rles)
        if m.ndim == 3:
            m = np.any(m, axis=2)
        m = m.astype(bool)
    return m

def sample_compact_blob(mask_bool: np.ndarray, n_pixels: int, rng=None, max_tries=100) -> np.ndarray:
    """
    在二值掩膜内采样一个“近似连通、面积≈n_pixels”的紧凑块。
    失败时兜底为在 ROI 中随机采样 n_pixels 个点（可能较分散）。
    """
    rng = rng or np.random.default_rng()
    H, W = mask_bool.shape
    mask_u8 = mask_bool.astype(np.uint8)

    # 估算半径：π r^2 ≈ n_pixels
    r = int(np.sqrt(max(n_pixels, 1) / np.pi))
    r = max(r, 3)

    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return np.zeros_like(mask_bool, bool)

    for _ in range(max_tries):
        i = rng.integers(0, len(xs))
        cx, cy = int(xs[i]), int(ys[i])

        blob = np.zeros((H, W), np.uint8)
        cv2.circle(blob, (cx, cy), r, 1, -1)
        blob &= mask_u8
        area = int(blob.sum())

        if area < n_pixels:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            it = 0
            while area < n_pixels and it < 200:
                grown = (cv2.dilate(blob, ker, iterations=1) & mask_u8)
                if int(grown.sum()) == area:
                    break
                blob = grown
                area = int(blob.sum())
                it += 1

        if area > n_pixels:
            flat = blob.ravel()
            idx  = np.flatnonzero(flat)
            drop = rng.choice(idx, size=area - n_pixels, replace=False)
            flat[drop] = 0
            blob = flat.reshape(H, W)

        if blob.any():
            return blob.astype(bool)

    # 兜底：随机采样 n_pixels
    flat = mask_u8.ravel()
    idx  = np.flatnonzero(flat)
    k    = min(n_pixels, len(idx))
    choose = rng.choice(idx, size=k, replace=False)
    out = np.zeros_like(flat, np.uint8); out[choose] = 1
    return out.reshape(H, W).astype(bool)

def sample_compact_multi(mask_bool: np.ndarray, n_pixels: int, rng=None, max_blobs=999) -> np.ndarray:
    """
    在 mask_bool (True=ROI) 内，跨多个连通域累计采样“紧凑块”，
    直到总像素数≈ n_pixels。每个块内部连续，多个块的并集满足总量。
    - max_blobs：最多取多少个块（防止碎成太多小块）
    """
    rng = rng or np.random.default_rng()
    H, W = mask_bool.shape
    out  = np.zeros((H, W), bool)
    remain = int(n_pixels)

    if not mask_bool.any() or remain <= 0:
        return out

    # 1) 连通域分割（8连通）
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
    # 跳过背景 label=0，按面积从大到小排序
    comps = sorted([(i, int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, num)],
                   key=lambda x: x[1], reverse=True)

    blobs_used = 0
    for lbl, area in comps:
        if remain <= 0 or blobs_used >= max_blobs:
            break
        comp_mask = (labels == lbl)

        # 该连通域最多能提供的像素
        take = min(remain, area)
        if take <= 0:
            continue

        # 2) 在该连通域内取一个“紧凑块”（面积≈take）
        blob = sample_compact_blob(comp_mask, take, rng)
        k = int(blob.sum())
        if k == 0:
            continue

        # 3) 累加到结果里，并更新剩余
        out |= blob
        remain -= k
        blobs_used += 1

    # 4) 如果仍未满足像素数：在“剩余 ROI 未覆盖部分”里随机补齐（兜底）
    if remain > 0:
        rest = mask_bool & (~out)
        if rest.any():
            flat = rest.ravel()
            idx  = np.flatnonzero(flat)
            add  = min(remain, len(idx))
            pick = rng.choice(idx, size=add, replace=False)
            tmp  = np.zeros_like(flat, np.uint8); tmp[pick] = 1
            out |= tmp.reshape(H, W).astype(bool)

    return out

def soft_mask(mask_bool, sigma):
    """将 bool 掩膜转为 0..1 的软掩膜，并做高斯羽化（sigma=0 则返回硬边）"""
    m = mask_bool.astype(np.float32)
    if sigma and sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigma)
    m = np.clip(m, 0, 1)
    return m

def scale_for_target_psnr(noise_unit, roi3, target_psnr_db=30.0):
    target_mse = (255.0**2) / (10.0**(target_psnr_db/10.0))
    # —— 把 (H,W,1) 掩膜挤压成 (H,W)：
    roi2d = roi3.squeeze(-1).astype(bool) if roi3.ndim == 3 else roi3.astype(bool)

    if roi2d.sum() == 0:
        return 0.0  # 或者返回 1.0/直接抛错，按你需求
    mse0 = (noise_unit[roi2d]**2).mean() + 1e-12   # noise_unit[roi2d] 形状 (N,3)
    return np.sqrt(target_mse / mse0)

def add_awgn_in_blob(img, blob_bool, target_psnr_db, rng, feather_sigma=1.5):
    """
    仅在 blob_bool 区域内按 target_psnr_db 加性高斯噪声（ROI-PSNR 对齐），边缘可羽化。
    返回加噪后的 uint8 图像。
    """
    imgf  = img.astype(np.float32)
    roi3  = blob_bool[..., None]
    # 单位方差噪声
    noise_unit = rng.normal(0, 1.0, imgf.shape).astype(np.float32)
    scale = scale_for_target_psnr(noise_unit, roi3, target_psnr_db)
    msoft = soft_mask(blob_bool, feather_sigma)[..., None]
    out = np.clip(imgf + scale * noise_unit * msoft, 0, 255).astype(np.uint8)
    return out

def label_on(img, text, org=(10,30), color=(0,255,0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

# 生成图框及图名
def pt_to_px(pt, dpi):
    return int(round(pt * dpi / 72.0))

def _load_font(size_px, font_path=None):
    # 优先你传入的字体；找不到就退到系统常见无衬线
    candidates = [
        font_path,
        r"C:\Windows\Fonts\arial.ttf",                      # Windows
        "/Library/Fonts/Arial.ttf",                         # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"   # Linux
    ]
    for fp in candidates:
        if fp and os.path.exists(fp):
            return ImageFont.truetype(fp, size=size_px)
    return ImageFont.load_default()

def make_panel_2x2(a, b, c, d,
                   captions=('Original image','Distorted image A','Distorted image B','Distorted image C'),
                   gap=30, margin=30, caption_h=48,
                   bg=(255,255,255), text_color=(0,0,0),
                   caption_pt=9.0,             # ← 标题字号（pt）
                   final_width_inch=3.5,       # ← 预期放在文章里的宽度（英寸），单栏=3.5
                   caption_font_path=None):    # ← 可传 Arial/Helvetica 的路径

    H, W = a.shape[:2]
    row_gap = col_gap = gap
    panel_h = margin + (H + caption_h) + row_gap + (H + caption_h) + margin
    panel_w = margin + W + col_gap + W + margin
    canvas  = np.full((panel_h, panel_w, 3), bg, dtype=np.uint8)

    # 子图位置
    x1, y1 = margin,                         margin
    x2, y2 = margin + W + col_gap,           margin
    x3, y3 = margin,                         margin + (H + caption_h) + row_gap
    x4, y4 = margin + W + col_gap,           margin + (H + caption_h) + row_gap

    # 粘贴
    canvas[y1:y1+H, x1:x1+W] = a
    canvas[y2:y2+H, x2:x2+W] = b
    canvas[y3:y3+H, x3:x3+W] = c
    canvas[y4:y4+H, x4:x4+W] = d

    # 计算“最终宽度”对应的 dpi，把 pt -> px
    dpi = canvas.shape[1] / float(final_width_inch)
    caption_px = pt_to_px(caption_pt, dpi)
    font = _load_font(caption_px, caption_font_path)

    # 用 Pillow 居中绘制标题
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    im  = Image.fromarray(rgb)
    drw = ImageDraw.Draw(im)

    def put_caption(x, y, text):
        # 白色条带
        cv2.rectangle(canvas, (x, y+H), (x+W, y+H+caption_h), (255,255,255), -1)
        # 重新拿到更新后的底图
        # (注：条带是在 canvas 上画的，不影响文字绘制)
        bbox = drw.textbbox((0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        tx = x + (W - tw)//2
        ty = y + H + (caption_h - th)//2
        drw.text((tx, ty), text, font=font, fill=(0,0,0))

    cap1, cap2, cap3, cap4 = captions
    put_caption(x1, y1, cap1)
    put_caption(x2, y2, cap2)
    put_caption(x3, y3, cap3)
    put_caption(x4, y4, cap4)

    # 回到 BGR
    canvas = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return canvas


# ========= 读 JSON / 找图片 =========
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    coco = json.load(f)
id2name = {c['id']: c.get('name', f'class_{c["id"]}') for c in coco.get('categories', [])}

# 找到这张图（兼容 file_name 含子目录）
key = os.path.basename(IMG_NAME).lower()
cands = [im for im in coco['images']
         if os.path.basename(im.get('file_name','')).lower() == key
         or im.get('file_name','').lower().endswith('/'+key)
         or im.get('file_name','').lower().endswith('\\'+key)]
if not cands:
    raise RuntimeError(f'在 {os.path.basename(JSON_PATH)} 里找不到图片: {IMG_NAME}')
img_rec = cands[0]
H, W, image_id = img_rec['height'], img_rec['width'], img_rec['id']

# 该图的所有实例
anns = [a for a in coco['annotations'] if a['image_id'] == image_id]

# 合成类别掩膜
def union_mask_for_cats(cats):
    m = np.zeros((H, W), bool)
    for a in anns:
        if a['category_id'] in cats:
            m |= ann_to_mask(a, H, W)
    return m

roi_fish = union_mask_for_cats({CAT_FISH})
roi_reef = union_mask_for_cats({CAT_REEF})

# 背景优先取 sea-floor(7)，若没有则用“非鱼非珊瑚”的其余标注；再不行用“整图减去已标注”
roi_seafloor = union_mask_for_cats({CAT_SEAFLOOR})
if not roi_seafloor.any():
    other = np.zeros((H, W), bool)
    for a in anns:
        if a['category_id'] not in {CAT_FISH, CAT_REEF}:
            other |= ann_to_mask(a, H, W)
    roi_seafloor = other
if not roi_seafloor.any():
    any_mask = np.zeros((H, W), bool)
    for a in anns:
        any_mask |= ann_to_mask(a, H, W)
    roi_seafloor = ~any_mask

# 读原图并对齐（若图尺寸与 JSON 不一致，resize 掩膜）
img_path = os.path.join(IMG_DIR, IMG_NAME)
img = imread_color(img_path)
ih, iw = img.shape[:2]
if (ih, iw) != (H, W):
    roi_fish     = cv2.resize(roi_fish.astype(np.uint8), (iw, ih), interpolation=cv2.INTER_NEAREST).astype(bool)
    roi_reef     = cv2.resize(roi_reef.astype(np.uint8), (iw, ih), interpolation=cv2.INTER_NEAREST).astype(bool)
    roi_seafloor = cv2.resize(roi_seafloor.astype(np.uint8), (iw, ih), interpolation=cv2.INTER_NEAREST).astype(bool)
    H, W = ih, iw

# ========= 在三个 ROI 里各抽一个“紧凑块”，面积≈N_PIX =========
rng = np.random.default_rng(RNG_SEED)
blob_fish = sample_compact_blob(roi_fish,     N_PIX, rng)
blob_reef = sample_compact_multi(roi_reef, N_PIX, rng, max_blobs=5)
blob_bg   = sample_compact_blob(roi_seafloor, N_PIX, rng)

# ========= 生成三张“ROI-PSNR 对齐”的加噪图 =========
noised_fish = add_awgn_in_blob(img, blob_fish, TARGET_PSNR, rng, FEATHER)
noised_reef = add_awgn_in_blob(img, blob_reef, TARGET_PSNR, rng, FEATHER)
noised_bg   = add_awgn_in_blob(img, blob_bg,   TARGET_PSNR, rng, FEATHER)

# ========= 拼 2×2 面板：左上原图 | 右上鱼 | 左下珊瑚 | 右下背景 =========
# # 1.原始代码
# panel_t = np.hstack([img.copy(), noised_fish.copy()])
# panel_b = np.hstack([noised_reef.copy(), noised_bg.copy()])
# panel   = np.vstack([panel_t, panel_b])
#
# label_on(panel, 'Original', (10, 30))
# label_on(panel, 'Fish (PSNR=%.1fdB, %d px)' % (TARGET_PSNR, int(blob_fish.sum())),
#          (img.shape[1] + 10, 30))
# label_on(panel, 'Reefs (PSNR=%.1fdB, %d px)' % (TARGET_PSNR, int(blob_reef.sum())),
#          (10, img.shape[0] + 30))
# label_on(panel, 'Background (PSNR=%.1fdB, %d px)' % (TARGET_PSNR, int(blob_bg.sum())),
#          (img.shape[1] + 10, img.shape[0] + 30))

#2.对应新图框的def
# ========= 拼 2×2 面板（带空隙与底部标题）=========
panel = make_panel_2x2(
    img.copy(), noised_fish.copy(), noised_reef.copy(), noised_bg.copy(),
    captions=('Original image','Distorted image A','Distorted image B','Distorted image C'),
    gap=30, margin=30, caption_h=48,
    bg=(255,255,255),
    caption_pt=8.0,                  # ← 9 pt（最终尺寸）
    final_width_inch=3.5,            # ← 准备放单栏；双栏就填 7.16
    caption_font_path=None           # ← 如需固定 Arial 路径可在此填写
)

# ========= 预览（自动缩放；手动关闭后保存）=========
h, w = panel.shape[:2]
scale = min(MAX_W / w, MAX_H / h, 1.0)
disp  = panel if scale >= 1.0 else cv2.resize(panel, (int(w*scale), int(h*scale)), cv2.INTER_AREA)
cv2.namedWindow('Preview', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Preview', disp)
while cv2.getWindowProperty('Preview', cv2.WND_PROP_VISIBLE) >= 1:
    cv2.waitKey(50)
cv2.destroyAllWindows()

# ========= 保存 =========
os.makedirs(SAVE_DIR, exist_ok=True)
base = os.path.splitext(IMG_NAME)[0]
panel_path = os.path.join(SAVE_DIR, f'{base}_panel_psnr{int(TARGET_PSNR)}_npix{N_PIX}.jpg')
cv2.imwrite(panel_path, panel)
print('Saved panel to:', panel_path)


##------------------------------------- 计算指标 --------------------------------------------------------------------

# 1. 计算全图PSNR和SSIM
def full_psnr_ssim(ref, x):
    """全图 PSNR & SSIM（ref/x: H×W×3, uint8 或可转 uint8）"""
    ref8 = ref.astype(np.uint8)
    x8   = x.astype(np.uint8)
    p = psnr(ref8, x8, data_range=255)
    s = ssim(ref8, x8, channel_axis=-1, data_range=255)
    return p, s

# 2.计算SSIM和MS-SSIM
def ssim_and_msssim(ref, x, mask=None):
    """
    计算 SSIM 与 MS-SSIM（都越大越好）。
    - ref, x: HxWx3，uint8 或可转为 uint8；BGR/RGB一致即可
    - mask  : 可选 bool 掩膜；提供则只在掩膜内评估（掩膜外与ref对齐）
    返回: (ssim_val, msssim_val)
    """
    ref8 = ref.astype(np.uint8)
    x8   = x.astype(np.uint8).copy()

    if mask is not None:
        m = mask.astype(bool)
        x8[~m] = ref8[~m]  # 中和 ROI 外影响

    # SSIM（scikit-image）
    ssim_val = ssim(ref8, x8, channel_axis=-1, data_range=255)

    # MS-SSIM（pytorch-msssim）
    def _to_t(img_uint8):
        arr = (img_uint8.astype(np.float32) / 255.).transpose(2, 0, 1)[None, ...]  # 1x3xHxW
        return torch.from_numpy(arr)
    R = _to_t(ref8)
    X = _to_t(x8)
    with torch.no_grad():
        msssim_val = MS_SSIM(X, R, data_range=1.0, size_average=True).item()

    return ssim_val, msssim_val

# ---- 直接算 ----
print('=== Full-image metrics ===')
for name, y in [('A_fish', noised_fish), ('B_reef', noised_reef), ('C_bg', noised_bg)]:
    P, S = full_psnr_ssim(img, y)
    print(f'{name:8s}  PSNR={P:.3f} dB   SSIM={S:.4f}')

# print('\n=== ROI-PSNR (按各自加噪的 blob) ===')
# print('Fish ROI-PSNR:', full_psnr_ssim(img, noised_fish))
# print('Reef ROI-PSNR:', full_psnr_ssim(img, noised_reef))
# print('Back ROI-PSNR:', full_psnr_ssim(img, noised_bg))

print('\n=== ROI-PSNR (按各自加噪的 blob) ===')
print('Fish SSIM-MSSSIM:', ssim_and_msssim(img, noised_fish,mask=blob_fish))
print('Reef SSIM-MSSSIM:', ssim_and_msssim(img, noised_reef,mask=blob_reef))
print('Back SSIM-MSSSIM:', ssim_and_msssim(img, noised_bg,mask=blob_bg))

#-----------计算ROI-PSNR
def add_awgn_match_roi_psnr(ref_u8, roi_bool, target_psnr_db=10.0, feather_sigma=1.5, rng=None):
    """
    仅在 roi_bool 区域内加高斯噪声，并通过二分搜索让 ROI-PSNR 精准等于 target_psnr_db。
    返回：加噪后的 uint8 图像
    """
    rng  = rng or np.random.default_rng()
    ref  = ref_u8.astype(np.float32)
    msoft = soft_mask(roi_bool, feather_sigma)[..., None]   # HxWx1
    noise = rng.normal(0, 1.0, ref.shape).astype(np.float32)

    # 目标 MSE（PSNR 定义：10 log10(255^2 / MSE)）
    target_mse = (255.0**2) / (10.0**(target_psnr_db/10.0))

    def mse_for(alpha):
        y = np.clip(ref + alpha * noise * msoft, 0, 255)
        # 用与评估一致的权重（软掩膜）做加权 MSE
        n = msoft.sum() * 3.0
        return ((ref - y)**2 * msoft).sum() / (n + 1e-12)

    # 二分搜索 alpha
    lo, hi = 0.0, 200.0
    for _ in range(20):                        # 20次足够收敛到 <1e-3 dB
        mid = (lo + hi) / 2
        if mse_for(mid) > target_mse:
            hi = mid
        else:
            lo = mid

    y = np.clip(ref + lo * noise * msoft, 0, 255).astype(np.uint8)
    return y

def roi_psnr_soft(ref_u8, x_u8, w01):
    """
    软掩膜版本：w01 为 0..1 权重图（HxW），内部做加权MSE。
    """
    ref = ref_u8.astype(np.float32)
    x   = x_u8.astype(np.float32)
    w   = w01.astype(np.float32)[..., None]     # HxWx1

    n = w.sum() * 3.0
    if n <= 0:
        return float('inf')

    mse = ((ref - x) ** 2 * w).sum() / n
    return 10.0 * np.log10((255.0 ** 2) / (mse + 1e-12))

rng = np.random.default_rng(0)
noised_fish = add_awgn_match_roi_psnr(img, blob_fish, target_psnr_db=10.0, feather_sigma=FEATHER, rng=rng)
noised_reef = add_awgn_match_roi_psnr(img, blob_reef, target_psnr_db=10.0, feather_sigma=FEATHER, rng=rng)
noised_bg   = add_awgn_match_roi_psnr(img, blob_bg,   target_psnr_db=10.0, feather_sigma=FEATHER, rng=rng)

# 校验
print('Fish  ROI-PSNR:', roi_psnr_soft(img, noised_fish, soft_mask(blob_fish, FEATHER)))
print('Reef  ROI-PSNR:', roi_psnr_soft(img, noised_reef, soft_mask(blob_reef, FEATHER)))
print('Bg    ROI-PSNR:', roi_psnr_soft(img, noised_bg,   soft_mask(blob_bg,   FEATHER)))