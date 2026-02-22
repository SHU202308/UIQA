import os
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils

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
N_PIX         = 49000        # 每个 ROI 中加噪像素数（等面积）
TARGET_PSNR   = 12.0        # 三张加噪图在各自 ROI 内对齐到的 PSNR(dB)，强度一致
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
blob_reef = sample_compact_blob(roi_reef,     N_PIX, rng)
blob_bg   = sample_compact_blob(roi_seafloor, N_PIX, rng)

# ========= 生成三张“ROI-PSNR 对齐”的加噪图 =========
noised_fish = add_awgn_in_blob(img, blob_fish, TARGET_PSNR, rng, FEATHER)
noised_reef = add_awgn_in_blob(img, blob_reef, TARGET_PSNR, rng, FEATHER)
noised_bg   = add_awgn_in_blob(img, blob_bg,   TARGET_PSNR, rng, FEATHER)

# ========= 拼 2×2 面板：左上原图 | 右上鱼 | 左下珊瑚 | 右下背景 =========
panel_t = np.hstack([img.copy(), noised_fish.copy()])
panel_b = np.hstack([noised_reef.copy(), noised_bg.copy()])
panel   = np.vstack([panel_t, panel_b])

label_on(panel, 'Original', (10, 30))
label_on(panel, 'Fish (PSNR=%.1fdB, %d px)' % (TARGET_PSNR, int(blob_fish.sum())),
         (img.shape[1] + 10, 30))
label_on(panel, 'Reefs (PSNR=%.1fdB, %d px)' % (TARGET_PSNR, int(blob_reef.sum())),
         (10, img.shape[0] + 30))
label_on(panel, 'Background (PSNR=%.1fdB, %d px)' % (TARGET_PSNR, int(blob_bg.sum())),
         (img.shape[1] + 10, img.shape[0] + 30))

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