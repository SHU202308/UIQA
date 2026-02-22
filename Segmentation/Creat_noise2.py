import os
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils

# ========== 路径与参数 ==========
JSON_PATH = r'F:\pycharm_practice\WaterMask-master\dataset\UIIS\UIIS\UDW\annotations\train.json'
IMG_DIR   = r'F:\pycharm_practice\WaterMask-master\dataset\UIIS\UIIS\UDW\train'
SAVE_DIR = r'F:\pycharm_practice\WaterMask-master\dataset\UIIS\UIIS\UDW'
IMG_NAME  = 'XL_185.jpg'      # 只写文件名
TARGET_CAT_ID = 2              # 目标类别(你的 1: fish)

# 噪声与可视化参数
SIGMA   = 12.0                 # 高斯噪声强度(标准差, 0~255)
ALPHA   = 4.0                  # 总体倍率
FEATHER = 1.5                  # 掩膜边缘羽化sigma(像素, 0=硬边)
RNG_SEED = 0                   # 随机种子；设为 None 则每次不同

# 预览窗口最大尺寸（只缩小不放大）
MAX_W, MAX_H = 1280, 800

# ========== 工具函数 ==========
def imread_color(path):
    """更稳的读图方式，兼容中文/空格路径"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise FileNotFoundError(f'无法读取图像: {path}')
    return img

def ann_to_mask(ann, h, w):
    """annotation -> 二值掩膜(bool)，兼容 Polygon 与 RLE"""
    seg = ann['segmentation']
    if isinstance(seg, dict) and 'counts' in seg:   # RLE
        m = maskUtils.decode(seg).astype(bool)
    else:                                           # 多段多边形
        rles = maskUtils.frPyObjects(seg, h, w)
        m = maskUtils.decode(rles)
        if m.ndim == 3:
            m = np.any(m, axis=2)
        m = m.astype(bool)
    return m

# ========== 读 JSON / 找图片 ==========
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    coco = json.load(f)
id2name = {c['id']: c.get('name', f'class_{c["id"]}') for c in coco.get('categories', [])}

# 兼容 file_name 可能带子目录/大小写差异
key = os.path.basename(IMG_NAME).lower()
cands = [im for im in coco['images']
         if os.path.basename(im.get('file_name','')).lower() == key
         or im.get('file_name','').lower().endswith('/'+key)
         or im.get('file_name','').lower().endswith('\\'+key)]
if not cands:
    raise RuntimeError(f'在 {os.path.basename(JSON_PATH)} 里找不到图片: {IMG_NAME}')

img_rec = cands[0]
H, W, image_id = img_rec['height'], img_rec['width'], img_rec['id']

anns = [a for a in coco['annotations'] if a['image_id'] == image_id]

# 这张图包含的类别（可选打印）
cls_ids = sorted({a['category_id'] for a in anns})
print('This image contains categories:', {cid: id2name.get(cid, '') for cid in cls_ids})

# ========== 合成目标类别的掩膜 ==========
roi = np.zeros((H, W), dtype=bool)
for a in anns:
    if a['category_id'] == TARGET_CAT_ID:
        roi |= ann_to_mask(a, H, W)

if not roi.any():
    print(f'该图不含类别 {TARGET_CAT_ID} ({id2name.get(TARGET_CAT_ID,"")})，退出。')
    raise SystemExit

# ========== 读原图并对齐尺寸 ==========
img_path = os.path.join(IMG_DIR, IMG_NAME)
img = imread_color(img_path)  # BGR
ih, iw = img.shape[:2]
if (ih, iw) != (H, W):
    print('Warning: image size differs from JSON; resize mask to image size.')
    roi = cv2.resize(roi.astype(np.uint8), (iw, ih), interpolation=cv2.INTER_NEAREST).astype(bool)
    H, W = ih, iw

# ========== 在 ROI 内加噪 ==========
imgf = img.astype(np.float32)

mask = roi.astype(np.float32)
if FEATHER > 0:
    mask = cv2.GaussianBlur(mask, (0, 0), FEATHER)   # 软掩膜
mask = mask[..., None]                                # HxWx1

rng = np.random.default_rng(RNG_SEED)
noise = rng.normal(0, SIGMA, imgf.shape)              # BGR 各通道高斯噪声
out = np.clip(imgf + ALPHA * noise * mask, 0, 255).astype(np.uint8)

# ========== 预览：原图 vs 加噪（自动缩放；手动关闭窗口后继续）==========
win = 'Preview (close window to save)'
side = np.hstack([img, out])
cv2.putText(side, 'Original', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.putText(side, f'Noised (cat={TARGET_CAT_ID})', (img.shape[1]+10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

h, w = side.shape[:2]
scale = min(MAX_W / w, MAX_H / h, 1.0)   # 只缩小不放大
disp = side if scale >= 1.0 else cv2.resize(
    side, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA
)

cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
cv2.imshow(win, disp)
while cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1:
    cv2.waitKey(50)
cv2.destroyAllWindows()

# ========== 关闭后保存（原分辨率）==========
base = os.path.splitext(IMG_NAME)[0]
out_path  = os.path.join(SAVE_DIR, f'{base}_noised_cat{TARGET_CAT_ID}.jpg')
mask_path = os.path.join(SAVE_DIR, f'{base}_cat{TARGET_CAT_ID}_mask.png')
cv2.imwrite(out_path,  out)
cv2.imwrite(mask_path, (roi.astype(np.uint8) * 255))
print('Saved:', out_path, mask_path)