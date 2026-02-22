import os, json, cv2, numpy as np
from pycocotools import mask as maskUtils

JSON_PATH = r'F:\pycharm_practice\WaterMask-master\dataset\UIIS\UIIS\UDW\annotations\train.json'
IMG_DIR    = r'F:\pycharm_practice\WaterMask-master\dataset\UIIS\UIIS\UDW\train'  # 按你的目录改
IMG_NAME   = 'XL_185.jpg'   # 要处理的那张图
TARGET_CAT_ID = 1            # 对“类别1”加噪（你前面打印的 1: fish）
SAVE_DIR = r'F:\pycharm_practice\WaterMask-master\dataset\UIIS\UIIS\UDW'

# ---- 读取 JSON 并建立类别映射 ----
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    coco = json.load(f)
id2name = {c['id']: c.get('name', f'class_{c["id"]}') for c in coco.get('categories', [])}

# ---- 找到这张图片的记录 ----
img_rec = next(im for im in coco['images'] if im['file_name'] == IMG_NAME)
H, W = img_rec['height'], img_rec['width']
image_id = img_rec['id']

# ---- 取出这张图的所有实例标注 ----
anns = [a for a in coco['annotations'] if a['image_id'] == image_id]

# 打印这张图“包含的类别”
cls_ids = sorted({a['category_id'] for a in anns})
print('This image contains categories:')
for cid in cls_ids:
    print(f'{cid}: {id2name.get(cid, "")}')

# ---- 构建“类别1”的像素级掩膜（可能多实例，统一合并）----
def ann_to_mask(a, h, w):
    seg = a['segmentation']
    if isinstance(seg, dict) and 'counts' in seg:         # RLE
        m = maskUtils.decode(seg).astype(bool)
    else:                                                  # 多边形
        rles = maskUtils.frPyObjects(seg, h, w)
        m = maskUtils.decode(rles)
        if m.ndim == 3:
            m = np.any(m, axis=2)
    return m

roi = np.zeros((H, W), dtype=bool)
for a in anns:
    if a['category_id'] == TARGET_CAT_ID:
        roi |= ann_to_mask(a, H, W)

# 如果这张图没有类别1，直接退出
if not roi.any():
    print(f'No pixels of category {TARGET_CAT_ID} ({id2name.get(TARGET_CAT_ID,"")}) in this image.')
    raise SystemExit

# ---- 读取原图（不要预先resize，否则与标注尺寸对不上）----
img = cv2.imread(os.path.join(IMG_DIR, IMG_NAME))  # BGR
if img is None:
    raise FileNotFoundError(os.path.join(IMG_DIR, IMG_NAME))
if (img.shape[0], img.shape[1]) != (H, W):
    print('Warning: image size differs from JSON; proceeding with JSON size.')

# ---- 在 ROI 内加随机高斯噪声（可选边缘羽化，减少硬边）----
sigma = 10.0
imgf  = img.astype(np.float32)

# 边缘羽化：把二值mask做轻微高斯模糊，得到0..1的“软掩膜”
soft = cv2.GaussianBlur(roi.astype(np.float32), (0,0), 1.5)  # σ=1.5
soft = np.clip(soft, 0, 1)[..., None]  # H×W×1

noise = np.random.normal(0, sigma, imgf.shape).astype(np.float32)

out = imgf.copy()
out = np.clip(out + noise * soft, 0, 255)


# === 先展示（手动关闭窗口后继续）===
win = 'original | noised'
side = np.hstack([img, out])                 # BGR 并排
cv2.putText(side, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.putText(side, 'Noised (cat=1)', (img.shape[1]+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.imshow(win, side)
# 放大一点，免得太小看不清
cv2.resizeWindow(win, side.shape[1] * 2, side.shape[0] * 2)

# 等你点窗口的“×”后才继续
while True:
    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
        break
    cv2.waitKey(50)
cv2.destroyAllWindows()


cv2.imwrite('noised_on_cat1.jpg', out.astype(np.uint8))     # 结果图
cv2.imwrite('cat1_mask.png', (roi.astype(np.uint8)*255))    # 掩膜可视化（可选）
print('Saved: noised_on_cat1.jpg, cat1_mask.png') 