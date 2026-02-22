import json
import os
from collections import Counter, defaultdict

p = r'F:\pycharm_practice\WaterMask-master\dataset\UIIS\UIIS\UDW\annotations\train.json'  # 或 train.json
d = json.load(open(p, 'r', encoding='utf-8'))
print(d.keys())                                    # 应该包含 images / annotations / categories
print('images:', len(d.get('images', [])))
print('annotations:', len(d.get('annotations', [])))
print('categories:', len(d.get('categories', [])))
print('first cats:', d.get('categories', [])[:5])  # 看前几条

with open(p, 'r', encoding='utf-8') as f:
    d = json.load(f)

cats = d.get('categories', [])

if cats:
    # 按 id 排序，只打印 “id: name”
    for c in sorted(cats, key=lambda x: int(x.get('id', 0))):
        print(f"{c['id']}: {c.get('name', '')}")
    print(f"num_classes = {len(cats)}")
else:
    # 若 JSON 里没有 categories，就反推出现过的 category_id
    ids = sorted({a['category_id'] for a in d.get('annotations', [])})
    print('No "categories" found. category_id list:', ids)


# 检查是否需要 pycocotools（是否含 RLE / iscrowd=1）
need = any(
    (isinstance(a.get('segmentation'), dict) and 'counts' in a['segmentation'])
    or a.get('iscrowd', 0) == 1
    for a in d.get('annotations', [])
)
print('need pycocotools:', need)

##---------------------------------------寻找某一图像的类标记--------------------------------------------------------------
# id <-> name 映射
id2name = {c['id']: c.get('name', str(c['id'])) for c in d.get('categories', [])}

def find_image_record(coco, file_name=None, image_id=None):
    """根据 file_name 或 image_id 找到 images 里的那条记录"""
    if image_id is not None:
        return next(im for im in coco['images'] if im['id'] == image_id)
    key = os.path.basename(file_name).lower()
    def norm(fn):  # 兼容子目录/大小写
        fn = os.path.basename(fn or '')
        return fn.lower()
    cands = [im for im in coco['images']
             if norm(im.get('file_name','')) == key
             or norm(im.get('file_name','')).endswith('/'+key)
             or norm(im.get('file_name','')).endswith('\\'+key)]
    if not cands:
        raise ValueError(f'找不到图片: {file_name}')
    return cands[0]

def categories_for_image(coco, *, file_name=None, image_id=None, with_counts=True, with_area=True):
    """返回该图包含的类别（可选：实例数和总标注面积）"""
    im = find_image_record(coco, file_name=file_name, image_id=image_id)
    iid = im['id']
    anns = [a for a in coco['annotations'] if a['image_id'] == iid]

    cats_counter = Counter(a['category_id'] for a in anns)
    area_sum = defaultdict(float)
    if with_area:
        for a in anns:
            area_sum[a['category_id']] += float(a.get('area', 0.0))

    # 组装输出（按 category_id 排序）
    rows = []
    for cid in sorted(cats_counter.keys()):
        rows.append({
            'category_id': cid,
            'category_name': id2name.get(cid, ''),
            'instances': cats_counter[cid] if with_counts else None,
            'area_sum': area_sum[cid] if with_area else None,
        })
    return im, rows

# ==== 使用示例 ====
img_name = 'XL_3126.jpg'       # XL_3126
im, rows = categories_for_image(d, file_name=img_name)
print(f"image_id={im['id']}  size={im['width']}x{im['height']}  file_name={im.get('file_name')}")
for r in rows:
    print(f"{r['category_id']:>2}: {r['category_name']}  "
          f"(instances={r['instances']}, area_sum={r['area_sum']:.1f})")