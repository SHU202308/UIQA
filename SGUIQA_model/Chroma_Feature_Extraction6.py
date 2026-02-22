import torch
from image_SLIC4 import image_slic
from skimage import io, color, transform
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
import math
from torch_geometric.utils import add_self_loops
import os, json, numpy as np
#from torch_geometric.data import Data

"""
用于提取图像块的邻接关系索引，图像块连接边的特征（颜色相似度、色度比值、图像块像素中心点距离）
"""

class RGBLabDataset(Dataset):
    """
    读一张图 -> 返回
        rgb_tensor :   [3,H,W]  float32 归一化 (供 CNN / ViT)
        lab_tensor :   [3,H,W]  float32 L0-100, a/b-128~127 (供 SLIC 或其他 Lab 处理)
        path       :   原图路径
    """
    def __init__(self, img_dir, size=(256, 256)):
        self.paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, f))
        ])
        self.size = size

        # # torch 侧 RGB 预处理：Resize→ToTensor→Normalize
        # self.rgb_tf= transforms.Compose([
        #     transforms.Resize(rgb_size),
        #     transforms.ToTensor(),                     # uint8 0-255 → float32 0-1
        # ])

        # ---------- 2) Resize 到固定大小 (H, W) ----------
        # preserve_range=True 保留 0-255 原始值；anti_aliasing=True 抗锯齿

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

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

        return rgb_tensor, lab_tensor, lab_norm_ts,path
####################################################################################################################


#======================================= 基于SLIC结果构建GAT的特征 ================================================
# --- 内部缓存（你无需关心） ---
_INDEX_CACHE = {}                 # {shards_dir: key2shard}
_SHARD_CACHE = {}                 # {(shards_dir, sid): pack_dict}
_SHARD_ORDER = []                 # LRU 顺序
def _get_index(shards_dir: str):
    idx = _INDEX_CACHE.get(shards_dir)
    if idx is None:
        with open(os.path.join(shards_dir, "index.json"), "r", encoding="utf-8") as f:
            idx = json.load(f)["key2shard"]
        _INDEX_CACHE[shards_dir] = idx
    return idx

def _get_shard_pack(shards_dir: str, sid: int, max_cached_shards: int):
    k = (shards_dir, sid)
    if k in _SHARD_CACHE:
        # touch (LRU)
        _SHARD_ORDER.remove(k)
        _SHARD_ORDER.append(k)
        return _SHARD_CACHE[k]
    # 读盘
    sp = os.path.join(shards_dir, f"shard_{sid:04d}.pt")
    pack = torch.load(sp, map_location="cpu")  # dict: {key: sample_dict}
    if max_cached_shards > 0:
        _SHARD_CACHE[k] = pack
        _SHARD_ORDER.append(k)
        # 淘汰
        while len(_SHARD_ORDER) > max_cached_shards:
            old = _SHARD_ORDER.pop(0)
            _SHARD_CACHE.pop(old, None)
    return pack

def _to_edge_index_2E(ei):
    # 返回 torch.LongTensor，形状 (2, E)
    if not isinstance(ei, torch.Tensor):
        ei = torch.as_tensor(ei, dtype=torch.long)
    if ei.ndim != 2:
        raise ValueError(f"edge_index ndim must be 2, got {ei.ndim}")
    if ei.shape[0] == 2:
        return ei.contiguous()
    if ei.shape[1] == 2:
        return ei.t().contiguous()
    raise ValueError(f"bad edge_index shape {tuple(ei.shape)}")

def _segments_to_np(seg):
    if isinstance(seg, torch.Tensor):
        return seg.cpu().numpy().astype(np.int32, copy=False)
    return np.asarray(seg, dtype=np.int32)

def _csr_to_label_positions(pos_idx, pos_ptr):
    if isinstance(pos_idx, torch.Tensor): pos_idx = pos_idx.cpu().numpy()
    if isinstance(pos_ptr, torch.Tensor): pos_ptr = pos_ptr.cpu().numpy()
    L = int(pos_ptr.size - 1)
    return [pos_idx[pos_ptr[i]:pos_ptr[i+1]].copy() for i in range(L)]

def _get_label_positions(samp: dict):
    lp = samp.get("label_positions", None)
    if lp is not None:
        return [np.asarray(a) for a in lp]
    # 否则用 CSR 还原
    pos_idx, pos_ptr = samp.get("pos_idx"), samp.get("pos_ptr")
    if pos_idx is None or pos_ptr is None:
        raise RuntimeError("neither label_positions nor pos_idx/pos_ptr present in sample")
    return _csr_to_label_positions(pos_idx, pos_ptr)

def load_triplet_from_key(shards_dir: str, key: str, *, max_cached_shards: int = 2):
    """
    输入:  shards_dir（分片目录），key（index.json 里的那个键）
    返回:  image_patch_connection_list(torch.Long (2,E)),
          label_positions(list[np.ndarray]),
          segments(np.int32 [H,W]),
          meta(dict)
    """
    key2shard = _get_index(shards_dir)
    sid = key2shard[key]
    pack = _get_shard_pack(shards_dir, sid, max_cached_shards)
    samp = pack[key]

    edge_index_2E = _to_edge_index_2E(samp["edge_index"])
    label_positions = _get_label_positions(samp)
    segments = _segments_to_np(samp["segments"])
    meta = samp.get("meta", {})
    return edge_index_2E, label_positions, segments, meta


def batch_image_slic_data(LAB_images_Fedge,X_output1,shards_dir,keys):
    batch_size = len(keys)
    batch_block_feature = []
    batch_ab_feature = []
    batch_edge_index = []
    batch_edge_attr = []
    batch_segments = []
    for i in range(batch_size):  #批量大小
        key = keys[i]
        #image_ab0 = image_Lab.permute(2, 0, 1)   #把原来的LAB数据结构从H*W*C更改为C*H*W,以便计算且符合张量习惯
        image_ab0 = LAB_images_Fedge[i]
        image_cnn_feature = X_output1[i]
        #image_patch_connection_list, label_positions, segments = image_slic(image_Lab)
        image_patch_connection_list, label_positions, segments, meta=load_triplet_from_key(shards_dir, key,  max_cached_shards= 3)
        #KEYS = list(label_positions.keys())
        # print("keys:",keys)
        # print("label_positions:",label_positions)
        block_feature = []
        ab_feature = []
        x_y = []
        N = len(label_positions)
        # print("N:", N)
        CN = image_patch_connection_list.shape[1]  # CN = connection number ，因为image_patch_connection_list是2*E的结构
        dwtz = torch.zeros(CN, 5)
        # 构建节点信息
        for j in range(N):
            #KEY = KEYS[j]
            block_feature0 = torch.zeros(X_output1.shape[-3:]) # 构建一个和每个图像CNN张量大小一致的元素为0的数据结构
            ab_feature0 = torch.zeros(2, 224, 224)  # Lab空间中的ab值
            position = label_positions[j] #选取一个键值对应的图像块像素位置
            num = len(position)  # 该图像块像素总数
            sum_x = 0
            sum_y = 0
            for x, y in position:
                block_feature0[:, x, y] += image_cnn_feature[:, x, y]  # 把0元素阵与对应图像块像素位置的CNN结果相加
                ab_feature0[-2:, x, y] += image_ab0[-2:, x, y]
                sum_x += x
                sum_y += y
            average_x = sum_x / num
            average_y = sum_y / num
            patch_xy= [average_x , average_y]  # 图像块的中心像素坐标
            block_feature1 = block_feature0.sum(dim=(1, 2)) / num   # 在某个图像块每个通道上把所有元素相加后除以该图像块像素总数形成一个行向量
            block_feature.append(block_feature1) # 每个图像块有一个CNN特征向量，利用列表收集向量
            ab_feature1 = ab_feature0.sum(dim=(1, 2)) / num  # 在某个图像块的a和b通道上把所有元素相加后除以该图像块像素总数形成一个行向量
            ab_feature.append(ab_feature1)  # 每个图像块有一个a和b均值构成的向量
            x_y.append(patch_xy)    # 把每个图像块的质心坐标拼接起来（图像块的x和y 的位置均值，位置信息从0开始）
        x_y = torch.tensor(x_y)
        #block_feature = np.array(block_feature)  # 把单个图像特征中的所有图像块对应的cnn向量列表转换为二维numpy数组
        block_feature = torch.stack(block_feature, dim=0)
        ab_feature = np.array(ab_feature)  # 把单个图像特征中的所有图像块对应的ab向量列表转换为二维数组
        magnitudes = np.linalg.norm(ab_feature, axis=1)   # 求解每个图像块对应的色度模长，由（a,b）计算
        ab_feature = np.hstack((ab_feature, magnitudes.reshape(-1, 1)))  # 把模长信息拼到第三列 即a\b\M
        ab_feature = torch.tensor(ab_feature)

        ############################  多维特征   ##################################333
        for k in range(image_patch_connection_list.shape[1]) :
            n1= image_patch_connection_list[0,k]
            n2 = image_patch_connection_list[1,k]
            vector1 = ab_feature[n1,:2]  # 目标块ab向量，这里取的是n1行的第一和第二列，分别表示目标块的平均ab值
            vector2 = ab_feature[n2,:2]  # 相邻块ab向量
            M1 = ab_feature[n1,2]  # 目标块ab色度值
            M2 = ab_feature[n2,2]  # 相邻块ab色度值
            zb1 = x_y[n1,:]
            zb2 = x_y[n2,:]
            cos_patch1_patch2 = np.dot(vector1, vector2) / (M1 * M2)
            cos = (1 + cos_patch1_patch2) / 2  # 把相似性映射在0-1区间
            chromaticity_bias = torch.abs(M1 - M2)  #目标块和相邻块直接的色度M的差
            d_patch = math.hypot(zb2[0] - zb1[0], zb2[1] - zb1[1]) #计算目标块和相邻块直接的欧氏距离
            dwtz[k,0] = cos
            dwtz[k,1] = chromaticity_bias
            dwtz[k,2] = d_patch
        max_value_M = torch.max(dwtz[:, 1])
        dwtz[:,3] = dwtz[:, 1] / max_value_M
        max_d = torch.max(dwtz[:, 2])
        dwtz[:, 4] = dwtz[:, 2] / max_d
        dwtz_selected = dwtz[:, [0, 3, 4]]

        # -------------------- 加入edge_index的自环以及多维特征------------------------------
        #print("image_patch_connection_list.shape",image_patch_connection_list.shape)
        edge_index_with_loops, edge_weight_with_loops = add_self_loops(image_patch_connection_list,num_nodes=N)
        # edge_index_with_loops = add_self_loops(image_patch_connection_list, num_nodes=N)
        #print("edge_index_with_loops.shape", edge_index_with_loops.shape)
        self_loop_attr = torch.tensor([[1.0, 0.0, 0.0]]).repeat(N, 1)
        edge_attr_with_loops = torch.cat([dwtz_selected, self_loop_attr], dim=0)

        # -------------------- 构建batch的相关量 -----------------------------------------
        batch_edge_attr.append(edge_attr_with_loops)
        batch_edge_index.append(edge_index_with_loops)
        batch_block_feature.append(block_feature)  # 沿着通道拼接的节点特征N*D，用于生成[batch,N,D]
        batch_ab_feature.append(ab_feature)  #沿着通道拼接的图像块a和b均值特征和M即N*3，用于生成[batch,N,3]，但是一个batch中N不同
        batch_segments.append(segments)     # 图像像素归属标记，如某个像素属于2号图像块，则对应像素位置就是2，其单个大小与特征图大小一致，每个segments都是256*256的矩阵

    return  batch_edge_index,batch_block_feature,batch_edge_attr,batch_segments


#-----------------------------------  因分割数量不同需对批量的节点数据等进行拼接 ---------------------------------------------
def batch_concatenate(batch_edge_index,batch_block_feature,batch_edge_attr):
    data_list = []
    num_nodes = []
    for feat, edge_index, edge_attr in zip(batch_block_feature, batch_edge_index, batch_edge_attr):
        n_i = feat.size(0)
        num_nodes.append(n_i)

        #  把它们装到 Data 里
        data_list.append(Data(x=feat,
                              edge_index=edge_index,
                              edge_attr=edge_attr))

    loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
    # 直接拿出第一个（也是唯一一个）Batch 对象
    batch = next(iter(loader))
    return batch

def Creat_batch_test(batch_block_feature, batch_edge_index, batch_edge_attr):
    data_list = []
    num_nodes = []
    for feat, edge_index, edge_attr in zip(batch_block_feature, batch_edge_index, batch_edge_attr):
        n_i = feat.size
        num_nodes.append(n_i)

        #  把它们装到 Data 里
        data_list.append(Data(x=feat,
                              edge_index=edge_index,
                              edge_attr=edge_attr))

    loader = GeometricDataLoader(data_list, batch_size=len(data_list), shuffle=False)
    # 直接拿出第一个（也是唯一一个）Batch 对象
    batch = next(iter(loader))
    return batch



#------------------------------------测试batch_image_slic_data -------------------------------------------------------
#
# # 创建 Dataset 实例
# dataset = RGBLabDataset(img_dir=r'F:\image_data\underwater_image_data\UIF\original_image\raw')
# # 创建 DataLoader 实例
# dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=0)
#
# # 遍历 DataLoader 加载图像
# for i, (rgb_tensor, lab_tensor, lab_norm_ts,filenames) in enumerate(dataloader):
#     if i == 1:  # 获取第二个批次的数据
#         rgb_tensor_1 = rgb_tensor
#         print('rgb_tensor_1 shape:', rgb_tensor_1.shape)
#         lab_tensor_1 = lab_tensor
#         print('lab_tensor_1 shape:', lab_tensor_1.shape)
#         lab_norm_ts_1 = lab_norm_ts
#         print('lab_norm_ts_1 shape:', lab_norm_ts_1.shape)
#         # 2) 打印这个 batch 中的所有图像路径
#         print(f"第 {i + 1} 个 batch 的图像路径：")
#         for p in filenames:
#             print("  ", p)
#         break  # 获取到第二个批次后，退出循环
#
#
# print("RGB数据:",rgb_tensor_1 )
# print("SLIC用的LAB数量:",lab_tensor_1 )
# print("网络用的归一化LAB数量:", lab_norm_ts_1)
#
#
# X_output1 = torch.rand(5,16,256,256)*50
# print("X_output1.type:",type(X_output1))
# print("X_output1.shape:",X_output1.shape)
#
# LAB_images = lab_tensor_1.permute(0, 2, 3, 1).cpu().numpy()  # 把数据转回[B,H,W,C]，且定为numpy
# LAB_images_Fedge = lab_tensor_1  # tensor数据，B,C,H,W
# Lab_norm = lab_norm_ts_1  # 卷积需要的归一化lab数据
# batch_edge_index, batch_block_feature, batch_edge_attr, batch_segments = batch_image_slic_data(LAB_images,LAB_images_Fedge,X_output1)
#
# # # AAA = batch_edge_index[0]
# # # print("AAA:",AAA)
# # print("图像块连接关系索引.shape:",batch_edge_index[0].shape)
# # print("图像块连接关系索引：",batch_edge_index[0])
# # print("节点特征.shape:",batch_block_feature[0].shape)
# # print("节点特征：",batch_block_feature[0])
# # print("边的特征.shape:",batch_edge_attr[0].shape)
# # print("边的特征：",batch_edge_attr[0])
# # print("像素标号索引.shape:",batch_segments[0].shape)
# # print("像素标号索引：",batch_segments[0])
#
#
#
# batch = Creat_batch_test(batch_block_feature,batch_edge_index,batch_edge_attr)
# print("ptr:", batch.ptr)
