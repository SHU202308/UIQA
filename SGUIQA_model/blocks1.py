import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn import GATConv
import torchvision.transforms.functional as TF



#---------------------------------------Color_Branch-------------------------------------------------------------------
"""
此颜色特征提取分支对图像Lab空间的ab通道进行CNN卷积后，构造GAT捕捉局部图像块之间的关系，期间利用CNN结果构成节点特征，后对CNN及GAT特征进行融合
"""

class CustomBlock(nn.Module):
    """
    # 定义一个包含卷积、激活和池化的自定义模块
    """

    def __init__(self, in_channels, out_channels, kernel_size, act_layer):
        super(CustomBlock, self).__init__()

        # chorm  初级特征提取
        # Sequential是pytorch中的一种容器，可以将多个层连接起来，此处是把2维卷积层、激活函数、通道归一化连接起来
        self.fe = nn.Sequential(
            # Conv + LReLU + IN
            # A//B 表示的是除以后的结果取整   A/B会取小数
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=False),  #卷积层
            nn.InstanceNorm2d(out_channels),   # 归一化层
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),   # 激活函数
            act_layer
        )

    def forward(self, x):  # B 3 H W
        # FE
        x = self.fe(x)  # B C H W
        return x

class SElayer(nn.Module):
    """
    # 定义一个SE模块
    """
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)  # 会自适应地对输入特征的最后两个维度进行池化
        # self.linear1 是线性层1
        self.linear1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # self.linear2 是线性层2
        self.linear2 = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, X_input):
        b, c, _, _ = X_input.size()
        y = self.avg_pool(X_input)
        y = y.view(b, c)  # 把经过池化的批量的输入[batch_size, channels, 1, 1]变为[batch_size, channels]
        y = self.linear1(y)
        y = self.linear2(y)
        y = y.view(b, c, 1, 1)
        X_output1 = X_input * y.expand_as(X_input)
        return X_output1


#----------------------------------------多头GAT ---------------------------------------------------------------------
"""
使用多头GAT
多头GAT通常能带来更好泛化和更细致的特征表达
假设CNN的颜色特征是64维度，为了与CNN特征融合，则多头GAT维度也需要时64，假设有4个头，则每个头是16维度
其输入是把同批量图像的分割节点关系、CNN节点特征、额外的多维特征全部拼接进而输入，假设5张图，分别有10、8、6、4、2个10维度节点，则输入的节点就是30个
最后的输出也是30*10
"""

class EdgeEncoder(nn.Module):
    """
    边特征编码器：
    功能：把多维 edge_attr  (E × d_edge)→ 映射成 head-wise 张量 (E × heads × d_head)
    share=True : 所有 head 共用一套 MLP；输出后 reshape
    share=False: 每个 head 独立一套 MLP（表达力更强，参数更多）
    形参:
      d_edge  : 原始边特征维度  (int)
      d_head  : 每个注意力头想要的边特征维度 (int)
      heads   : 注意力头数量 K  (int)
      share   : True  → 所有头共享 1 套 MLP
                False → 每个头独立 1 套 MLP

    """
    def __init__(self, d_edge: int, d_head: int, heads: int, share: bool = True):
        super().__init__()
        self.share, self.heads, self.d_head = share, heads, d_head
        if share:   # 共享1套MLP
            self.mlp = nn.Sequential(
                nn.Linear(d_edge, heads * d_head, bias=False),
                nn.ReLU()
            )
        else:
            self.mlps = nn.ModuleList([
                nn.Sequential(nn.Linear(d_edge, d_head, bias=False),
                               nn.ReLU())
                for _ in range(heads)
            ])

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        # share: 先一次编码再 reshape；否则逐 head 编码后 stack
        if self.share:
            out = self.mlp(edge_attr).view(-1, self.heads, self.d_head)
        else:
            outs = [mlp(edge_attr) for mlp in self.mlps]       # 列表(E, d_head)
            out  = torch.stack(outs, dim=1)                    # (E, heads, d_head)
        return out




class HW_E_GATConv(MessagePassing):
    """
    Head-Wise Edge-aware GAT:
      └─ z_ij^k = [ W_k h_i  ||  W_k h_j  ||  P_ij^k ]
      └─ e_ij^k = LeakyReLU( a_k^T · z_ij^k )
      in_dim：节点输入的特征维度
      out_dim：节点输出维度
      edge_dim： 原始边的特征维度
      heads： 头数
      dropout：对注意力权重 α 的 dropout
      neg_slp：LeakyReLU 负斜率
      share_edge_mlp：边编码器是否共享一套 MLP（已在 EdgeEncoder 解释）
    """
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int,
                 heads: int = 4, dropout: float = 0.6,
                 negative_slope: float = 0.2, share_edge_mlp: bool = True):
        super().__init__(aggr='add')           # “加和”聚合
        self.heads, self.out_dim = heads, out_dim
        self.dropout, self.neg_slp = dropout, negative_slope

        # 节点线性映射 W_k
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        # 边特征编码器
        self.edge_enc = EdgeEncoder(edge_dim, out_dim, heads,
                                    share=share_edge_mlp)
        # 初始化边计算注意力值的时候的aT，因为是有两个边和多维特征拼接，所以是3* out_dim，有多个头，数据规模heads × (3×out_dim)   每个边进行注意力计算都是用这个aT
        self.att = nn.Parameter(torch.empty(heads, 3 * out_dim))
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_attr):
        """
        x          : (N_total, in_dim)         节点特征
        edge_index : (2, E_total)              有向边索引
        edge_attr  : (E_total, edge_dim)       多维边特征
        """
        N = x.size(0)
        # 1) 节点 → 多头表示  (N, heads, D)
        x = self.lin(x).view(N, self.heads, self.out_dim)
        # 2) 边特征 → 多头表示 (E, heads, D)
        p = self.edge_enc(edge_attr)

        # 3) 调用 propagate 进入 message/aggregate
        # propagate 中会自动根据输入的x和edge_index 把图像节点信息根据edge_index重新划分为目标节点特征和邻接节点特征，然后对message模块进行修改，在x_i和x_j的基础上再拼接边的多维特征
        # aggr_out = self.propagate(edge_index, x=x, edge_attr=p)  # 返回 (N, heads, D)
        # # return out.view(N, self.heads * self.out_dim)       # 拼接 => (N, heads·D)
        # return aggr_out       # 拼接 => (N, heads·D)
        return self.propagate(edge_index, x=x, edge_attr=p)   # 此时的数据结构是(N, heads, D)，会自动生成aggr_out并传递给update进行数据重塑得到(N, heads·D)

    # ---------- message ----------
    def message(self, x_i, x_j, edge_attr, index):
        """
        x_i / x_j : (E, heads, D)     目标 / 源 节点特征；E表示本批次所有图拼在一起后，edge_index 中的列数
        edge_attr : (E, heads, D)     编码后边的多维特征
        index     : (E)               目标节点 dst，用于分组 softmax
        """
        # 拼接  → (E, heads, 3D)    三部分是在通道上拼接
        z = torch.cat([x_j, x_i, edge_attr], dim=-1)
        # 点积 (broadcast) → (E, heads)
        e = (z * self.att.unsqueeze(0)).sum(-1)   # 逐元素做点积并求和    （E*heads)
        e = F.leaky_relu(e, negative_slope=self.neg_slp)  # （E*heads)

        # softmax 按 dst 分组（不同图已偏移，不会混组）
        # 基于GCN的特殊softmax，index是edge_index的第二行即目标节点（第一行是源节点），会根据目标节点来自动分组e，然后对相同目标节点的e进行softmax归一化
        # 假设目标节点是1，现在相关的有3条边，头数是4，所以就是对这3*4规模的数据在每个头上进行归一化，即对每一列的数据进行一次归一化
        # 归一化结束后，其数据还是按照之前分组时候e的序号进行填充，例如上例中3条边在e的序号分别是e2、e5、e6（从e0开始），那么归一化后a2、a5、a6就对应着归一化的数据
        α = softmax(e, index)
        α = F.dropout(α, p=self.dropout, training=self.training)   # （E*heads)

        # 返回加权的邻居特征   (E, heads, D)
        # α.unsqueeze(-1) 是在α的最后一个维上添加一个维度  这里就变为了（E*heads*1)
        # x_j * α.unsqueeze(-1) 表示源节点对目标节点的贡献程度  结果的规模是(E, heads, D)
        return x_j * α.unsqueeze(-1)

    # ---------- aggregate ----------
    def update(self, aggr_out):
        """
        aggr_out : [N,H,D] → reshape 拼接
        """
        N = aggr_out.size(0)
        return aggr_out.view(N, self.heads * self.out_dim)        # [N, H*D]



class HW_E_GATNet(nn.Module):
    def __init__(self,
                 d_in: int,
                 edge_dim: int,
                 hidden: int,
                 d_out: int,
                 heads: list ,
                 dropout: float ,
                 use_edge_all: bool ):
        """
        heads : len == 层数；如 [4,4,1] → 3 层（有几个heda就表明信息传递几层）
        use_edge_all : True  后续层也用边特征 (HW_E_GATConv)
                       False  仅首层用边特征，后续用普通 GATConv
        """
        super().__init__()  # 调用父类（nn.Module）的构造函数
        self.layers = nn.ModuleList() # 定义一个 空的ModuleList 容器，用于存储和管理多个子模块，并确保这些子模块的参数能够被自动注册和管理。
        # 利用dim创建一个关于层的维度列表   d_in是输入层维度，hidden是隐藏层的输入和输出维度，除了最后一层其他的输出维度都是hidden规定
        # 假设d_in=32 ;hidden = 64 ; len(head) = 4    那么下面的dims=[32,64,64,64]
        dims = [d_in] + [hidden] * (len(heads) - 1)


        # 以下for循环的内容主要是根据 heads 列表的内容构建网络层，并将其添加到 self.layers
        for l, h in enumerate(heads):   # 其中l是heads的索引，h是该索引对应的值即head的具体数值
            # 当索引是l的时候，in_d取值dims[l]表示输入特征维度，而输出特征维度取hidden  但如果索引是最后一个，则这时候out_d取值d_out
            in_d, final_out_d = dims[l], hidden if l < len(heads) - 1 else d_out
            if l == 0 or use_edge_all:
                self.layers.append(HW_E_GATConv(in_d, final_out_d, edge_dim,
                                                heads=h, dropout=dropout))
            else:
                self.layers.append(GATConv(in_d, final_out_d, heads=h, concat=(h > 1),
                                           dropout=dropout))
        self.dropout = dropout  #将 dropout 值保存为模型的一个属性，方便在其他地方访问

    def forward(self, x, edge_index, edge_attr):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, HW_E_GATConv):  # 用来检查当前层是否是HW_E_GATConv 类型的层
                x = layer(x, edge_index, edge_attr)
            else:  # 普通 GATConv
                x = layer(x, edge_index)
            if i != len(self.layers) - 1:  # 如果不是最后一层使用ELU激活函数
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)  # 进行dropout操作
        return x

# -------------------------- 多头GAT ----------------------------------------------------------------------------------


# ------------------------------- 处理多头GAT的输出 --------------------------------------------------------------------
"""
GAT网络的输出是该批量图像的所有节点更新后的新节点信息，即（N,C)  N=N1+N2+N3....
为了与对应的CNN特征进行融合，需要把GAT特征重新切回列表形式，每个列表元素对应一个图像的更新节点特征
把对应图像的节点特征转换到对应特征图上，即某节点对应的像素均使用该节点的信息
以下代码应放置在GAT模型forward之后进行一次操作
"""
def GAT_final_feats(x_all, batch, batch_segments):
    """
        x_all   : (ΣN, D_out)   # GAT 输出的节点特征，第 0 维按 Batch 拼接
        batch   : Batch 对象    # 含有 batch.ptr / batch.batch
        segments: List[Tensor]  # 每张图的超像素 label，shape = (H, W)
        """

    # ① 用 ptr 把节点特征按图切回来，放到列表 per_graph_feats
    per_graph_feats = [
        x_all[batch.ptr[i]: batch.ptr[i + 1]]  # (N_i, D_out)
        for i in range(batch.num_graphs)  # 遍历每张图
    ]
    # ② 把每张图的节点特征映射回像素 (H, W, D)
    batch_pix_feats = []
    for feats, segments in zip(per_graph_feats, batch_segments):
        # feats: (N_i, D_out)  每个图像的节点信息
        # p2n  : (H, W), 其中每个位置都是 0 <= idx < N_i   对应图像的分块标记segments
        # 直接把 p2n 当成索引，得到 (H, W, D_out)
        pix_feat = feats[segments]  # → (H, W, D_out)
        batch_pix_feats.append(pix_feat)
    # 再把所有图拼成一个 batch：
    batch_GAT_feats = torch.stack(batch_pix_feats, dim=0)
    batch_GAT_feats = batch_GAT_feats.permute(0, 3, 1, 2).contiguous()  # (B, d_out, H, W)  ← NCHW
    return batch_GAT_feats


# # x=model(GAT) 即最后GAT输出的拼接在一起的节点特征；num_nodes是对批量数据使用PyG拼接时构建的节点数量列表；dim=0表示从行的方向进去切分
# per_graph_feats = list(torch.split(x, num_nodes, dim=0))
# # 每个图像对应像素都有了对应的GAT特征，然后多个图的此特征构建为列表
# batch_GAT_feats = []
# for feats, p2n in zip(per_graph_feats, batch_segments):
#     # feats: (N_i, D_out)
#     # p2n  : (H, W), 其中每个位置都是 0 <= idx < N_i
#     # 直接把 p2n 当成索引，得到 (H, W, D_out)
#     pix_feat = feats[p2n]            # → (H, W, D_out)
#     pixel_feats.append(pix_feat)
#
# # 3) 再把所有图拼成一个 batch：
# batch_pix_feats = torch.stack(pixel_feats, dim=0)

#--------------------------------Color Branch block End -----------------------------------------------------------------



#---------------------------------------Noise_Branch-------------------------------------------------------------------
"""
此噪声分支通过融合空域噪声和频域噪声进行图像的噪声特征提取
"""
# 自编码器
class BasicBlock(nn.Sequential):
    """
    Conv-BN-ReLU
    nn.Sequential 自带 forward()，这个父类已经实现了“把输入按定义顺序依次喂给内部子模块”的 forward()，我们只是在 __init__ 里把 3 个层（Conv → BN → ReLU）按顺序塞进去。
    因此 BasicBlock(x) 本质上就是 ReLU(BN(Conv(x)))，直接调用即可，无需再手写 forward()。
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )


# 定义ConvAlign
class ConvAlign(nn.Module):
    def __init__(self, stride_total):
        super().__init__()
        if stride_total == 1:          # I1 → 3 ch
            self.align = nn.Sequential(
                nn.Conv2d(3, 3, 3, 1, 1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True))
        elif stride_total == 2:        # I2
            self.align = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1, bias=False),  # ↓2
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        elif stride_total == 4:        # I3
            self.align = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1, bias=False),  # ↓2
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 2, 1, bias=False),  # 再 ↓2
                nn.BatchNorm2d(128), nn.ReLU(inplace=True))
    def forward(self, x): return self.align(x)


class SimpleCAE(nn.Module):
    def __init__(self, in_ch=3, base=64, latent=256):
        super().__init__()
        # ---------- Encoder ----------
        self.g1 = BasicBlock(in_ch, base, 3, 2)          # 1/2
        self.g2 = BasicBlock(base, base*2, 3, 2)         # 1/4
        self.g3 = BasicBlock(base*2, base*4, 3, 2)       # 1/8
        self.to_z = nn.Conv2d(base*4, latent, 1)         # 压到 latent dim

        # ---------- Decoder ----------
        self.up3 = nn.ConvTranspose2d(latent, base*4, 4, 2, 1)  # 1/4
        self.f3 = BasicBlock(base*4, base*2)                    # f3 (大尺度)
        self.up2 = nn.ConvTranspose2d(base*2, base*2, 4, 2, 1)  # 1/2
        self.f2 = BasicBlock(base*2, base)                      # f2
        self.up1 = nn.ConvTranspose2d(base, base, 4, 2, 1)      # 1/1
        self.f1 = BasicBlock(base, in_ch)                       # f1 (与输入同分辨率)

        # ----------ConvAlign-------------
        self.align1 = ConvAlign(1)  # 256→256   C=3
        self.align2 = ConvAlign(2)  # 256→128   C=64
        self.align3 = ConvAlign(4)  # 256→64    C=128

    def forward(self, x):
        g1 = self.g1(x)
        g2 = self.g2(g1)
        g3 = self.g3(g2)
        z  = self.to_z(g3)

        d3 = self.up3(z)
        f3 = self.f3(d3)

        d2 = self.up2(f3)
        f2 = self.f2(d2)

        d1 = self.up1(f2)
        f1 = self.f1(d1)

        I1 = self.align1(x)
        I2 = self.align2(x)
        I3 = self.align3(x)

        R1 = f1 - I1
        R2 = f2 - I2
        R3 = f3 - I3

        # 若仅需重建，可直接输出 f1；但子网络中要把 f1-f3 作为残差对齐
        return f1, f2, f3, I1, I2, I3, R1, R2, R3


# 求解U---------start
# ---------- 工具函数 ----------
def local_mean(x, k=5):
    pad = k // 2
    return F.avg_pool2d(x, k, stride=1, padding=pad)

def local_std(x, k=5):
    mu = local_mean(x, k)
    mean_sq = local_mean(x * x, k)
    var = torch.clamp(mean_sq - mu * mu, min=1e-10)
    return torch.sqrt(var)

# ---------- Lite 分支 ----------
class ULite(nn.Module):
    """DW-Conv(3×3, in=1->mid) + 1×1(mid->1)"""
    def __init__(self, mid_ch: int = 16):
        super().__init__()
        self.dw = nn.Conv2d(1, mid_ch, 3, 1, 1, groups=1)  # 对单通道相当于普通 3×3
        self.bn = nn.BatchNorm2d(mid_ch)
        self.pw = nn.Conv2d(mid_ch, 1, 1, 1, 0)

    def forward(self, x):
        x = F.relu(self.bn(self.dw(x)))
        return torch.sigmoid(self.pw(x))        # 0-1 软权重图


# ---------- 主类 ----------
class UncertaintyHead(nn.Module):
    """
    R_gray -> U_stat (Std+CV)  (+可选 ULite 修正)
    参数:
        k              : 滑窗大小
        learnable_lam  : 是否让 λ1, λ2 可训练 (softmax 归一)
        use_lite       : 是否启用 ULite 分支
        learnable_beta : β 是否可训练 (若 False, β 固定值)
        beta_init      : β 初值 (0.0–1.0)
        mid_lite       : ULite 隐层通道数
    """
    def __init__(self,
                 k: int = 5,
                 learnable_gray=False,
                 learnable_lam: bool = False,
                 use_lite: bool = False,
                 learnable_beta: bool = True,
                 beta_init: float = 0.7,
                 mid_lite: int = 16):
        super().__init__()
        self.k = k
        # ---- 灰度映射 ----
        if learnable_gray:
            conv = nn.Conv2d(3, 1, 1, bias=False)
            with torch.no_grad():
                conv.weight[:] = torch.tensor([[[[0.2989]],
                                                [[0.5870]],
                                                [[0.1140]]]])
            self.rgb2gray = conv  # 可学习 1×1
        else:
            self.rgb2gray = None  # forward 用库函数
        # λ1, λ2
        if learnable_lam:
            self.lam = nn.Parameter(torch.tensor([0.7, 0.3]))
        else:
            self.register_buffer("lam", torch.tensor([0.7, 0.3]))  # 固定
        self.learnable_lam = learnable_lam
        # ULite
        # ─────  ULite 分支 + β  ─────
        self.use_lite = use_lite
        if self.use_lite:
            self.lite = ULite(mid_lite)  # 轻量卷积修正

            beta_logit = torch.tensor(beta_init).logit()  # logit(p)
            if learnable_beta:  # β 可训练
                self.beta = nn.Parameter(beta_logit)
                self.learnable_beta = True
            else:  # β 固定
                self.register_buffer("beta", beta_logit)
                self.learnable_beta = False
        else:
            # 未启用 ULite：无 β，无学习
            self.lite = None
            self.beta = None
            self.learnable_beta = False

    # ---------- 灰度助手 ----------
    def _gray(self, x):
        if self.rgb2gray is None:                    # 走库函数
            return TF.rgb_to_grayscale(x, num_output_channels=1)
        else:                                        # 1×1 卷积
            return self.rgb2gray(x)

    # ---------- 手工 U_stat ----------
    def _u_stat(self, R):
        σ  = local_std(R, self.k)
        μ  = local_mean(R, self.k)
        cv = σ / (μ + 1e-3)
        lam = F.softmax(self.lam, 0) if self.learnable_lam else self.lam
        return torch.sigmoid(lam[0] * σ + lam[1] * cv)        # (B,1,H,W)

    def forward(self, f1, img_rgb):
        """
        Parameters
        ----------
        f1       : (B,3,H,W)  VAE 重建
        img_rgb  : (B,3,H,W)  原始图像
        Returns
        -------
        R_gray   : (B,1,H,W)
        U        : (B,1,H,W)
        """
        f1_gray = self._gray(f1)
        img_gray = TF.rgb_to_grayscale(img_rgb, num_output_channels=1)
        R_gray = f1_gray - img_gray

        U_stat = self._u_stat(R_gray)

        if not self.use_lite:
            return U_stat

        U_lite = self.lite(R_gray)
        beta = torch.sigmoid(self.beta)  # 标量 β ∈ (0,1)
        U = beta * U_stat + (1.0 - beta) * U_lite
        return U

# 反卷积模块，用于上采样并调整通道数
class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        return self.deconv(x)
# 主模块，输入三个特征图R1, R2, R3，输出拼接结果
class RFeatureFusion(nn.Module):
    def __init__(self):
        super(RFeatureFusion, self).__init__()
        # Conv1x1 用于调整 R1 的通道数为64
        self.conv1x1_R1   = nn.Conv2d(3, 64, kernel_size=1)  # 假设 R1 通道数为128
        self.conv1x1_Rcat = nn.Conv2d(192, 64, kernel_size=1)  # 假设 R1 通道数为128
        # 反卷积模块，用于上采样 R2 和 R3 的尺寸，同时调整通道数为64
        self.deconv_R2 = DeconvBlock(64, 64)  # R2 通道数是 64，尺寸是 (1/2H)*(1/2W)
        self.deconv_R3 = DeconvBlock(128, 64)  # R3 通道数是 128，尺寸是 (1/4H)*(1/4W)

    def forward(self, R1, R2, R3):
        # R1 通道调整：假设 R1 的尺寸是 HxW，通道数为 3
        R1_out = self.conv1x1_R1(R1)  # 调整 R1 的通道数为 64
        # R2 上采样并调整通道数
        R2_up = self.deconv_R2(R2)  # 上采样并调整通道数为 64
        # R3 上采样并调整通道数
        R3_up_1 = self.deconv_R3(R3)  # 上采样并调整通道数为 64
        R3_up   = self.deconv_R2(R3_up_1)
        # 拼接三个输出特征图（通道数一致，尺寸一致）
        R_cat = torch.cat([R1_out, R2_up, R3_up], dim=1)  # 拼接后通道数为 3*64，尺寸为 HxW
        # 对R_cat进行降维
        R = self.conv1x1_Rcat(R_cat)
        return R

# 膨胀卷积
class DilatedConvBlock(nn.Sequential):
    """3×3 DilatedConv + BN + ReLU"""
    def __init__(self, in_ch, out_ch, rate):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3,
                      padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class MultiRateDilatedConv(nn.Module):
    """并行 rate = 1, 2, 4，然后 concat"""
    def __init__(self, in_ch, out_ch, rates=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList(
            [DilatedConvBlock(in_ch, out_ch, r) for r in rates]
        )
        # 可选 1×1 压通道，保持 out_ch
        #self.fuse = nn.Conv2d(out_ch * len(rates), out_ch, 1, bias=False)

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return x


class noise_space(nn.Module):
    def __init__(self, rates=(1, 2, 4)):
        super().__init__()
        # 1) U 升维 1→64
        self.expand_U = nn.Conv2d(1, 64, kernel_size=1)  # 假设 R1 通道数为128
        # 2) 多尺度 DilConv: 输入 128 ch → 输出 48 ch
        self.mrdc = MultiRateDilatedConv(in_ch=128, out_ch=48, rates=rates)
        # 3) 对多尺度膨胀卷积的拼接结果 降维 48*3→128
        self.Multi_conv = nn.Conv2d(144, 128, kernel_size=1)  # 假设 R1 通道数为128
        # 4) SE + SA
        self.se = SElayer(128, reduction=16)
        self.SA = SpatialAttentionModule(128,1)


    def forward(self, R, U):
        # U: 1→64
        U64 = self.expand_U(U)  # (B,64,H,W)
        # 拼接→128 ch
        x = torch.cat([R, U64], dim=1)  # (B,128,H,W)
        # 多尺度 Dilated Pyramid
        x1 = self.mrdc(x)  # (B,64,H,W)  已含后续 1×1 Fuse
        # 对多尺度膨胀卷积拼接结果降维到128
        x2 = self.Multi_conv(x1)
        # 通道 SE
        x_se = self.se(x)
        # 进行SA
        Ms = self.SA(x_se)
        # 像素级乘权
        F_sn = x2 * Ms

        return F_sn



#----------------------------------------- 频域部分
# 频域分解中的多尺度卷积
class Frequency_MultiScaleConv(nn.Module):
    """
    多尺度卷积：在输入的频谱幅度图上并行地做 1×1、3×3、5×5 卷积，
    然后在通道维度把结果拼起来。
    输入： [B*N, 1, P, P]
    输出： [B*N, 3*out_channels, P, P]
    """
    def __init__(self, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(1, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(1, out_channels, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.conv1(x)
        f2 = self.conv3(x)
        f3 = self.conv5(x)
        return torch.cat([f1, f2, f3], dim=1)


# 图像划分patch且进行多尺度卷积的分支
class FFTSingleBranch(nn.Module):
    """单尺度分支：切 patch → FFT → 多尺度卷积 → fold 回原图"""
    def __init__(self, patch_size: int, out_channels: int):
        super().__init__()
        self.P = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.msc    = Frequency_MultiScaleConv(out_channels)

    def forward(self, x_gray: torch.Tensor):
        B, C, H, W = x_gray.shape
        # 1) 切 patch，得到 [B, P*P, N]
        patches = self.unfold(x_gray)
        # 2) 恢复成小图 [B*N,1,P,P]
        _, PP, N = patches.shape
        patches = patches.transpose(1,2).reshape(B*N, 1, self.P, self.P)
        # 3) FFT 幅度谱
        fft = torch.fft.fft2(patches)
        amp = torch.sqrt(fft.real**2 + fft.imag**2)
        # 4) 多尺度卷积 → [B*N, 3*out_c_each, P, P]
        feat = self.msc(amp)
        # 5) fold 回 [B, 3*out_c_each, H, W]
        feat = feat.view(B, N, feat.size(1), self.P, self.P)
        feat = feat.permute(0,2,3,4,1).reshape(B, -1, N)
        out  = F.fold(feat, (H, W), kernel_size=self.P, stride=self.P)
        return out

# =====================================================CBAM模块=======================================================
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # 1. 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 2. 全局最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 3. 全连接层，用于学习通道权重
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )

        # 4. Sigmoid 激活函数，输出通道注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # 全局平均池化 -> 卷积学习权重
        max_out = self.fc(self.max_pool(x))  # 全局最大池化 -> 卷积学习权重
        out = avg_out + max_out  # 将平均池化和最大池化结果加权
        return self.sigmoid(out)  # 输出加权后的通道注意力权重


class SpatialAttentionModule(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        # 1. 拼接全局平均池化和全局最大池化的结果
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        # 2. Sigmoid 激活函数，输出空间注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 全局平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 全局最大池化
        # 将两种池化结果拼接，得到空间特征
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 用卷积生成空间注意力
        out = self.conv(x_cat)
        return self.sigmoid(out)  # 输出空间注意力权重


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 1. 通道注意力模块
        self.channel_attention = ChannelAttentionModule(in_channels, reduction_ratio)
        # 2. 空间注意力模块
        self.spatial_attention = SpatialAttentionModule(kernel_size)

        def forward(self, x):
            # 1. 先进行通道注意力
            x = x * self.channel_attention(x)
            # 2. 再进行空间注意力
            x = x * self.spatial_attention(x)
            return x



# 图像进行多尺度patch划分及卷积的整体分支
class MultiScaleFFTBranch(nn.Module):
    """
    总分支：显式创建3个尺度的子模块，不用循环。
    最终输出通道 = 3（patch 尺寸）×3（卷积核）×out_c_each
    """
    def __init__(self, out_c_each: int):
        super().__init__()
        self.branch32 = FFTSingleBranch(32, out_c_each)
        self.branch16 = FFTSingleBranch(16, out_c_each)
        self.branch8  = FFTSingleBranch(8,  out_c_each)

    def forward(self, x_gray: torch.Tensor):
        f32 = self.branch32(x_gray)   # [B, 3*out_c_each, H, W]
        f16 = self.branch16(x_gray)
        f8  = self.branch8(x_gray)
        # 拼接三种尺度
        return torch.cat([f32, f16, f8], dim=1)  # [B, 9*out_c_each, H, W]

# CBAM模块
