import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import torchvision.models as models

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))


    def init_params(self, clsts, traindescs):

        clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
        dots = np.dot(clstsAssign, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :] # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

            

    def forward(self, x):
        N, C = x.shape[:2]
        x_flatten = x.view(N, C, -1)
        
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)

            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class REM(nn.Module):
    def __init__(self, from_scratch=False, rotations=8):
        super(REM, self).__init__()
        
        # cnn backbone
        pretrain = not from_scratch
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrain else None
        encoder = models.resnet34(weights=weights) #resnet34
        layers = list(encoder.children())[:-4]
        self.encoder = nn.Sequential(*layers)

        # rotations
        self.angles = -torch.arange(0,359.00001,360.0/rotations)/180*torch.pi

    
    def forward(self, x):
        
        equ_features = []
        
        batch_size = x.size(0)

        for i in range(len(self.angles)):

            # input warp grids
            aff = torch.zeros(batch_size,2,3, device=x.device, dtype=x.dtype)
            aff[:,0,0]=torch.cos(-self.angles[i])
            aff[:,0,1]=torch.sin(-self.angles[i])
            aff[:,1,0]=-torch.sin(-self.angles[i])
            aff[:,1,1]=torch.cos(-self.angles[i])
            grid = F.affine_grid(aff, torch.Size(x.size()),align_corners=True).type(x.type())
            
            # input warp
            warped_im = F.grid_sample(x, grid,align_corners=True,mode='bicubic')
                                    
            # cnn backbone feature
            out = self.encoder(warped_im) 

            # output feature warp grids           
            if i==0:
                im1_init_size = out.size()

            aff = torch.zeros(batch_size,2,3, device=x.device, dtype=x.dtype)
            aff[:,0,0]=torch.cos(self.angles[i])
            aff[:,0,1]=torch.sin(self.angles[i])
            aff[:,1,0]=-torch.sin(self.angles[i])
            aff[:,1,1]=torch.cos(self.angles[i])
            grid = F.affine_grid(aff, torch.Size(im1_init_size),align_corners=True).type(x.type())

            # output feature warp    
            out = F.grid_sample(out, grid ,align_corners=True,mode='bicubic')

            equ_features.append(out.unsqueeze(-1))
        

        equ_features = torch.cat(equ_features, axis=-1)  # B C H W R

        B, C, H, W, R = equ_features.shape
        equ_features=torch.max(equ_features,dim=-1,keepdim=False)[0] # max pooling along rotations

        aff = torch.zeros(batch_size,2,3, device=x.device, dtype=x.dtype)
        aff[:,0,0]=1
        aff[:,0,1]=0
        aff[:,1,0]=0
        aff[:,1,1]=1

        
        # upsample for NetVLAD
        B,C,H,W = x.size()
        grid = F.affine_grid(aff, torch.Size((B, C, H//4, W//4)),align_corners=True).type(x.type())#,align_corners=True)
        out1 = F.grid_sample(equ_features, grid,align_corners=True,mode='bicubic')
        out1 = F.normalize(out1, dim=1)
        
        # upsample for keypoints
        grid = F.affine_grid(aff, torch.Size((B, C, H, W)),align_corners=True).type(x.type())#,align_corners=True)
        out2 = F.grid_sample(equ_features, grid,align_corners=True,mode='bicubic')
        out2 = F.normalize(out2, dim=1)
        
        return out1, out2

class REIN(nn.Module):
    def __init__(self):
        super(REIN, self).__init__()
        self.rem = REM()
        self.pooling = NetVLAD()

        self.local_feat_dim = 128
        self.global_feat_dim = self.local_feat_dim*64
    
    def forward(self, x):

        out1, local_feats = self.rem(x)

        global_desc = self.pooling(out1)

        return out1, local_feats, global_desc
    # =========================================================================
# 【新增】1. 空間幾何注意力機制 (Spatial Attention Module)
# =========================================================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道維度上計算平均池化與最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接後通過卷化層與 Sigmoid 函數得到幾何權重圖
        res = torch.cat([avg_out, max_out], dim=1)
        res = self.conv1(res)
        return self.sigmoid(res) * x  # 將空間注意力乘回原特徵圖

# =========================================================================
# 【新增】2. 學生版的輕量化 REM (使用預訓練好的 MobileNetV3)
# =========================================================================
class StudentREM(nn.Module):
    def __init__(self, from_scratch=False, rotations=8, target_dim=128):
        super(StudentREM, self).__init__()
        self.rotations = rotations
        self.target_dim = target_dim
        
        # 實作更小的 Backbone：使用預訓練的 MobileNetV3 Large
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if not from_scratch else None
        mobilenet = models.mobilenet_v3_large(weights=weights)
        # 抽取前 11 個特徵層，此時輸出通道數（C）為 80
        self.encoder = nn.Sequential(*list(mobilenet.features[:11]))
        
        # 關鍵對接點：MobileNet 輸出是 80 通道，但老師 ResNet 是 128 通道。
        # 我們加一個 1x1 卷積把通道數轉換成 128，以便之後跟老師計算 Loss_Spatial！
        self.channel_adapter = nn.Conv2d(80, target_dim, kernel_size=1) 
        self.angles = -torch.arange(0, 359.00001, 360.0 / rotations) / 180 * torch.pi

    def forward(self, x):
        equ_features = []
        batch_size = x.size(0)

        for i in range(len(self.angles)):
            # 輸入影像旋轉
            aff = torch.zeros(batch_size, 2, 3, device=x.device, dtype=x.dtype)
            aff[:, 0, 0] = torch.cos(-self.angles[i])
            aff[:, 0, 1] = torch.sin(-self.angles[i])
            aff[:, 1, 0] = -torch.sin(-self.angles[i])
            aff[:, 1, 1] = torch.cos(-self.angles[i])
            grid = F.affine_grid(aff, torch.Size(x.size()), align_corners=True).type(x.type())
            warped_im = F.grid_sample(x, grid, align_corners=True, mode='bicubic')
                                    
            # 輕量化 Backbone 提特徵
            out = self.encoder(warped_im) 
            out = self.channel_adapter(out)  # 映射到 128 維度對齊老師
            
            if i == 0:
                im1_init_size = out.size()

            # 特徵圖反向旋轉對齊
            aff = torch.zeros(batch_size, 2, 3, device=x.device, dtype=x.dtype)
            aff[:, 0, 0] = torch.cos(self.angles[i])
            aff[:, 0, 1] = torch.sin(self.angles[i])
            aff[:, 1, 0] = -torch.sin(self.angles[i])
            aff[:, 1, 1] = torch.cos(self.angles[i])
            grid = F.affine_grid(aff, torch.Size(im1_init_size), align_corners=True).type(x.type())
            out = F.grid_sample(out, grid, align_corners=True, mode='bicubic')
            equ_features.append(out.unsqueeze(-1))
        
        # 旋轉角度特徵融合 (Max Pooling)
        equ_features = torch.cat(equ_features, axis=-1)
        equ_features = torch.max(equ_features, dim=-1, keepdim=False)[0]

        # 沿用原作者的 Upsample 採樣邏輯，完美對接資料流
        aff = torch.zeros(batch_size, 2, 3, device=x.device, dtype=x.dtype)
        aff[:, 0, 0], aff[:, 1, 1] = 1, 1
        
        B, C, H, W = x.size()
        # 輸出用於聚合的 out1 特徵圖 (尺寸為 H//4, W//4)
        grid = F.affine_grid(aff, torch.Size((B, self.target_dim, H // 4, W // 4)), align_corners=True).type(x.type())
        out1 = F.grid_sample(equ_features, grid, align_corners=True, mode='bicubic')
        out1 = F.normalize(out1, dim=1)
        
        # 輸出原始尺寸的 local_feats
        grid = F.affine_grid(aff, torch.Size((B, self.target_dim, H, W)), align_corners=True).type(x.type())
        out2 = F.grid_sample(equ_features, grid, align_corners=True, mode='bicubic')
        out2 = F.normalize(out2, dim=1)
        
        return out1, out2

# =========================================================================
# 【新增】3. 完整的學生版 REIN 網路 (NetVLAD 換成 Spatial Attention)
# =========================================================================
class StudentREIN(nn.Module):
    def __init__(self, teacher_global_dim=8192, feat_h=40, feat_w=40):
        super(StudentREIN, self).__init__()
        self.rem = StudentREM(target_dim=128)
        
        # 【核心修改】用 Spatial Attention 替換原本的 NetVLAD
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # 為了能和老師的 Global Descriptor (8192維) 計算 Loss_Global，並且相容 main.py 的 Cache 與測試系統
        # 我們將 Attention 加權後的特徵圖 Flatten，再用一個全連接層（Linear）投影到 8192 維度！
        # 使用 Adaptive Average Pooling 固定輸出空間尺寸，使模型相容任意輸入影像大小
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.flat_features_dim = 128 * feat_h * feat_w
        self.fc_global = nn.Linear(self.flat_features_dim, teacher_global_dim)

        # 變數名稱對齊原本的 REIN，確保 main.py 讀取時不會噴錯
        self.local_feat_dim = 128
        self.global_feat_dim = teacher_global_dim
    
    def forward(self, x):
        # 1. 通過輕量化的 MobileNet REM 提取特徵
        out1, local_feats = self.rem(x)
        
        # 2. 將 out1 餵入幾何空間注意力機制中
        attn_feats = self.spatial_attention(out1)
        
        # 3. 使用 Adaptive Average Pooling 固定空間維度，確保任意輸入大小都能正確展平
        B = x.size(0)
        pooled = F.adaptive_avg_pool2d(attn_feats, (self.feat_h, self.feat_w))
        flat_feats = pooled.view(B, -1)
        global_desc = self.fc_global(flat_feats)
        global_desc = F.normalize(global_desc, p=2, dim=1) # L2 歸一化
        
        # 防護：確保 local_feats 為 4D 張量 (B, C, H, W)，否則上游的插值會失敗
        if local_feats.dim() != 4:
            raise RuntimeError(f"StudentREIN.forward: local_feats 必須為 4D 張量 (B,C,H,W)，目前形狀 {tuple(local_feats.shape)}")


        return out1, local_feats, global_desc