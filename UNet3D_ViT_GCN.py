# # 导入必要的库
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
#
#
# # 定义3D UNet模型
# class UNet3D(nn.Module):
#     def __init__(self):
#         super(UNet3D, self).__init__()
#         # 编码器部分
#         self.enc_conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
#         self.enc_conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.enc_conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
#         self.enc_conv4 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
#         # 解码器部分
#         self.dec_conv1 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
#         self.dec_conv2 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
#         self.dec_conv3 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
#         self.dec_conv4 = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         # 编码器部分
#         x1 = F.relu(self.enc_conv1(x))
#         x2 = F.relu(self.enc_conv2(F.max_pool3d(x1, 2)))
#         x3 = F.relu(self.enc_conv3(F.max_pool3d(x2, 2)))
#         x4 = F.relu(self.enc_conv4(F.max_pool3d(x3, 2)))
#         # 解码器部分
#         y1 = F.relu(self.dec_conv1(F.interpolate(x4, scale_factor=2, mode='trilinear', align_corners=True)))
#         y2 = F.relu(self.dec_conv2(torch.cat([y1, x3], dim=1)))
#         y3 = F.relu(self.dec_conv3(torch.cat([y2, x2], dim=1)))
#         y4 = self.dec_conv4(torch.cat([y3, x1], dim=1))
#         return y4
#
#
# # 定义Vision Transformer模型
# class ViT(nn.Module):
#     def __init__(self, img_size, patch_size, num_classes, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
#         super(ViT, self).__init__()
#         self.patch_size = patch_size
#         self.num_patches = (img_size // patch_size) ** 2
#         self.patch_dim = 3 * patch_size ** 2
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.patch_embeddings = nn.Linear(self.patch_dim, dim)
#         self.dropout = nn.Dropout(dropout)
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
#             num_layers=depth)
#         self.layer_norm = nn.LayerNorm(dim)
#         self.fc = nn.Linear(dim, num_classes)
#
#     def forward(self, x):
#         # 将3D医学图像数据转换为图像块
#         x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).unfold(4,
#                                                                                                              self.patch_size,
#                                                                                                              self.patch_size)
#         x = x.contiguous().view(x.size(0), -1, self.patch_dim)
#         # 将图像块映射到低维空间
#         x = self.patch_embeddings(x)
#         x = self.dropout(x)
#         # 添加CLS token
#         cls_token = self.cls_token.expand(x.size(0), -1, -1)
#         x = torch.cat([cls_token, x], dim=1)
#         # 使用Transformer编码器提取全局信息
#         x = self.transformer(x)
#         x = self.layer_norm(x)
#         # 提取CLS token作为全局特征
#         x = x[:, 0, :]
#         # 使用全连接层进行分类
#         x = self.fc(x)
#         return x
#
#
# # 定义GCN模型
# class GCN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return x
#
#
# # 将3D UNet、Vision Transformer和GCN模型连接起来
# class UNet3D_ViT_GCN(nn.Module):
#     def __init__(self, img_size, patch_size, num_classes, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1,
#                  in_channels=64, hidden_channels=128, out_channels=256):
#         super(UNet3D_ViT_GCN, self).__init__()
#         self.unet = UNet3D()
#         self.vit = ViT(img_size, patch_size, dim, depth, heads, mlp_dim, dropout)
#         self.gcn = GCN(in_channels, hidden_channels, out_channels)
#         self.fc = nn.Linear(out_channels, num_classes)
#
#     def forward(self, x, edge_index):
#         # 使用3D UNet模型提取特征
#         x = self.unet(x)
#         # 使用Vision Transformer模型提取全局特征
#         y = self.vit(x)
#         # 使用GCN模型对全局特征进行建模
#         y = self.gcn(y, edge_index)
#         # 使用全连接层进行分类
#         y = self.fc(y)
#         return y
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import functional as F
from vit_pytorch import ViT


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv_Block, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UNet3D_ViT_GCN(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, patch_size=16, num_classes=2, dim=128, depth=6, heads=8, mlp_ratio=4., qkv_bias=True, gcn_channels=32, gcn_layers=2):
        super().__init__()
        # #新加的代码
        # num_classes=2
        # self.final_conv = nn.Conv3d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0)#这里结束
        self.num_classes=num_classes
        self.patch_size = patch_size
        # UNet3D Encoder
        self.encoder = nn.ModuleList([self._make_encoder_layer(in_channels, dim * 2 ** i) for i in range(depth)])
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # UNet3D Decoder
        self.upconv = nn.ModuleList([nn.ConvTranspose3d(dim * 2 ** i, dim * 2 ** (i - 1), kernel_size=2, stride=2) for i in range(depth, 1, -1)])
        self.decoder = nn.ModuleList([self._make_decoder_layer(dim * 2 ** i, dim * 2 ** (i - 1)) for i in range(depth, 0, -1)])
        # ViT
        self.vit = ViT(image_size=patch_size,patch_size=patch_size, mlp_dim=dim, in_channels=dim, num_classes=gcn_channels, dim=gcn_channels, depth=gcn_layers, heads=heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
        # GCN
        self.gcn = nn.ModuleList([GCNConv(gcn_channels, gcn_channels) for i in range(gcn_layers)])
        # Output
        self.out_conv = nn.Conv3d(dim, out_channels, kernel_size=1)
    def _make_encoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def _make_decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        # UNet3D Encoder
        features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x.unsqueeze(0))
            if x.size(1) != encoder_layer[0].weight.size(1):
                encoder_layer[0].weight = nn.Parameter(encoder_layer[0].weight[:, :x.size(1), :, :, :])
            features.append(x)
            x = self.pool(x)
        # ViT
        patches = self._extract_patches(features[-1])
        patches = rearrange(patches, 'b p c -> b c p')
        g = self.vit(patches)
        g = rearrange(g, 'b c p -> b p c')
        # GCN
        for gcn_layer in self.gcn:
            g = gcn_layer(g, self._build_adjacency_matrix(g))
        # UNet3D Decoder
        for i, decoder_layer in enumerate(self.decoder):
            x = self.upconv[i](x)
            x = torch.cat([features[-i-2], x], dim=1)
            x = decoder_layer(x)
        # Output
        x = self.out_conv(x)
        return x
    def _extract_patches(self, x):
        b, c, d, h, w = x.size()
        patches = x.unfold(2, self.patch_size, self.patch_size // 2).unfold(3, self.patch_size, self.patch_size // 2).unfold(4, self.patch_size, self.patch_size // 2)
        patches = patches.contiguous().view(b, c, -1, self.patch_size, self.patch_size, self.patch_size)
        return patches
    def _build_adjacency_matrix(self, x, k=6):
        b, n, c = x.size()
        x = F.normalize(x, dim=-1)
        similarity = torch.matmul(x, x.transpose(1, 2))
        _, indices = similarity.topk(k=k, dim=-1)
        row_indices = torch.arange(b).view(-1, 1, 1).repeat(1, n, k).cuda()
        col_indices = indices.cuda()
        adjacency_matrix = torch.zeros(b, n, n).cuda()
        adjacency_matrix[row_indices, col_indices] = 1
        return adjacency_matrix