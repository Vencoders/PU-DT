import math
import sys
from pointnet2_ops._ext import gather_points
sys.path.append("../")
import torch
import torch.nn as nn
from torch.nn import Conv1d, Conv2d
from knn_cuda import KNN
from pointnet2_ops.pointnet2_utils import grouping_operation
import torch.nn.functional as F
import pointnet2_ops.pointnet2_utils as pointnet2

class fea_fuse(nn.Module):
    def __init__(self, k=16, in_channel=3, out_channel=32):
        super(fea_fuse, self).__init__()
        self.in_channel = in_channel
        self.k = k
        self.KNN = KNN(k=k + 1, transpose_mode=False)
        self.conv1 = nn.Sequential(nn.Conv2d(2*3, out_channel, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(2*in_channel, out_channel, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(inplace=True))
    def forward(self, fea, x, idx):
        point_cloud_neighbors = grouping_operation(x, idx.contiguous().int())
        point_cloud_central = x.unsqueeze(2).repeat(1, 1, self.k, 1)
        point_real = torch.cat([point_cloud_central, (point_cloud_neighbors - point_cloud_central)], dim=1)
        geo_fea = self.conv1(point_real)
        fea_neighbors = grouping_operation(fea, idx.contiguous().int())
        fea_centres = fea.unsqueeze(2).repeat(1, 1, self.k, 1)
        fea_ral = torch.cat([fea_centres, (fea_neighbors - fea_centres)], dim=1)
        fea_fea = self.conv2(fea_ral)
        res = torch.cat([geo_fea, fea_fea], dim=1)
        res = torch.max(res, dim=2)[0]
        return res


class dual_extraction(nn.Module):
    def __init__(self, transform_dim=64, k=16):
        super(dual_extraction, self).__init__()
        self.k = k
        self.KNN = KNN(k=k + 1, transpose_mode=False)
        self.conv = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=1)
        self.dual_transformer1 = DualTransformer(in_channel=16, out_channel=32, transform_dim=transform_dim)
        self.dual_transformer2 = DualTransformer(in_channel=64, out_channel=64, transform_dim=transform_dim)
    def forward(self, x):
        # index
        dist, idx = self.KNN(x, x)
        idx = idx[:, 1:, :]
        fea = self.conv(x)
        fea1 = self.dual_transformer1(fea, x, idx)
        fea2 = self.dual_transformer2(fea1,x, idx)
        return fea2

class DualTransformer(nn.Module):
    def __init__(self, in_channel=16, out_channel=32, transform_dim=64,K=16):
        super(DualTransformer, self).__init__()
        self.transform_dim = transform_dim
        self.K = K
        self.fea = fea_fuse(k=16, in_channel=in_channel, out_channel=out_channel)
        self.conv1 = nn.Conv1d(2*out_channel, self.transform_dim, 1)
        self.w_qs = nn.Conv1d(self.transform_dim, self.transform_dim, 1)
        self.w_ks = nn.Conv1d(self.transform_dim, self.transform_dim, 1)
        self.w_vs = nn.Conv1d(self.transform_dim, self.transform_dim, 1)
        self.fc_delta = nn.Sequential(
            nn.Conv2d(10, 64, [1, 1]),
            nn.ReLU(),
            nn.Conv2d(64, self.transform_dim, [1, 1]),
        )
        self.fc_gamma = nn.Sequential(
            nn.Conv2d(self.transform_dim, 4 * self.transform_dim, [1, 1], ),
            nn.ReLU(),
            nn.Conv2d(4 * self.transform_dim, self.transform_dim, [1, 1], ),
        )
        self.trans_conv = nn.Conv1d(self.transform_dim, self.transform_dim, 1)
        self.after_norm = nn.BatchNorm1d(self.transform_dim)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.conv2 = nn.Conv1d(self.transform_dim, 2*out_channel, 1)
        self.conv3 = nn.Conv1d(self.transform_dim, 2*out_channel, 1)

        self.lbr = nn.Sequential(
            nn.Conv1d(2 * out_channel, 2 * out_channel, 1),
            nn.BatchNorm1d(2 * out_channel),
            nn.ReLU(),
        )
    def forward(self, feature, xyz, idx):
        fea = self.fea(feature, xyz, idx)
        x = self.conv1(fea)
        q, k, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)  
        group_xyz = grouping_operation(xyz, idx.contiguous().int())  
        rel_xyz = group_xyz - xyz[:, :, None, :]
        rel_pos = torch.cat(
            [
                xyz.unsqueeze(2).repeat(1, 1, self.K, 1),
                group_xyz,
                rel_xyz,
                torch.norm(rel_xyz, dim=1, keepdim=True)
            ],
            dim=1,
        )  
        pos_enc = self.fc_delta(rel_pos)  
        k_local = grouping_operation(k, idx.contiguous().int())
        v_local = grouping_operation(v, idx.contiguous().int())  
        attn = self.fc_gamma(q[:, :, None, :] - k_local + pos_enc)
        attn = F.softmax(attn, dim=-2)  
        res = torch.einsum("bmnf,bmnf->bmf", attn, v_local + pos_enc)
        res_local = self.conv2(res)
        energy = torch.matmul(q.permute(0, 2, 1).contiguous(), k)  
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.matmul(v, attention)  
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        res_global = x + x_r
        res_global = self.conv3(res_global)
        fea1 = torch.add(res_local, res_global)
        fea2 = self.lbr(fea1)
        fea3 = torch.add(fea, fea2)
        return fea3

class get_edge_feature(nn.Module):
    """construct edge feature for each point
    Args:
        tensor: input a point cloud tensor,batch_size,num_dims,num_points
        k: int
    Returns:
        edge features: (batch_size,num_dims,num_points,k)
    """
    def __init__(self, k=16):
        super(get_edge_feature, self).__init__()
        self.KNN = KNN(k=k + 1, transpose_mode=False)
        self.k = k

    def forward(self, point_cloud):
        dist, idx = self.KNN(point_cloud, point_cloud)
        """
        idx is batch_size,k,n_points
        point_cloud is batch_size,n_dims,n_points
        point_cloud_neightbors is batch_size,n_dims,k,n_points
        """
        idx = idx[:, 1:, :]
        point_cloud_neighbors = grouping_operation(point_cloud, idx.contiguous().int())
        point_cloud_central = point_cloud.unsqueeze(2).repeat(1, 1, self.k, 1)
        edge_feature = torch.cat(
            [point_cloud_central, point_cloud_neighbors - point_cloud_central], dim=1
        )
        return edge_feature, idx

class denseconv(nn.Module):
    def __init__(self, growth_rate=64, k=16, in_channels=6):
        super(denseconv, self).__init__()
        self.k = k
        self.edge_feature_model = get_edge_feature(k=k)
        """
        input to conv1 is batch_size,2xn_dims,k,n_points
        """
        self.conv1 = nn.Sequential(
            Conv2d(
                in_channels=2 * in_channels,
                out_channels=growth_rate,
                kernel_size=[1, 1],
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            Conv2d(
                in_channels=growth_rate + in_channels,
                out_channels=growth_rate,
                kernel_size=[1, 1],
            ),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            Conv2d(
                in_channels=2 * growth_rate + in_channels,
                out_channels=growth_rate,
                kernel_size=[1, 1],
            ),
        )

    def forward(self, input):
        """
        y should be batch_size,in_channel,k,n_points
        """
        y, idx = self.edge_feature_model(input)  # B c k n
        inter_result = torch.cat(
            [self.conv1(y), input.unsqueeze(2).repeat([1, 1, self.k, 1])], dim=1
        )
        inter_result = torch.cat([self.conv2(inter_result), inter_result], dim=1)
        inter_result = torch.cat([self.conv3(inter_result), inter_result], dim=1)
        final_result = torch.max(inter_result, dim=2)[0]  
        return final_result, idx


class feature_extraction(nn.Module):
    def __init__(self, growth_rate=24, dense_n=3, k=16):
        super(feature_extraction, self).__init__()
        self.growth_rate = growth_rate
        self.dense_n = dense_n
        self.k = k
        self.input_channel = 3
        comp = self.growth_rate * 2
        """
        make sure to permute the input, the feature dimension is in the second one.
        input of conv1 is batch_size,num_dims,num_points
        """
        self.conv1 = nn.Sequential(
            Conv1d(
                in_channels=self.input_channel,
                out_channels=24,
                kernel_size=1,
                padding=0,
            ),
            nn.ReLU(),
        )
        self.denseconv1 = denseconv(
            in_channels=24, growth_rate=self.growth_rate, k=self.k
        )
        self.conv2 = nn.Sequential(
            Conv1d(in_channels=120, out_channels=comp, kernel_size=1), nn.ReLU()
        )
        self.denseconv2 = denseconv(
            in_channels=comp, growth_rate=self.growth_rate, k=self.k
        )
        self.conv3 = nn.Sequential(
            Conv1d(in_channels=240, out_channels=comp, kernel_size=1), nn.ReLU()
        )
        self.denseconv3 = denseconv(
            in_channels=comp, growth_rate=self.growth_rate, k=self.k
        )
        self.conv4 = nn.Sequential(
            Conv1d(in_channels=360, out_channels=comp, kernel_size=1), nn.ReLU()
        )
        self.denseconv4 = denseconv(
            in_channels=comp, growth_rate=self.growth_rate, k=self.k
        )

    def forward(self, input):
        l0_features = self.conv1(input)  # b,24,n
        l1_features, l1_index = self.denseconv1(l0_features)  
        l1_features = torch.cat([l1_features, l0_features], dim=1)  
        l2_features = self.conv2(l1_features)  # b,48,n
        l2_features, l2_index = self.denseconv2(l2_features)
        l2_features = torch.cat([l2_features, l1_features], dim=1)  
        l3_features = self.conv3(l2_features)  # b,48,n
        l3_features, l3_index = self.denseconv3(l3_features)
        l3_features = torch.cat([l3_features, l2_features], dim=1)  
        l4_features = self.conv4(l3_features)  # b,48,n
        l4_features, l4_index = self.denseconv4(l4_features)
        l4_features = torch.cat([l4_features, l3_features], dim=1)  
        return l4_features

class ps_expand(nn.Module):
    def __init__(self,up_ratio=4,in_channels=130):
        super(ps_expand, self).__init__()
        self.up_ratio = up_ratio
        self.conv1 = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=128, kernel_size=1), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            Conv1d(in_channels=128, out_channels=128, kernel_size=1), nn.ReLU()
        )
        self.deconv1 = nn.ConvTranspose1d(in_channels-2, in_channels-2, up_ratio, up_ratio, bias=True)
        self.grid = self.gen_grid(up_ratio).clone().detach()
        self.attention_unit = attention_unit(in_channels=in_channels)

    def forward(self, inputs):
        net = inputs  # b,240,n
        grid = self.grid.clone().to(net.device)
        grid = grid.unsqueeze(0).repeat(net.shape[0], 1, net.shape[2])  
        grid = grid.view([net.shape[0], -1, 2])  
        net = self.deconv1(net)
        net = torch.cat([net.permute(0, 2, 1).contiguous(), grid], dim=2)  
        net = net.permute(0, 2, 1).contiguous()
        net = self.attention_unit(net)
        net = self.conv1(net)
        net = self.conv2(net)
        return net

    class up_block(nn.Module):
        def __init__(self, up_ratio=4, in_channels=130, device=None):
            super(up_block, self).__init__()
            self.up_ratio = up_ratio
            self.conv1 = nn.Sequential(
                Conv1d(in_channels=in_channels, out_channels=128, kernel_size=1), nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                Conv1d(in_channels=128, out_channels=128, kernel_size=1), nn.ReLU()
            )
            self.grid = self.gen_grid(up_ratio).clone().detach()
            self.attention_unit = attention_unit(in_channels=in_channels)

        def forward(self, inputs):
            net = inputs  # b,128,n
            grid = self.grid.clone().to(net.device)
            grid = grid.unsqueeze(0).repeat(net.shape[0], 1, net.shape[2])  
            grid = grid.view([net.shape[0], -1, 2])  
            net = net.permute(0, 2, 1).contiguous()  
            net = net.repeat(1, self.up_ratio, 1)  
            net = torch.cat([net, grid], dim=2)  
            net = net.permute(0, 2, 1).contiguous()  
            net = self.attention_unit(net)
            net = self.conv1(net)
            net = self.conv2(net)
            return net
    def gen_grid(self, up_ratio):
        import math
        sqrted = int(math.sqrt(up_ratio)) + 1
        for i in range(1, sqrted + 1).__reversed__():
            if (up_ratio % i) == 0:
                num_x = i
                num_y = up_ratio // i
                break
        grid_x = torch.linspace(-0.2, 0.2, num_x)
        grid_y = torch.linspace(-0.2, 0.2, num_y)
        x, y = torch.meshgrid([grid_x, grid_y])
        grid = torch.stack([x, y], dim=-1)  # 2,2,2
        grid = grid.view([-1, 2])  # 4,2
        return grid

class refiner(nn.Module):
    def __init__(self, K1=16, K2=8, transform_dim=64,in_channel=128):
        super(refiner, self).__init__()
        self.K1 = K1 + 1
        self.K2 = K2
        self.KNN = KNN(self.K1)
        self.in_channel = in_channel
        self.transform_dim = transform_dim
        self.gamma_dim = self.transform_dim
        self.fc_gamma = nn.Sequential(
            nn.Conv2d(self.gamma_dim, 4 * self.gamma_dim, [1, 1], ),
            nn.BatchNorm2d(4 * self.gamma_dim),
            nn.ReLU(),
            nn.Conv2d(4 * self.gamma_dim, self.gamma_dim, [1, 1], ),
        )
        self.w_qs = nn.Conv2d(262, self.transform_dim, [1, 1])
        self.w_ks = nn.Conv2d(262, self.transform_dim, [1, 1])
        self.w_vs = nn.Conv2d(262, self.transform_dim, [1, 1])
        self.attention = attention_unit(in_channels=192)
        self.conv1 = nn.Conv1d(self.transform_dim, in_channel,1)

    def forward(self, feature, xyz):  
        # b n c
        _, idx = self.KNN(xyz, xyz)
        idx = idx[:, 1:, :]            
        idx_1 = idx[:, :self.K2, :]
        group_xyz = grouping_operation(xyz, idx.contiguous().int())
        group_xyz_1 = grouping_operation(xyz, idx_1.contiguous().int())
        rel_xyz = group_xyz - xyz[:, :, None, :]
        rel_xyz_1 = group_xyz_1 - xyz[:, :, None, :]
        group_fea = grouping_operation(feature, idx.contiguous().int())
        group_fea_1 = grouping_operation(feature, idx_1.contiguous().int())
        rel_feature = group_fea - feature[:, :, None, :]
        rel_feature_1 = group_fea_1-feature[:, :, None, :]
        rel_pos = torch.cat(
            [rel_xyz,
             xyz.unsqueeze(2).repeat(1, 1, self.K1-1, 1),
             rel_feature,
             feature.unsqueeze(2).repeat(1,1,self.K1-1,1)],
            dim=1
        )
        rel_pos_1 = torch.cat(
            [rel_xyz_1,
             xyz.unsqueeze(2).repeat(1, 1, self.K2, 1),
             rel_feature_1,
             feature.unsqueeze(2).repeat(1, 1, self.K2, 1)],
            dim=1
        )
        q, k, v = self.w_qs(rel_pos_1), self.w_ks(rel_pos), self.w_vs(rel_pos)
        attn = torch.einsum("bmnf,bmjf->bnjf", q, k)
        attn = F.softmax(attn, dim=-2)  
        res = torch.einsum("bnjf,bmjf->bmf", attn, v) 
        print("this is",attn.shape, res.shape)
        res = self.conv1(res)
        res = res + feature
        return res

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_point = args.num_point  
        if args.extraction == "dual":
            self.extractor = dual_extraction(transform_dim=args.transform_dim)   
            print("use dual extraction")
            self.in_channels = 128
        if args.up_module == 'ps':
            self.up_unit = ps_expand(args.up_ratio, in_channels=self.in_channels + 2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1),
        )
        self.refiner = refiner(K1=args.K1, K2=args.K2, transform_dim=64, in_channel=128)

    def forward(self, input):
        # Point Generator module
        features = self.extractor(input)   
        H = self.up_unit(features)  
        sparse_coord = self.conv1(H)
        coarse_coord = self.conv2(sparse_coord) 
        # Point Refiner module
        refine = self.refiner(H, coarse_coord)  
        refine_coord = self.conv3(refine)
        refine_coord = self.conv4(refine_coord)
        refine_coord = coarse_coord + refine_coord
        return coarse_coord, refine_coord

class node_shuffle(nn.Module):
    def __init__(self, scale=2, in_channels=128, out_channels=128):
        super(node_shuffle, self).__init__()
        self.scale = scale
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels = in_channels, out_channels=out_channels, kernel_size=1), nn.ReLU()
        )  # 480 256
        self.edge_conv = edge_conv(out_channels, out_channels * scale)

    def forward(self, inputs):
        B, C, N = inputs.shape[0], inputs.shape[1], inputs.shape[2]
        net = inputs  # b,480,n
        net = self.conv(net)  # 128
        net = self.edge_conv(net)  # b out_channel, 1 ,n
        net = net.squeeze(-2).contiguous()
        net = net.reshape([B, -1, self.scale * N]).contiguous()
        return net

class edge_conv(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super(edge_conv, self).__init__()
        self.k = k
        self.KNN = KNN(self.k + 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, inputs):  # b c n
        _, idx = self.KNN(inputs, inputs)
        idx = idx[:, 1:, :]
        pc_neighbors = grouping_operation(inputs, idx.contiguous().int())  # b c k n
        inputs = inputs.unsqueeze(-2).contiguous()
        pc_central = inputs.repeat([1, 1, self.k, 1]).contiguous()
        message = self.conv1(pc_neighbors - pc_central)
        x_center = self.conv2(inputs)
        edge_features = x_center + message
        edge_features = self.relu(edge_features)
        y = torch.max(edge_features, -2, keepdims=True)[0]
        return y

class attention_unit(nn.Module):
    def __init__(self, in_channels=130):
        super(attention_unit, self).__init__()
        self.convF = nn.Sequential(
            Conv1d(
                in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1
            ),
            nn.ReLU(),
        )
        self.convG = nn.Sequential(
            Conv1d(
                in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1
            ),
            nn.ReLU(),
        )
        self.convH = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.gamma = nn.Parameter(
            torch.zeros([1]).clone().detach().requires_grad_(True)
        )

    def forward(self, inputs):
        f = self.convF(inputs)
        g = self.convG(inputs)  # b,32,n
        h = self.convH(inputs)
        s = torch.matmul(g.permute(0, 2, 1).contiguous(), f)  # b,n,n
        beta = F.softmax(s, dim=2)  # b,n,n
        o = torch.matmul(h, beta)  # b,130,n
        x = self.gamma * o + inputs
        return x

class up_block(nn.Module):
    def __init__(self, up_ratio=4, in_channels=130, device=None):
        super(up_block, self).__init__()
        self.up_ratio = up_ratio
        self.conv1 = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=256, kernel_size=1), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            Conv1d(in_channels=256, out_channels=128, kernel_size=1), nn.ReLU()
        )
        self.grid = self.gen_grid(up_ratio).clone().detach()
        self.attention_unit = attention_unit(in_channels=in_channels)

    def forward(self, inputs):
        net = inputs  # b,128,n
        grid = self.grid.clone().to(net.device)
        grid = grid.unsqueeze(0).repeat(net.shape[0], 1, net.shape[2])  # b,4,2*n
        grid = grid.view([net.shape[0], -1, 2])  # b,4*n,2
        net = net.permute(0, 2, 1).contiguous()  # b,n,128
        net = net.repeat(1, self.up_ratio, 1)  # b,4n,128
        net = torch.cat([net, grid], dim=2)  # b,n*4,130
        net = net.permute(0, 2, 1).contiguous()  # b,130,n*4
        net = self.attention_unit(net)
        net = self.conv1(net)
        net = self.conv2(net)
        return net

    def gen_grid(self, up_ratio):
        import math
        sqrted = int(math.sqrt(up_ratio)) + 1
        for i in range(1, sqrted + 1).__reversed__():
            if (up_ratio % i) == 0:
                num_x = i
                num_y = up_ratio // i
                break
        grid_x = torch.linspace(-0.2, 0.2, num_x)
        grid_y = torch.linspace(-0.2, 0.2, num_y)
        x, y = torch.meshgrid([grid_x, grid_y])
        grid = torch.stack([x, y], dim=-1)  # 2,2,2
        grid = grid.view([-1, 2])  # 4,2
        return grid

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from common.configs import args
    from time import time
    from thop import profile
    f = Model(args).cuda()
    times = []
    for i in range(100):
        a = torch.randn([2, 3, 256])
        a = a.float().cuda()
        start = time()
        result = f(a)
        end = time()
        times.append((end - start) / 2)
        print((end - start))
    flops, params = profile(f, inputs=(a,))
    print(flops / 1024 / 1024 / 1024 / 2, params / 1024 / 1024 * 4)
    para_num = sum([p.numel() for p in f.parameters()])
    print("=== The number of parameters in model: {:.4f} K === ".format(float(para_num / 1e3)))

