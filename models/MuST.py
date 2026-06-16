import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MDAM(nn.Module):
    """
    多尺度分布预测模块 (Multi-scale Distribution Adaptation Module)
    """
    def __init__(self, scales, seq_len, pred_len, num_nodes):
        super(MDAM, self).__init__()
        self.scales = sorted(scales, reverse=True) 
        self.K = len(self.scales)
        self.c_K = self.scales[-1]
        self.num_nodes = num_nodes
        
        self.L_K = seq_len // self.c_K
        self.S_K = pred_len // self.c_K
        self.pred_len = pred_len
        
        # 统计特征预测子模块 (SFP)
        self.time_proj_mu = nn.Linear(self.L_K, self.S_K)
        self.time_proj_sigma = nn.Linear(self.L_K, self.S_K)
        
        self.mlp_mu = nn.Sequential(
            nn.Linear(self.K * num_nodes, 512),
            nn.ReLU(),
            nn.Linear(512, self.K * num_nodes)
        )
        
        self.mlp_sigma = nn.Sequential(
            nn.Linear(self.K * num_nodes, 512),
            nn.ReLU(),
            nn.Linear(512, self.K * num_nodes),
            nn.Softplus() 
        )
        
        self.eps = 1e-5

    def forward(self, x):
        B, T, N = x.shape
        mu_dict, sigma_dict, x_norm_dict = {}, {}, {}
        
        # 1. 多尺度标准化
        for k, c_k in enumerate(self.scales):
            L_k = T // c_k
            x_reshaped = x.view(B, L_k, c_k, N)
            mu = x_reshaped.mean(dim=2, keepdim=True)
            sigma = torch.sqrt(x_reshaped.var(dim=2, keepdim=True, unbiased=False) + self.eps)
            x_norm = (x_reshaped - mu) / sigma
            x_norm_dict[c_k] = x_norm.view(B, T, N)
            mu_dict[c_k] = mu.squeeze(2)
            sigma_dict[c_k] = sigma.squeeze(2)
            
        # 2. 统计特征对齐与上采样
        mu_aligned, sigma_aligned = [], []
        for c_k in self.scales:
            factor = c_k // self.c_K
            mu_up = F.interpolate(mu_dict[c_k].transpose(1, 2), scale_factor=factor, mode='nearest').transpose(1, 2)
            sigma_up = F.interpolate(sigma_dict[c_k].transpose(1, 2), scale_factor=factor, mode='nearest').transpose(1, 2)
            mu_aligned.append(mu_up)
            sigma_aligned.append(sigma_up)
            
        H_mu = torch.cat(mu_aligned, dim=-1)
        H_sigma = torch.cat(sigma_aligned, dim=-1)
        
        # 3. 统计特征预测
        H_mu_future = self.time_proj_mu(H_mu.transpose(1, 2)).transpose(1, 2)
        H_sigma_future = self.time_proj_sigma(H_sigma.transpose(1, 2)).transpose(1, 2)
        H_mu_pred = self.mlp_mu(H_mu_future)        
        H_sigma_pred = self.mlp_sigma(H_sigma_future) 
        
        # 4. 预测特征还原
        pred_mu_dict, pred_sigma_dict = {}, {}
        mu_preds = torch.split(H_mu_pred, self.num_nodes, dim=-1)
        sigma_preds = torch.split(H_sigma_pred, self.num_nodes, dim=-1)
        
        for idx, c_k in enumerate(self.scales):
            # 目标时间步长 (保证至少为 1)
            target_length = max(1, self.S_K // (c_k // self.c_K))
            
            # 使用自适应平均池化来解决长度不匹配问题
            mu_down = F.adaptive_avg_pool1d(mu_preds[idx].transpose(1, 2), target_length).transpose(1, 2)
            sigma_down = F.adaptive_avg_pool1d(sigma_preds[idx].transpose(1, 2), target_length).transpose(1, 2)
            
            pred_mu_dict[c_k] = mu_down
            pred_sigma_dict[c_k] = sigma_down

        return x_norm_dict, pred_mu_dict, pred_sigma_dict


class STFFM(nn.Module):
    """
    时空特征融合模块 (SpatioTemporal Feature Fusion Module)
    已整合细节架构图中的双重交叉注意力逻辑
    """
    def __init__(self, seq_len, pred_len, num_nodes, d_model=512, n_heads=8):
        super(STFFM, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.time_embed = nn.Linear(seq_len, d_model)
        self.space_embed = nn.Linear(seq_len, d_model)
        
        self.W_Q_t = nn.Linear(d_model, d_model)
        self.W_K_t = nn.Linear(d_model, d_model)
        self.W_V_t = nn.Linear(d_model, d_model)
        
        self.W_Q_s = nn.Linear(d_model, d_model)
        self.W_K_s = nn.Linear(d_model, d_model)
        self.W_V_s = nn.Linear(d_model, d_model)
        
        self.norm_T = nn.LayerNorm(d_model)
        self.norm_S = nn.LayerNorm(d_model)
        self.conv_T = nn.Linear(d_model, d_model)
        self.conv_S = nn.Linear(d_model, d_model)
        
        self.conv_Z_T = nn.Linear(d_model, d_model)
        self.conv_Z_S = nn.Linear(d_model, d_model)
        self.norm_Z_T = nn.LayerNorm(d_model)
        self.norm_Z_S = nn.LayerNorm(d_model)
        
        self.sigma = nn.Parameter(torch.tensor(0.5))
        self.out_proj = nn.Linear(d_model, pred_len)

    def forward(self, x_norm):
        B, T, N = x_norm.shape
        D = self.d_model
        
        # --- 1. 时间依赖注意力 (倒转视角) ---
        x_inv = x_norm.transpose(1, 2)
        E_t = self.time_embed(x_inv)
        Q_t, K_t, V_t = self.W_Q_t(E_t), self.W_K_t(E_t), self.W_V_t(E_t)
        
        scores_t = torch.matmul(Q_t, K_t.transpose(-1, -2)) / math.sqrt(D // self.n_heads)
        attn_t = F.softmax(scores_t, dim=-1)
        A_T = torch.matmul(attn_t, V_t) 
        
        # --- 2. 空间关联注意力 (相关性引导) ---
        E_s = self.space_embed(x_inv)
        E_s_norm = F.layer_norm(E_s, E_s.size()[1:])

        # 原代码：
        # R = F.cosine_similarity(E_s_norm.unsqueeze(2), E_s_norm.unsqueeze(1), dim=-1)

        # 优化后的代码：利用矩阵乘法计算余弦相似度
        # E_s_norm 已经是层归一化过的 [B, N, D]
        # 对 D 维度进行 L2 归一化
        E_s_l2 = F.normalize(E_s_norm, p=2, dim=-1)
        # 通过矩阵乘法得到 [B, N, N] 的相似度矩阵，避免了 [B, N, N, D] 中间变量的产生
        R = torch.matmul(E_s_l2, E_s_l2.transpose(-1, -2))

        Q_s, K_s, V_s = self.W_Q_s(E_s), self.W_K_s(E_s), self.W_V_s(E_s)
        
        scores_s = torch.matmul(Q_s, K_s.transpose(-1, -2)) / math.sqrt(D // self.n_heads)
        attn_s = F.softmax(scores_s * R, dim=-1) 
        A_S = torch.matmul(attn_s, V_s) 
        
        # --- 3. 门控自适应融合 (双重交叉注意力) ---
        # 对应细节图中的 Norm & Conv 处理
        F_A = self.conv_T(self.norm_T(A_T)) # [B, N, D] (即图中的 F^A)
        F_B = self.conv_S(self.norm_S(A_S)) # [B, N, D] (即图中的 F^B)
        
        Q1, K1, V1 = F_A, F_A, F_A
        Q2, K2, V2 = F_B, F_B, F_B
        
        # S1 是变量维度注意力 (C x C): Q1 @ K2^T -> [B, N, D] @ [B, D, N] -> [B, N, N]
        S1 = F.softmax(torch.matmul(Q1, K2.transpose(-1, -2)), dim=-1) 
        
        # S2 是特征维度注意力 (HW x HW): Q2^T @ K1 -> [B, D, N] @ [B, N, D] -> [B, D, D]
        S2 = F.softmax(torch.matmul(Q2.transpose(-1, -2), K1), dim=-1) 
        
        # Z1 = S1^T @ V1 @ S2 [依据细节图公式修正]
        # [B, N, N] @ [B, N, D] @ [B, D, D] -> [B, N, D]
        Z1 = torch.matmul(torch.matmul(S1.transpose(-1, -2), V1), S2)
        
        # Z2 = S1 @ V2 @ S2^T [依据细节图公式修正]
        # [B, N, N] @ [B, N, D] @ [B, D, D] -> [B, N, D]
        Z2 = torch.matmul(torch.matmul(S1, V2), S2.transpose(-1, -2))
        
        # 残差连接与最终融合
        A_T_hat = A_T + self.norm_Z_T(self.conv_Z_T(Z1))
        A_S_hat = A_S + self.norm_Z_S(self.conv_Z_S(Z2))
        
        gate = torch.sigmoid(self.sigma)
        F_fused = gate * A_T_hat + (1 - gate) * A_S_hat 
        
        return self.out_proj(F_fused).transpose(1, 2)


class Model(nn.Module):
    """
    MuST 主模型: 面向全域分布偏移的多尺度时空特征融合预测
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.num_nodes = args.enc_in
        self.scales = getattr(args, 'scales', [168, 24, 12, 6]) 
        
        self.mdam = MDAM(self.scales, self.seq_len, self.pred_len, self.num_nodes)
        self.stffms = nn.ModuleList([
            STFFM(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                num_nodes=self.num_nodes,
                d_model=getattr(args, 'd_model', 512),
                n_heads=getattr(args, 'n_heads', 8)
            ) for _ in range(len(self.scales))
        ])

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, _, N = x.shape
        # 1. 分布特征解耦
        x_norm_dict, pred_mu_dict, pred_sigma_dict = self.mdam(x)
        
        Y_preds = []
        # 2. 多尺度建模与反标准化
        for idx, c_k in enumerate(self.scales):
            Y_norm_k = self.stffms[idx](x_norm_dict[c_k])
            
            mu_expand = F.interpolate(pred_mu_dict[c_k].transpose(1, 2), size=self.pred_len, mode='linear').transpose(1, 2)
            sigma_expand = F.interpolate(pred_sigma_dict[c_k].transpose(1, 2), size=self.pred_len, mode='linear').transpose(1, 2)
            
            Y_preds.append(Y_norm_k * sigma_expand + mu_expand)
            
        # 3. 均值融合
        return sum(Y_preds) / len(self.scales)
