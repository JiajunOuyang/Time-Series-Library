import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PrototypePool(nn.Module):
    """
    原型池驱动的偏移适配模块 (PPDM)
    采用在线动态更新的 Memory Bank 机制
    """
    def __init__(self, seq_len, pred_len, num_nodes, pool_size=1024, top_k=5, alpha=0.6):
        super(PrototypePool, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.pool_size = pool_size
        self.top_k = top_k
        self.alpha = alpha  # 相似度去重阈值
        
        # 预分配 Memory Bank 的 buffer (不参与梯度更新)
        # 存储: X_concat (用于检索), delta_mu, delta_sigma (用于预测)
        feature_dim = seq_len * num_nodes * 3 # concat(X_norm, mu, sigma) 展平

        # === 【修改前】 ===
        # target_dim = pred_len * num_nodes 
        
        # === 【修改后】 ===
        target_dim = num_nodes
        
        self.register_buffer('bank_features', torch.zeros(pool_size, feature_dim))
        self.register_buffer('bank_delta_mu', torch.zeros(pool_size, target_dim))
        self.register_buffer('bank_delta_sigma', torch.zeros(pool_size, target_dim))
        self.register_buffer('bank_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('is_ready', torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def _update_bank(self, x_concat, delta_mu, delta_sigma):
        """在线更新原型池 (仅在训练阶段调用)"""
        B = x_concat.size(0)
        ptr = int(self.bank_ptr)
        
        # 展平特征以便存储
        x_concat_flat = x_concat.view(B, -1)
        delta_mu_flat = delta_mu.view(B, -1)
        delta_sigma_flat = delta_sigma.view(B, -1)
        
        # 简单的相似度去重 (与库中现有原型的最大相似度)
        if self.is_ready:
            # === 【修改前】 ===
            # sim_matrix = F.cosine_similarity(x_concat_flat.unsqueeze(1), self.bank_features.unsqueeze(0), dim=-1)
            
            # === 【修改后】先 L2 归一化，再做矩阵乘法 ===
            x_flat_norm = F.normalize(x_concat_flat, p=2, dim=1)
            bank_norm = F.normalize(self.bank_features, p=2, dim=1)
            sim_matrix = torch.mm(x_flat_norm, bank_norm.transpose(0, 1)) # [B, pool_size]
            
            max_sim, _ = sim_matrix.max(dim=1)
            valid_mask = max_sim < self.alpha
            x_concat_flat = x_concat_flat[valid_mask]
            delta_mu_flat = delta_mu_flat[valid_mask]
            delta_sigma_flat = delta_sigma_flat[valid_mask]
            B = x_concat_flat.size(0)
            
        if B == 0:
            return

        # 环形队列更新
        if ptr + B <= self.pool_size:
            self.bank_features[ptr:ptr+B] = x_concat_flat
            self.bank_delta_mu[ptr:ptr+B] = delta_mu_flat
            self.bank_delta_sigma[ptr:ptr+B] = delta_sigma_flat
            ptr = (ptr + B) % self.pool_size
        else:
            overflow = (ptr + B) - self.pool_size
            self.bank_features[ptr:] = x_concat_flat[:B-overflow]
            self.bank_delta_mu[ptr:] = delta_mu_flat[:B-overflow]
            self.bank_delta_sigma[ptr:] = delta_sigma_flat[:B-overflow]
            self.bank_features[:overflow] = x_concat_flat[B-overflow:]
            self.bank_delta_mu[:overflow] = delta_mu_flat[B-overflow:]
            self.bank_delta_sigma[:overflow] = delta_sigma_flat[B-overflow:]
            ptr = overflow
            
        self.bank_ptr[0] = ptr
        if ptr > self.top_k: # 只要池子里有足够的样本就开始运作
            self.is_ready[0] = True

    def forward(self, x_norm, mu, sigma, future_mu=None, future_sigma=None):
        B, T, N = x_norm.shape
        x_concat = torch.cat([x_norm, mu, sigma], dim=-1) # [B, T, 3N]
        
        # 1. 训练时更新原型池
        if self.training and future_mu is not None and future_sigma is not None:
            # 获取真实未来分布以计算 delta
            # 此处假设 future_mu 维度为 [B, pred_len, N]
            delta_mu = future_mu - mu.mean(dim=1, keepdim=True)
            delta_sigma = future_sigma - sigma.mean(dim=1, keepdim=True)
            self._update_bank(x_concat, delta_mu, delta_sigma)
            
        # 2. 检索预测
        # 如果池子还没准备好 (比如最初的几个 batch)，退化为直接返回均值 0
        if not self.is_ready:
            return torch.zeros(B, self.pred_len, N, device=x_norm.device), \
                   torch.zeros(B, self.pred_len, N, device=x_norm.device)
                   
        x_concat_flat = x_concat.view(B, -1)
        
        # === 【修改前】 ===
        # 计算余弦相似度: [B, pool_size]
        # sim = F.cosine_similarity(x_concat_flat.unsqueeze(1), self.bank_features.unsqueeze(0), dim=-1)
        
        # === 【修改后】采用矩阵乘法，避免巨大的临时显存占用 ===
        x_flat_norm = F.normalize(x_concat_flat, p=2, dim=1)
        bank_norm = F.normalize(self.bank_features, p=2, dim=1)
        sim = torch.mm(x_flat_norm, bank_norm.transpose(0, 1)) # [B, pool_size]


        # Top-K 检索
        topk_sim, topk_idx = torch.topk(sim, self.top_k, dim=-1) # [B, K]
        
        # 权重归一化
        weights = F.softmax(topk_sim, dim=-1) # [B, K]
        
        # 提取对应的 delta_mu, delta_sigma: [B, K, pred_len * N]
        retrieved_delta_mu = self.bank_delta_mu[topk_idx]
        retrieved_delta_sigma = self.bank_delta_sigma[topk_idx]
        
        # 加权融合
        pred_delta_mu = torch.einsum('bk, bkd -> bd', weights, retrieved_delta_mu)
        pred_delta_sigma = torch.einsum('bk, bkd -> bd', weights, retrieved_delta_sigma)
        
        # 还原维度并加上当前序列的统计基准
        # === 【修改前】 ===
        # mu_pred = mu.mean(dim=1, keepdim=True) + pred_delta_mu.view(B, self.pred_len, N)
        # sigma_pred = sigma.mean(dim=1, keepdim=True) + pred_delta_sigma.view(B, self.pred_len, N)
        
        # === 【修改后】改为广播维度 [B, 1, N] ===
        mu_pred = mu.mean(dim=1, keepdim=True) + pred_delta_mu.view(B, 1, N)
        sigma_pred = sigma.mean(dim=1, keepdim=True) + pred_delta_sigma.view(B, 1, N)
        
        return mu_pred, sigma_pred


class DWC_TA(nn.Module):
    """双窗口协同时间注意力"""
    def __init__(self, seq_len, num_nodes, d_model=256, n_heads=8, local_window=24):
        super(DWC_TA, self).__init__()
        self.seq_len = seq_len
        self.w = local_window
        
        self.embed_local = nn.Linear(num_nodes, d_model)
        self.embed_global = nn.Linear(num_nodes, d_model)
        
        self.attn_local = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_global = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        self.sigma = nn.Parameter(torch.tensor(0.5)) # 门控参数
        
    def _get_sinusoidal_encoding(self, x):
        B, T, N = x.shape
        pe = torch.zeros(T, N, device=x.device)
        position = torch.arange(0, T, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, N, 2, dtype=torch.float, device=x.device) * -(math.log(10000.0) / N))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)

        # 【核心修改】：通过 [:N//2] 动态截断，完美兼容 N 为奇数的情况
        pe[:, 1::2] = torch.cos(position * div_term)[:, :N//2]

        return x + pe.unsqueeze(0)

    def forward(self, x_norm):
        B, T, N = x_norm.shape
        # 1. 位置编码注入
        x_pos = self._get_sinusoidal_encoding(x_norm)
        
        # 2. 窗口切分
        x_local = x_pos[:, -self.w:, :] # [B, w, N]
        x_global = x_pos # [B, T, N]
        
        # 3. 嵌入与注意力计算
        e_local = self.embed_local(x_local) # [B, w, D]
        e_global = self.embed_global(x_global) # [B, T, D]
        
        a_local, _ = self.attn_local(e_local, e_local, e_local)
        a_global, _ = self.attn_global(e_global, e_global, e_global)
        
        # 4. 局部窗口零填充对齐
        pad_len = T - self.w
        a_local_padded = F.pad(a_local.transpose(1, 2), (0, pad_len), "constant", 0).transpose(1, 2)
        
        # 5. 门控融合
        gate = torch.sigmoid(self.sigma)
        A_T = gate * a_local_padded + (1 - gate) * a_global
        
        return A_T # [B, T, D]


class DNA_SA(nn.Module):
    """动态邻域感知空间注意力"""
    def __init__(self, seq_len, num_nodes, d_model=256, n_heads=8):
        super(DNA_SA, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.embed = nn.Linear(seq_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x_norm):
        B, T, N = x_norm.shape
        # 注意：空间注意力是对变量(N)进行交互，所以转置输入
        x_inv = x_norm.transpose(1, 2) # [B, N, T]
        
        # 1. 映射与变量聚合
        e_s = self.embed(x_inv) # [B, N, D]
        e_v = self.norm(e_s) # 这里的 e_s 已经消除了时间步维度，等价于原论文的聚合
        
        # 2. 动态邻域相关性矩阵 R
        # L2归一化后做矩阵乘法得到余弦相似度
        e_v_l2 = F.normalize(e_v, p=2, dim=-1)
        G = torch.matmul(e_v_l2, e_v_l2.transpose(-1, -2)) # [B, N, N]
        
        # 动态掩码 M
        gamma = G.mean(dim=-1, keepdim=True) # 每个变量的关联均值作为阈值
        M = (G.abs() >= gamma).float()
        R = G * M # [B, N, N]
        
        # 3. 邻域引导的空间注意力
        Q = self.W_q(e_s).view(B, N, self.n_heads, -1).transpose(1, 2) # [B, h, N, d]
        K = self.W_k(e_s).view(B, N, self.n_heads, -1).transpose(1, 2)
        V = self.W_v(e_s).view(B, N, self.n_heads, -1).transpose(1, 2)
        
        # Q * R * K^T 的实现：将 R 融入 attention score
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_model // self.n_heads)
        scores = scores * R.unsqueeze(1) # R 广播到多头维度 [B, 1, N, N]
        
        attn = F.softmax(scores, dim=-1)
        A_S = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, -1)
        
        # 恢复到 [B, T, D] 的形状语义 (方便后续与时间特征交互)
        # 这里使用一层全连接将其还原回时间维度语义
        A_S = self.out_proj(A_S).unsqueeze(1).repeat(1, T, 1, 1).mean(dim=2) # 简化对齐逻辑
        return A_S # [B, T, D]


class ST_CAF(nn.Module):
    """时空跨注意力融合机制"""
    def __init__(self, d_model=256, n_heads=8):
        super(ST_CAF, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, A_S, A_T):
        # A_S 引导 (Query), A_T 支撑 (Key, Value)
        A_ST, _ = self.cross_attn(query=A_S, key=A_T, value=A_T)
        
        # 残差增强
        F_fused = A_T + self.linear(A_ST)
        return F_fused


class Model(nn.Module):
    """
    ProSTA 主模型: 面向渐进分布偏移的原型时空预测
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.num_nodes = getattr(args, 'enc_in', 321) # 兼容TSlib
        
        # ProSTA 专属超参数 (如果 args 里没有则使用默认值)
        self.d_model = getattr(args, 'st_dim', 256)
        self.local_window = getattr(args, 'local_window', 24)
        self.top_k = getattr(args, 'top_k', 5)
        self.alpha = getattr(args, 'alpha', 0.6)
        
        # 1. 原型池驱动的偏移适配模块 (PPDM)
        self.ppdm = PrototypePool(
            seq_len=self.seq_len, 
            pred_len=self.pred_len, 
            num_nodes=self.num_nodes,
            top_k=self.top_k,
            alpha=self.alpha
        )
        
        # 2. 时空注意力融合模块 (STAFM)
        self.dwc_ta = DWC_TA(self.seq_len, self.num_nodes, self.d_model, n_heads=8, local_window=self.local_window)
        self.dna_sa = DNA_SA(self.seq_len, self.num_nodes, self.d_model, n_heads=8)
        self.st_caf = ST_CAF(self.d_model, n_heads=8)
        
        # 3. 输出层 (将融合特征映射至预测窗口)
        self.fc_out = nn.Linear(self.seq_len, self.pred_len)
        self.proj_nodes = nn.Linear(self.d_model, self.num_nodes)

    def _multiscale_norm(self, x):
        """简化的加权多尺度标准化，剥离出当前分布特征"""
        mu = x.mean(dim=1, keepdim=True)
        sigma = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x - mu) / sigma
        return x_norm, mu.expand_as(x), sigma.expand_as(x)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, T, N = x_enc.shape
        
        # 1. 结构不变量提取与解耦
        x_norm_input, mu_input, sigma_input = self._multiscale_norm(x_enc)
        
        # 提取训练阶段的真实未来分布 (用于在线更新原型池)
        future_mu, future_sigma = None, None
        if self.training and x_dec is not None:
            # TSlib 的 x_dec 包含 label_len + pred_len，我们只取后 pred_len 个时间步
            real_future = x_dec[:, -self.pred_len:, :]
            future_mu = real_future.mean(dim=1, keepdim=True)
            future_sigma = torch.sqrt(real_future.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            
        # 2. PPDM: 原型池预测未来分布
        mu_pred, sigma_pred = self.ppdm(x_norm_input, mu_input, sigma_input, future_mu, future_sigma)
        
        # 3. STAFM: 时空特征建模
        A_T = self.dwc_ta(x_norm_input)       # [B, T, D]
        A_S = self.dna_sa(x_norm_input)       # [B, T, D]
        F_fused = self.st_caf(A_S, A_T)       # [B, T, D]
        
        # 4. 预测输出与反归一化
        # 先经过节点投影 [B, T, N]，再经过时间投影 [B, pred_len, N]
        y_norm_trans = self.proj_nodes(F_fused).transpose(1, 2)
        y_norm_pred = self.fc_out(y_norm_trans).transpose(1, 2)
        
        # 结合 PPDM 预测的基准进行反归一化
        Y_pred = y_norm_pred * sigma_pred + mu_pred
        
        return Y_pred
        