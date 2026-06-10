import torch
import torch.nn as nn
from layers.RevIN import RevIN
import torch.nn.functional as F
import math

class HilbertFeatureBank(nn.Module):
    def __init__(self, eps=1e-6, normalize_views=True):
        super().__init__()
        self.eps = eps
        self.normalize_views = normalize_views

    def _analytic_signal(self, x):
        B, C, L = x.shape  # x: [B,C,L]
        Xf = torch.fft.fft(x, dim=-1)  # 沿时间维做FFT
        h = torch.zeros(L, device=x.device, dtype=x.dtype)  # Hilbert频域mask
        if L % 2 == 0:
            h[0] = 1.0; h[L // 2] = 1.0; h[1:L // 2] = 2.0  # 偶数长度
        else:
            h[0] = 1.0; h[1:(L + 1) // 2] = 2.0  # 奇数长度
        z = torch.fft.ifft(Xf * h.view(1, 1, L), dim=-1)  # 解析信号
        return z

    def _norm_view(self, v):
        mean = v.mean(dim=-1, keepdim=True)  # 时间维均值
        std = v.std(dim=-1, keepdim=True).clamp_min(self.eps)  # 时间维标准差
        return (v - mean) / std  # 每个变量内部标准化

    def forward(self, x):
        if x.dim() != 3: raise ValueError(f"Expected x shape [B,C,L], but got {x.shape}")  # 输入检查
        z = self._analytic_signal(x)  # [B,C,L] complex
        amp = torch.abs(z).to(dtype=x.dtype)  # 包络
        log_amp = torch.log(amp + self.eps)  # log包络
        phase = torch.angle(z).to(dtype=x.dtype)  # 相位
        cos_phase = torch.cos(phase)  # cos相位
        sin_phase = torch.sin(phase)  # sin相位

        tokens = [log_amp, cos_phase, sin_phase]  # 只保留核心Hilbert视角
        if self.normalize_views:
            tokens = [self._norm_view(v) for v in tokens]  # 每个view单独标准化

        views = torch.stack(tokens, dim=2)  # [B,C,3,L]
        return views

class HilbertSummary(nn.Module):
    def __init__(self, num_views=3, d_router=64, dropout=0.1):
        super().__init__()
        self.num_views = num_views
        self.d_router = d_router
        self.in_dim = num_views * 2  # mean/last
        self.summary_mlp = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(self.in_dim, d_router),
            nn.GELU()
        )

    def forward(self, views):
        if views.dim() != 4: raise ValueError(f"Expected views shape [B,C,M,L], but got {views.shape}")  # 输入检查
        B, C, M, L = views.shape  # [B,C,M,L]
        if M != self.num_views: raise ValueError(f"Expected num_views={self.num_views}, but got M={M}")  # view数量检查

        mean = views.mean(dim=-1)  # [B,C,M]，窗口整体振荡状态
        last = views[..., -1]  # [B,C,M]，预测起点处的瞬时振荡状态

        summary = torch.cat([mean, last], dim=-1)  # [B,C,2M]
        guide = self.summary_mlp(summary)  # [B,C,d_router]
        return guide

class LinearEncoder_Multihead(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 4, feature_num: int = 21, dropout: float = 0.5, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.embed_dim = embed_dim  # 序列embedding维度，比如512
        self.num_heads = num_heads  # 多头数量，比如4
        self.feature_num = feature_num  # 特征变量数量，比如21
        self.head_dim = embed_dim // num_heads  # 每个头的维度，比如128
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 对最后一维512做V映射
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 多头拼接后再映射回512
        self.weight_mat = nn.Parameter(torch.randn(num_heads, feature_num, feature_num))  # 每个头一个[21,21]可学习注意力矩阵
        self.attn_dropout = nn.Dropout(dropout)  # 注意力矩阵dropout
        self.out_dropout = nn.Dropout(dropout)  # 输出dropout

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape  # x:[B,21,512]
        assert N == self.feature_num, f"输入特征数N={N}必须等于feature_num={self.feature_num}"
        assert D == self.embed_dim, f"输入embedding维度D={D}必须等于embed_dim={self.embed_dim}"
        V = self.v_proj(x)  # [B,21,512]
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,4,21,128]
        A = F.softplus(self.weight_mat)  # [4,21,21]，保证权重为正
        A = F.normalize(A, p=1, dim=-1)  # [4,21,21]，每一行归一化
        A = self.attn_dropout(A)  # [4,21,21]
        out = torch.matmul(A.unsqueeze(0), V)  # [1,4,21,21] @ [B,4,21,128] -> [B,4,21,128]
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B,21,512]
        out = self.out_dropout(self.out_proj(out))  # [B,21,512]
        return out

class HilbertOscillationRouter(nn.Module):
    def __init__(self, d_model=512, d_router=64, num_heads=4, feature_num=21, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # 高维变量表征维度
        self.d_router = d_router  # Hilbert路由向量维度
        self.num_heads = num_heads  # 多头线性增强头数
        self.feature_num = feature_num  # 变量数量
        self.dropout = dropout  # dropout比例

        self.osc_prior_proj = nn.Sequential(nn.Linear(d_router, d_model))  # Hilbert guide -> 振荡路由先验
        self.prior_norm = nn.LayerNorm(d_model)  # 振荡先验归一化，可稳定注入
        self.prior_alpha = nn.Parameter(torch.tensor(-4.0))  # 振荡先验注入强度，初始很小

        self.reliability_alpha = nn.Parameter(torch.tensor(0.0))  # 可靠性调制强度，初始为0
        self.reliability_gate = nn.Sequential(nn.LayerNorm(d_router), nn.Linear(d_router, 1), nn.Sigmoid())  # [B,C,d_router]->[B,C,1]
        self.out_proj = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.d_model, self.d_model), nn.GELU(), )

        self.channel_encoder = LinearEncoder_Multihead(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            feature_num=self.feature_num,
            dropout=dropout
        )  # 线性通道增强编码器


    def forward(self, x_embed, guide):
        if x_embed.dim() != 3: raise ValueError(f"Expected x_embed shape [B,C,D], but got {x_embed.shape}")
        if guide.dim() != 3: raise ValueError(f"Expected guide shape [B,C,d_router], but got {guide.shape}")

        B, C, D = x_embed.shape  # x_embed:[B,C,D]
        if C != self.feature_num: raise ValueError(f"Expected feature_num={self.feature_num}, but got C={C}")
        if D != self.d_model: raise ValueError(f"Expected d_model={self.d_model}, but got D={D}")
        if guide.size(-1) != self.d_router: raise ValueError(f"Expected d_router={self.d_router}, but got {guide.size(-1)}")

        osc_prior = self.osc_prior_proj(guide)  # [B,C,D]，由Hilbert振荡摘要生成的路由先验
        osc_prior = self.prior_norm(osc_prior)  # [B,C,D]，稳定先验尺度
        prior_scale = torch.sigmoid(self.prior_alpha)  # 标量，初始约0.018，避免训练初期过强注入
        routed_embed = x_embed + prior_scale * osc_prior  # [B,C,D]，注入振荡先验后的路由化变量表征

        encoded_embed = self.channel_encoder(routed_embed)  # [B,C,D]，线性通道增强后的变量表征

        reliability = self.reliability_gate(guide)  # [B,C,1]，Hilbert振荡先验的可靠性估计
        reliability_scale = 1.0 + 0.1 * torch.tanh(self.reliability_alpha) * (reliability - 0.5)  # 初始为1
        # enhanced_embed = routed_embed + reliability_scale * encoded_embed  # [B,C,D]，可靠性调制后的残差融合
        enhanced_embed = reliability_scale * encoded_embed
        enhanced_out = self.out_proj(enhanced_embed)
        return enhanced_out

        # encoded_embed = self.channel_encoder(routed_embed)
        # enhanced_out = self.out_proj(x_embed)
        # return enhanced_out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin
        self.revin_layer = RevIN(self.enc_in, affine=True)

        self.tem = torch.nn.Parameter(torch.zeros(1,self.enc_in, self.d_model), requires_grad=True)
        self.tem02 = torch.nn.Parameter(torch.zeros(1,self.enc_in, self.d_model), requires_grad=True)
        self.tem03 = torch.nn.Parameter(torch.zeros(1, self.enc_in, self.d_model), requires_grad=True)
        self.tem04 = torch.nn.Parameter(torch.zeros(1, self.enc_in, self.seq_len), requires_grad=True)

        # self.channelAggregator = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, batch_first=True,dropout=0.5)
        self.model = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.seq_len, self.d_model), nn.GELU(), )
        self.model02 = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.d_model, self.d_model), nn.GELU())
        self.input_proj = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.d_model, self.d_model),nn.GELU(),)
        self.output_proj = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.d_model, self.pred_len),)

        # ==============================================================================================================
        self.d_router = 64
        self.hilbert_bank = HilbertFeatureBank(eps=1e-6, normalize_views=True)
        self.hilbert_summary = HilbertSummary(num_views=3, d_router=self.d_router, dropout=self.dropout)

        self.linear_encoder = HilbertOscillationRouter(
            d_model=self.d_model,
            d_router=self.d_router,
            num_heads=4,
            feature_num=self.enc_in,
            dropout=0.)

    def forward(self, x, cycle_index):
        if self.use_revin:
            x = self.revin_layer(x, mode='norm')

        x_input = x.permute(0, 2, 1)

        x_exp = self.model(x_input)

        views = self.hilbert_bank(x_input)  # [B,C,M,L]
        guide = self.hilbert_summary(views)  # [B,C,64]
        enc_out = self.linear_encoder(x_exp, guide)  # [B,C,D]，内部已经做 x_exp + attn_out

        x_out = self.output_proj(enc_out)

        output = x_out.permute(0, 2, 1)
        if self.use_revin:
            output = self.revin_layer(output, mode='denorm')
        return output
