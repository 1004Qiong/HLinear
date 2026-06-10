import torch
import torch.nn as nn
from layers.RevIN import RevIN
import torch.nn.functional as F

class HilbertFeatureBank(nn.Module):
    def __init__(self, eps=1e-6, use_raw=True, use_inst_freq=True, use_amp_diff=True, normalize_views=True):
        super().__init__()
        self.eps = eps
        self.use_raw = use_raw
        self.use_inst_freq = use_inst_freq
        self.use_amp_diff = use_amp_diff
        self.normalize_views = normalize_views

    def _analytic_signal(self, x):
        B, C, L = x.shape
        Xf = torch.fft.fft(x, dim=-1)
        h = torch.zeros(L, device=x.device, dtype=x.dtype)
        if L % 2 == 0:
            h[0] = 1.0
            h[L // 2] = 1.0
            h[1:L // 2] = 2.0
        else:
            h[0] = 1.0
            h[1:(L + 1) // 2] = 2.0
        z = torch.fft.ifft(Xf * h.view(1, 1, L), dim=-1)
        return z

    def _norm_view(self, v):
        mean = v.mean(dim=-1, keepdim=True)
        std = v.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (v - mean) / std

    def _first_order_diff(self, v):
        diff = v[..., 1:] - v[..., :-1]
        pad = torch.zeros_like(diff[..., :1])
        return torch.cat([pad, diff], dim=-1)

    def _circular_phase_diff(self, phase):
        diff = phase[..., 1:] - phase[..., :-1]
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        pad = torch.zeros_like(diff[..., :1])
        return torch.cat([pad, diff], dim=-1)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B,C,L], but got {x.shape}")
        z = self._analytic_signal(x)
        amp = torch.abs(z).to(dtype=x.dtype)
        log_amp = torch.log(amp + self.eps)
        phase = torch.angle(z).to(dtype=x.dtype)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        tokens = []
        if self.use_raw:
            tokens.append(x)
        tokens.extend([log_amp, cos_phase, sin_phase])
        if self.use_inst_freq:
            tokens.append(self._circular_phase_diff(phase))
        if self.use_amp_diff:
            tokens.append(self._first_order_diff(log_amp))
        if self.normalize_views:
            tokens = [self._norm_view(v) for v in tokens]
        views = torch.stack(tokens, dim=2)
        return views


class HilbertSummary(nn.Module):
    def __init__(self, num_views=6, d_router=64, dropout=0.1, eps=1e-6):
        super().__init__()
        self.num_views = num_views
        self.d_router = d_router
        self.eps = eps
        self.in_dim = num_views * 4
        self.summary_mlp = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Dropout(0.5),
            nn.Linear(self.in_dim, d_router),
            nn.GELU(),
        )

    def forward(self, views):
        if views.dim() != 4:
            raise ValueError(f"Expected views shape [B,C,M,L], but got {views.shape}")
        B, C, M, L = views.shape
        if M != self.num_views:
            raise ValueError(f"Expected num_views={self.num_views}, but got M={M}")
        mean = views.mean(dim=-1)
        std = views.std(dim=-1).clamp_min(self.eps)
        last = views[..., -1]
        slope = views[..., -1] - views[..., 0]
        summary = torch.cat([mean, std, last, slope], dim=-1)
        guide = self.summary_mlp(summary)
        return guide


class LinearEncoder_Multihead(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 4, feature_num: int = 21, dropout: float = 0.5, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feature_num = feature_num
        self.head_dim = embed_dim // num_heads
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.weight_mat = nn.Parameter(torch.randn(num_heads, feature_num, feature_num))
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape  # x: [B,21,512]
        assert N == self.feature_num, f"The input feature number N={N} must be equal to feature_num={self.feature_num}"
        assert D == self.embed_dim, f"The input embedding dimension D={D} must be equal to embed_dim={self.embed_dim}"
        V = self.v_proj(x)  # [B,21,512]
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,4,21,128]
        A = F.softplus(self.weight_mat)  # [4,21,21]
        A = F.normalize(A, p=1, dim=-1)  # [4,21,21]
        A = self.attn_dropout(A)  # [4,21,21]
        out = torch.matmul(A.unsqueeze(0), V)  # [1,4,21,21] @ [B,4,21,128] -> [B,4,21,128]
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B,21,512]
        out = self.out_dropout(self.out_proj(out))  # [B,21,512]
        return out


class Linear_Channel(nn.Module):
    def __init__(self, d_model=512, d_router=64, num_heads=4, feature_num=21, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_router = d_router
        self.feature_num = feature_num
        self.num_heads = num_heads
        self.dropout =dropout

        self.hilbert_q_proj = nn.Sequential(nn.Linear(d_router, d_model),)

        self.q_norm = nn.LayerNorm(d_model)

        self.query_alpha = nn.Parameter(torch.tensor(-4.0))
        self.rel_alpha = nn.Parameter(torch.tensor(0.0))
        self.rel_gate = nn.Sequential(nn.LayerNorm(d_router), nn.Linear(d_router, 1), )
        self.LinearEncoder = LinearEncoder_Multihead(embed_dim=self.d_model, num_heads=self.num_heads,
                                                     feature_num=self.feature_num, dropout=dropout)

    def forward(self, x_embed, guide):
        if x_embed.dim() != 3: raise ValueError(f"Expected x_embed shape [B,C,D], but got {x_embed.shape}")
        if guide.dim() != 3: raise ValueError(f"Expected guide shape [B,C,d_router], but got {guide.shape}")

        B, C, D = x_embed.shape
        if C != self.feature_num: raise ValueError(f"Expected feature_num={self.feature_num}, but got C={C}")
        if D != self.d_model: raise ValueError(f"Expected d_model={self.d_model}, but got D={D}")
        if guide.size(-1) != self.d_router: raise ValueError(
            f"Expected d_router={self.d_router}, but got {guide.size(-1)}")

        osc_prior = self.hilbert_q_proj(guide) # [B,C,D]
        prior_scale = torch.sigmoid(self.query_alpha)
        routed_embed = x_embed + prior_scale * osc_prior # [B,C,D]

        # routed_embed = x_embed + 0.3 * osc_prior
        enc_out = self.LinearEncoder(routed_embed) # [B,C,D]

        rel = self.rel_gate(guide) # [B,C,1]
        rel_mod = 1.0 + 0.1 * torch.tanh(self.rel_alpha) * (rel - 0.5)
        out = routed_embed + rel_mod * enc_out

        return out

class OPE(nn.Module):
    def __init__(self, dropout, d_router):
        super(OPE, self).__init__()
        self.dropout = dropout
        self.d_router = d_router

        self.hilbert_bank = HilbertFeatureBank(eps=1e-6, use_raw=False, use_inst_freq=True, use_amp_diff=True, normalize_views=True)
        self.hilbert_summary = HilbertSummary(num_views=5, d_router=self.d_router, dropout=self.dropout)

    def forward(self, x_input):

        views = self.hilbert_bank(x_input)  # [B,C,M,L]
        guide = self.hilbert_summary(views)  # [B,C,64]

        return guide


class OCE(nn.Module):
    def __init__(self, dropout=0., d_model=512, d_router=64, enc_in=21):
        super(OCE, self).__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.d_router = d_router
        self.enc_in = enc_in

        self.input_proj = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.d_model, self.d_model), nn.GELU(), )
        self.Linear_Channel_Mixing = Linear_Channel(
            d_model=self.d_model,
            d_router=self.d_router,
            num_heads=4,
            feature_num=self.enc_in,
            dropout=self.dropout
        )

    def forward(self, x_exp, G_h):
        chanel = self.Linear_Channel_Mixing(x_exp, G_h)  # [B,C,D]，内部已经做 x_exp + attn_out
        x_out = self.input_proj(chanel)
        return x_out


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
        self.d_router = configs.d_router

        self.Linear_Embedding = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.seq_len, self.d_model), nn.GELU(), )
        self.OPE = OPE(self.dropout, self.d_router)
        self.OCE = OCE(self.dropout, self.d_model, self.d_router, self.enc_in)
        self.output_proj = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.d_model, self.pred_len),)


    def forward(self, x):
        if self.use_revin:
            x = self.revin_layer(x, mode='norm')

        x_input = x.permute(0, 2, 1)

        x_exp = self.Linear_Embedding(x_input)

        G_h = self.OPE(x_input)
        y_out = self.OCE(x_exp, G_h)
        y_out = self.output_proj(y_out)

        output = y_out.permute(0, 2, 1)
        if self.use_revin:
            output = self.revin_layer(output, mode='denorm')
        return output

