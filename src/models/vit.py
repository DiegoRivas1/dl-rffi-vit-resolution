"""ViT convencional (Dosovitskiy et al.) con positional embedding absoluto aprendido.

Si la cuadrícula de parches difiere de la de entrenamiento, el pos-embedding se
INTERPOLA (bicúbico) en el forward. Es un ajuste geométrico, no actualiza ningún peso:
la evaluación en R_test sigue siendo sin fine-tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads, drop):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = drop
        self.out_drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        return self.out_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, drop):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hid, dim), nn.Dropout(drop))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class ViT(nn.Module):
    def __init__(self, num_classes, train_hw, in_chans=1, patch_size=8, embed_dim=192,
                 depth=6, num_heads=3, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        H, W = train_hw
        p = patch_size
        if H % p or W % p:
            raise ValueError(f"La entrada de entrenamiento {train_hw} debe ser múltiplo del parche {p}.")
        self.p = p
        self.grid0 = (H // p, W // p)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, p, p)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.grid0[0] * self.grid0[1], embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _pos(self, gh, gw):
        if (gh, gw) == self.grid0:
            return self.pos_embed
        cls, grid = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        g0h, g0w = self.grid0
        grid = grid.reshape(1, g0h, g0w, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid.float(), size=(gh, gw), mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, gh * gw, -1).to(cls.dtype)
        return torch.cat([cls, grid], dim=1)

    def forward(self, x):
        B, _, H, W = x.shape
        if H % self.p or W % self.p:
            raise ValueError(f"Entrada {H}x{W} no es múltiplo del parche {self.p}.")
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self._pos(H // self.p, W // self.p).to(x.dtype))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x)[:, 0])


if __name__ == "__main__":
    m = ViT(6, (64, 64)).eval()
    n_par = sum(p.numel() for p in m.parameters())
    before = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        for hw in [(32, 32), (64, 64), (96, 64), (128, 128)]:
            out = m(torch.randn(2, 1, *hw))
            assert out.shape == (2, 6)
            print(f"entrada {hw} -> logits {tuple(out.shape)} (grid {hw[0]//8}x{hw[1]//8})")
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items()), "los pesos cambiaron!"
    print(f"parámetros: {n_par/1e6:.2f} M | pesos intactos tras evaluar en otras resoluciones: OK")
