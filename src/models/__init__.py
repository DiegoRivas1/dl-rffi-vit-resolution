from .vit import ViT
from .msvit import MSViT

def build_model(model_cfg, num_classes, train_hw):
    if model_cfg.type == "vit":
        return ViT(num_classes=num_classes, train_hw=train_hw, in_chans=model_cfg.in_chans,
                   patch_size=model_cfg.patch_size, embed_dim=model_cfg.embed_dim,
                   depth=model_cfg.depth, num_heads=model_cfg.num_heads,
                   mlp_ratio=model_cfg.mlp_ratio, dropout=model_cfg.dropout)
    if model_cfg.type == "msvit":
        g = model_cfg.get
        return MSViT(num_classes=num_classes, in_chans=g("in_chans", 1), patch_size=g("patch_size", 4),
                     dims=tuple(g("dims", (64, 128, 256))), depths=tuple(g("depths", (2, 4, 2))),
                     head_dim=g("head_dim", 32), mlp_ratio=g("mlp_ratio", 4.0), dropout=g("dropout", 0.1),
                     rep_kernel=g("rep_kernel", 3), use_scconv=g("use_scconv", True),
                     use_repcpe=g("use_repcpe", True))

    raise ValueError(f"modelo no soportado todavía: {model_cfg.type}")  # f3: msvit


