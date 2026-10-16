from .vit import ViT


def build_model(model_cfg, num_classes, train_hw):
    if model_cfg.type == "vit":
        return ViT(num_classes=num_classes, train_hw=train_hw, in_chans=model_cfg.in_chans,
                   patch_size=model_cfg.patch_size, embed_dim=model_cfg.embed_dim,
                   depth=model_cfg.depth, num_heads=model_cfg.num_heads,
                   mlp_ratio=model_cfg.mlp_ratio, dropout=model_cfg.dropout)
    raise ValueError(f"modelo no soportado todavía: {model_cfg.type}")  # f3: msvit
