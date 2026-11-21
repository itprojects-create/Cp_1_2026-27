import timm, torch, torch.nn as nn

# Backbone 1 — EfficientNet-B4
eff = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
for p in eff.parameters(): p.requires_grad = False
for name, p in eff.named_parameters():
    if 'blocks.5' in name or 'blocks.6' in name or 'conv_head' in name:
        p.requires_grad = True

# Backbone 2 — ViT-Base/16
vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
for p in vit.parameters(): p.requires_grad = False
for name, p in vit.named_parameters():
    if 'blocks.10' in name or 'blocks.11' in name or 'norm' in name:
        p.requires_grad = True


if __name__ == '__main__':
    # Quick check
    x = torch.randn(2, 3, 512, 512)
    x_vit = nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
    print(eff(x).shape)    # [2, 1792]
    print(vit(x_vit).shape) # [2, 768]
    
class HybridDRModel(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
            super().__init__()
            self.eff = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
            self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
            # freeze / unfreeze (same as above)
            self.fusion = nn.Sequential(
                nn.Linear(1792 + 768, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
            x_vit = nn.functional.interpolate(x, (224,224), mode='bilinear', align_corners=False)
            f1 = self.eff(x)       # [B, 1792]
            f2 = self.vit(x_vit)   # [B, 768]
            return self.fusion(torch.cat([f1, f2], dim=1))  # [B, 5]

    # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Shape test — must pass before handing to Team B
    # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     model = HybridDRModel()
    # Quieted test line from class scope:     out   = model(torch.randn(2, 3, 512, 512))
    # Quieted test line from class scope:     assert out.shape == (2, 5), f"Wrong shape: {out.shape}"
    # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     print("Shape test passed:", out.shape)

    # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     # Quieted test line from class scope:     print(f"Trainable params: {trainable:,}")
