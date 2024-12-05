from torchvision import models

weights = models.ViT_B_16_Weights.IMAGENET1K_V1
model = models.vit_b_16(weights=weights)

print(model.heads[0].in_features)
