import torch

# Simulate the input tensor shape from the dataloader
batch_images = torch.randn(2, 1, 1, 256, 256)  # [B, C, Z, H, W]
print(f"Original shape: {batch_images.shape}")
print(
    f"Dimensions: B={batch_images.shape[0]}, C={batch_images.shape[1]}, Z={batch_images.shape[2]}, H={batch_images.shape[3]}, W={batch_images.shape[4]}"
)

# Test different squeeze operations
print(f"squeeze(dim=2): {torch.squeeze(batch_images, dim=2).shape}")  # Remove Z
print(
    f"squeeze(dim=-3): {torch.squeeze(batch_images, dim=-3).shape}"
)  # Remove Z (same as dim=2)
print(
    f"squeeze(dim=-4): {torch.squeeze(batch_images, dim=-4).shape}"
)  # Would remove C (wrong!)
