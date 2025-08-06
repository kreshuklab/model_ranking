import torch

# Single image case like in the test script
single_image = torch.randn(1, 1, 1, 256, 256)  # [B=1, C=1, Z=1, H=256, W=256]
print(f"Single image shape: {single_image.shape}")

# Test different squeeze operations
print(f"squeeze(dim=2): {torch.squeeze(single_image, dim=2).shape}")  # Remove Z
print(
    f"squeeze(dim=-3): {torch.squeeze(single_image, dim=-3).shape}"
)  # Should remove Z
print(f"squeeze(): {torch.squeeze(single_image).shape}")  # Remove all size-1 dims

# What if we squeeze the wrong dimension?
print(
    f"squeeze(dim=0): {torch.squeeze(single_image, dim=0).shape}"
)  # Remove batch dim (wrong!)
print(
    f"squeeze(dim=1): {torch.squeeze(single_image, dim=1).shape}"
)  # Remove channel dim (wrong!)
