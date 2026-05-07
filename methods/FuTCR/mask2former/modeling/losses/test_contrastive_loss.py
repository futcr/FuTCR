
import sys,os
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


import torch
from mask2former.modeling.losses import HybridContrastiveLoss

loss_fn = HybridContrastiveLoss(temperature=0.1, use_auto_overlap_scaling=True)

# Test 1: Random data → loss = 0 (expected)
features_curr = torch.randn(200, 256)
features_prev = torch.randn(200, 256)
prototypes_prev = torch.randn(80, 256)
loss1 = loss_fn(features_curr, features_prev, prototypes_prev)
print(f"Random data loss: {loss1.item():.4f}")  # Should be 0.0000 ✓

# Test 2: Structured data → loss > 0 (expected)
# Create 3 prototypes, 20 features each (clustered)
prototypes = torch.randn(3, 256)
features = torch.cat([
    prototypes[0] + 0.1*torch.randn(20, 256),  # 20 features near prototype 0
    prototypes[1] + 0.1*torch.randn(20, 256),  # 20 features near prototype 1
    prototypes[2] + 0.1*torch.randn(20, 256),  # 20 features near prototype 2
])
loss2 = loss_fn(features, features, prototypes)
print(f"Structured data loss: {loss2.item():.4f}")  # Should be > 0 ✓