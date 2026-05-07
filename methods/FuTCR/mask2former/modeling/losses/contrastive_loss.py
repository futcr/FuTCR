import torch
from torch import nn
from torch.nn import functional as F


"""
HybridContrastiveLoss: A supervised contrastive loss module designed for continual 
panoptic segmentation to replace rigid KL-divergence based consistency constraints.

Key design:
- Uses contrastive learning to maintain old-class separability while allowing adaptive feature drift
- Incorporates optional overlap-aware scaling to reduce loss weight when class overlap is high
- Robust to NaN/Inf values that can arise from extreme similarity scores
- Replaces method1's CSL (Consistency Selection Loss) which uses rigid KL divergence

References:
- Inspired by: Contrastive Pseudo Learning (Sun et al., ICCV 2023)
- Continual Universal Segmentation (CUE) decoupling principle: high-level features can adapt
"""


class HybridContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, use_auto_overlap_scaling=True, overlap_threshold=0.5):
        super().__init__()
        self.temperature = temperature
        self.use_auto_overlap_scaling = use_auto_overlap_scaling
        self.overlap_threshold = overlap_threshold
    
    def forward(self, features_curr, features_prev, prototypes_prev):
        """Features can be raw or normalized - we handle both!"""
        # Validate shapes
        assert features_curr.dim() == 2 and features_curr.shape[1] == prototypes_prev.shape[1]
        assert features_prev.dim() == 2
        assert prototypes_prev.dim() == 2
        
        # Normalize (safe if already normalized)
        feat_curr = F.normalize(features_curr, p=2, dim=1)
        feat_prev = F.normalize(features_prev, p=2, dim=1)
        proto = F.normalize(prototypes_prev, p=2, dim=1)
        
        # Similarities
        sim_curr = torch.mm(feat_curr, proto.T) / self.temperature  # [N, C_old]
        sim_prev = torch.mm(feat_prev, proto.T) / self.temperature  # [N, C_old]
        
        # Labels
        labels = torch.argmax(sim_curr.detach(), dim=1)
        
        # Loss
        loss = self.supervised_contrastive_loss(feat_curr, labels)
        
        # Optional overlap scaling
        if self.use_auto_overlap_scaling:
            overlap_ratio = self.estimate_overlap(sim_prev)
            print(f"overlap ratio {overlap_ratio}")
            overlap_ratio = min(max(overlap_ratio, 0.0), 1.0)
            loss = (1.0 - overlap_ratio) * loss
        
        return loss
    
    def supervised_contrastive_loss(self, feats, labels):
        """Supervised contrastive loss with NaN/Inf protection."""
        N = feats.size(0)
        
        if N <= 1:
            return feats.new_tensor(0.0)
        
        # Similarities
        sim = torch.matmul(feats, feats.T) / self.temperature  # [N, N]
        
        # Mask self
        self_mask = torch.eye(N, dtype=torch.bool, device=feats.device)
        sim = sim.masked_fill(self_mask, float('-inf'))
        
        # Positive pairs
        labels_expand = labels.view(-1, 1)
        pos_mask = (labels_expand == labels_expand.T) & (~self_mask)
        
        # Log prob
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        
        # ===== HANDLE NaN/Inf =====
        log_prob = torch.where(torch.isfinite(log_prob), log_prob, log_prob.new_zeros(1))
        
        # Loss per sample
        pos_counts = pos_mask.sum(dim=1).float()
        valid = pos_counts > 0
        
        if not valid.any():
            return feats.new_tensor(0.0)
        
        loss_per_sample = torch.zeros(N, dtype=feats.dtype, device=feats.device)
        
        for i in range(N):
            if valid[i]:
                pos_log_probs = log_prob[i][pos_mask[i]]
                pos_log_probs = torch.where(torch.isfinite(pos_log_probs), pos_log_probs, pos_log_probs.new_zeros(1))
                
                if pos_log_probs.numel() > 0:
                    loss_per_sample[i] = -pos_log_probs.mean()
        
        valid_loss = loss_per_sample[valid]
        if valid_loss.numel() == 0:
            return feats.new_tensor(0.0)
        
        loss = valid_loss.mean()
        
        if not torch.isfinite(loss):
            return feats.new_tensor(0.0)
        
        return loss
    
    def estimate_overlap(self, similarities):
        """Estimate overlap ratio."""
        max_sim, _ = similarities.max(dim=1)
        similar = (max_sim > self.overlap_threshold).float()
        return min(max(similar.mean().item(), 0.0), 1.0)
