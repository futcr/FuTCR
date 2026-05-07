
# mask2former/modeling/future_aware/future_region_contrast.py

import torch
from torch import nn
import torch.nn.functional as F

from .helper_functions_update import (
    select_future_like_masks,
    compute_region_prototypes,
    sample_pixels_from_regions,
    info_nce_loss,
    ignore_repulsion_loss,  
)

from . import get_future_aware_logger

logger = get_future_aware_logger()  # info, debug, error, warning

#futcr_author1 2026-03-9
class FutureRegionContrastModule(nn.Module):
    """
    Computes future-aware losses from two sources:

      1) Region-based pixel-to-region contrastive loss over predicted masks
         that correspond to *future-like* regions.

      2) Ignore-repulsion loss that pushes features of ignore/unlabeled pixels
         away from known-class prototypes.

    Inputs (from Criterion.loss_future_aware):
      - mask_features: Tensor [B, C, H, W]
          Pixel embeddings from the pixel decoder (same H,W as pred_masks).
      - pred_masks   : Tensor [B, Q, H, W]
          Query-wise mask logits from the transformer decoder (last layer).
      - gt_semantic  : list[Tensor H×W] length B
          Each gt_semantic[b] is int labels (built from instance targets).
      - known_class_ids: iterable[int]
          Class indices already known at current task (old + current).
      - ignore_id    : int
          Label used for unlabeled pixels in gt_semantic.

    Output:
      - dict with key "loss_future_contrast": scalar Tensor.
    """

    def __init__(self, cfg):
        super().__init__()

        # Global weight for this module
        # final loss = loss_weight * (L_region + λ_ignore * L_ignore)
        self.loss_weight = cfg.CONT.FUTURE_AWARE.LOSS_WEIGHT
         # ----------------------------------------------------------------------
        # Region-based contrast branch
        self.region_contrast_enable = cfg.CONT.FUTURE_AWARE.REGION_CONTRAST_ENABLE
        self.num_pixels_per_region = int(cfg.CONT.FUTURE_AWARE.NUM_SAMPLED_PIXELS_PER_REGION)
        self.temperature = cfg.CONT.FUTURE_AWARE.TEMPERATURE
        self.mask_threshold = cfg.CONT.FUTURE_AWARE.MASK_THRESHOLD

        # ----------------------------------------------------------------------
        # Ignore-repulsion branch
        self.ignore_repulsion_enable = cfg.CONT.FUTURE_AWARE.IGNORE_REPULSION_ENABLE
        self.ignore_repulsion_weight = cfg.CONT.FUTURE_AWARE.IGNORE_REPULSION_WEIGHT
        self.ignore_repulsion_margin = cfg.CONT.FUTURE_AWARE.IGNORE_REPULSION_MARGIN
        self.max_ignore_pixels = cfg.CONT.FUTURE_AWARE.MAX_IGNORE_PIXELS
        
        # ----------------------------------------------------------------------
        # Auxiliary classifier branch
        self.aux_enable = cfg.CONT.FUTURE_AWARE.AUX_CLS_ENABLE
        self.aux_num_clusters = int(cfg.CONT.FUTURE_AWARE.AUX_CLS_NUM_CLUSTERS)
        self.aux_hidden_dim = cfg.CONT.FUTURE_AWARE.AUX_CLS_HIDDEN_DIM
        self.aux_loss_weight = cfg.CONT.FUTURE_AWARE.AUX_CLS_LOSS_WEIGHT
        self.aux_update_freq = cfg.CONT.FUTURE_AWARE.AUX_CLS_UPDATE_FREQ
        self.aux_buffer_size = cfg.CONT.FUTURE_AWARE.AUX_CLS_BUFFER_SIZE
        self.aux_classifier = None

        # Prototype buffer for clustering: [N_buf, C]
        self.register_buffer("aux_proto_buffer", None)
        self.aux_step_counter = 0

        # Cluster centers: [K, C]; lazily initialized
        self.register_buffer("aux_cluster_centers", None)
        # ----------------------------------------------------------------------



    def forward(self, mask_features, pred_masks, gt_semantic, known_class_ids, ignore_id):
        """
        mask_features : Tensor [B, C, H, W]
        pred_masks    : Tensor [B, Q, H, W] (logits)
        gt_semantic   : list[Tensor H×W] length B
        known_class_ids: iterable[int]
        ignore_id     : int

        Returns:
          {"loss_future_contrast": scalar}  (zero scalar if no signal)
        """
        # Sanity checks on shapes
        assert mask_features.dim() == 4, f"mask_features must be [B,C,H,W], got {mask_features.shape}"
        assert pred_masks.dim() == 4, f"pred_masks must be [B,Q,H,W], got {pred_masks.shape}"
        B, C, H, W = mask_features.shape
        B2, Q, H2, W2 = pred_masks.shape
        assert B == B2 and H == H2 and W == W2, \
            f"mask_features and pred_masks must share B,H,W; got {mask_features.shape} and {pred_masks.shape}"
        assert len(gt_semantic) == B, f"gt_semantic length {len(gt_semantic)} != batch size {B}"

        # ----------------------------------------------------------------------
        # 1) Region-based pixel-to-region contrastive loss (InfoNCE)
        # ----------------------------------------------------------------------
        loss_region = mask_features.sum() * 0.0  # zero scalar on correct device
        loss_aux = mask_features.sum() * 0.0
         
         
        if self.region_contrast_enable:
            # 1.1 Select indices of predicted masks that are "future-like"
            # future_region_indices: list length B;
            #   future_region_indices[b]: list[int] of query indices q (0 <= q < Q)
            #futcr_author1*: list of lists (query indices of list of Q), for each image in B, that suggest a future class
            future_region_indices = select_future_like_masks(
                pred_masks=pred_masks,        # [B, Q, H, W], logits
                gt_semantic=gt_semantic,      # list[H,W] int labels
                known_class_ids=known_class_ids,
                mask_threshold=self.mask_threshold,
                ignore_id=ignore_id,
            )

            # If no candidate future regions in the entire batch, region loss stays zero.
            if not all(len(idxs) == 0 for idxs in future_region_indices):
                # 1.2 Compute region prototypes (mean feature per region)
                #    Each region is defined by (b,q) with q in future_region_indices[b],
                #    and support pixels where sigmoid(pred_masks[b,q]) > mask_threshold.
                
                # region_prototypes: Tensor [N_regions, C] futcr_author1* Regions in the entire batch
                #   Each row is mean feature over pixels in one future-like region.
                #
                # region_assignments: list of length N_regions
                #   region_assignments[r] = (b_idx, pixel_indices)
                #   where:
                #     b_idx: int in [0, B)
                #     pixel_indices: 1D LongTensor of positions in [0, H*W)
                #
                #   We don't need the query index anymore once prototypes are computed.
                region_prototypes, region_assignments = compute_region_prototypes(
                    mask_features=mask_features,      # [B, C, H, W]
                    pred_masks=pred_masks,            # [B, Q, H, W]
                    future_region_indices=future_region_indices,
                    mask_threshold=self.mask_threshold,
                )
                num_regions = region_prototypes.shape[0]

                if num_regions > 0:
                    # 1.3 Sample pixels from regions: anchors for InfoNCE
                    #Sample pixels from regions to build InfoNCE pairs.
                    #    We sample up to num_pixels_per_region per region.
                    # sampled_feats    : Tensor [N_samples, C]
                    # sampled_region_ids: LongTensor [N_samples], each in [0, N_regions)
                    sampled_feats, sampled_region_ids = sample_pixels_from_regions(
                        mask_features=mask_features,          # [B, C, H, W]
                        region_assignments=region_assignments,
                        num_pixels_per_region=self.num_pixels_per_region,
                    )
                    # sampled_feats    : [N_samples, C]
                    # sampled_region_ids: [N_samples] each in [0, N_regions)

                    if sampled_feats.shape[0] > 0:
                        # 1.4 Build positives and negatives for InfoNCE
                        pos_prototypes = region_prototypes[sampled_region_ids]  # [N_samples, C]
                        all_prototypes = region_prototypes                     # [N_regions, C]

                        # 1.5 Compute InfoNCE loss
                        #   For each pixel feature f_i and positive prototype p_i,
                        #   L_i = - log ( exp(sim(f_i, p_i)/τ) / sum_j exp(sim(f_i, r_j)/τ) )
                        # where sim is cosine similarity, τ = self.temperature.
                        loss_region = info_nce_loss(
                            pixel_feats=sampled_feats,      # [N_samples, C]
                            pos_prototypes=pos_prototypes,  # [N_samples, C]
                            all_prototypes=all_prototypes,  # [N_regions, C]
                            temperature=self.temperature,
                        )
                        
                        if self.aux_enable:
                            loss_aux = self._aux_classifier_loss(region_prototypes)

                        logger.debug(
                            f"[future_region_contrast] loss_region={float(loss_region.item()):.6f} "
                            f"N_regions={num_regions}, N_samples={sampled_feats.shape[0]}"
                            f"[future_region_contrast] loss_aux={float(loss_aux.item()):.6f} "
                        )

        # ----------------------------------------------------------------------
        # 2) Ignore-repulsion loss (ignore pixels vs known-class prototypes)
        # ----------------------------------------------------------------------
        loss_ignore = mask_features.sum() * 0.0  # zero scalar

        if self.ignore_repulsion_enable:
            loss_ignore = ignore_repulsion_loss(
                mask_features=mask_features,      # [B, C, H, W]
                gt_semantic=gt_semantic,          # list[H,W]
                known_class_ids=known_class_ids,
                ignore_id=ignore_id,
                max_ignore_pixels=self.max_ignore_pixels,
                margin=self.ignore_repulsion_margin,
            )
            logger.debug(
                f"[future_region_contrast] loss_ignore={float(loss_ignore.item()):.6f}"
            )

        # ----------------------------------------------------------------------
        # 3) Combine with weights and return
        # ----------------------------------------------------------------------
        # total_internal = L_region + λ_ignore * L_ignore + λ_aux * L_aux
        # total_internal = loss_region + self.ignore_repulsion_weight * loss_ignore
        total_internal = (
            loss_region
            + self.ignore_repulsion_weight * loss_ignore
            + self.aux_loss_weight * loss_aux
        )

        # Apply global scaling
        loss_con_weighted = self.loss_weight * total_internal

        logger.debug(
            f"[future_region_contrast] total_loss={float(loss_con_weighted.item()):.6f} "
            f"(region={float(loss_region.item()):.6f}, "
            f"ignore={float(loss_ignore.item()):.6f}, "
            f"aux={float(loss_aux.item()):.6f}, "
            f"λ_main={self.loss_weight}, "
            f"λ_ignore={self.ignore_repulsion_weight}, "
            f"λ_aux={self.aux_loss_weight})"
        )

        return {"loss_future_contrast": loss_con_weighted}
    
    
    
    def _init_aux_classifier_and_centers(self, feat_dim, device):
        """
        Lazily initialize auxiliary classifier MLP and cluster centers.
        feat_dim: int (C), feature dimension of prototypes.
        """
        if self.aux_classifier is None:
            self.aux_classifier = nn.Sequential(
                nn.Linear(feat_dim, self.aux_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.aux_hidden_dim, self.aux_num_clusters),
            ).to(device)

        if self.aux_cluster_centers is None:
            # Randomly initialize cluster centers on the unit sphere
            centers = torch.randn(self.aux_num_clusters, feat_dim, device=device)
            centers = F.normalize(centers, dim=1)
            self.aux_cluster_centers = centers
            
            
    def _aux_classifier_loss(self, region_prototypes):
        """
        Compute auxiliary classification & balance loss on region_prototypes.

        region_prototypes: Tensor [N_regions, C]
        Returns scalar loss (Tensor).
        """
        if region_prototypes.numel() == 0:
            return region_prototypes.sum() * 0.0

        N, C = region_prototypes.shape
        device = region_prototypes.device

        # Initialize classifier and centers if needed
        self._init_aux_classifier_and_centers(C, device)

        # 1) Update buffer
        self._update_aux_buffer(region_prototypes)

        # 2) Periodically update cluster centers via K-means
        self.aux_step_counter += 1
        if self.aux_step_counter % self.aux_update_freq == 0:
            self._run_kmeans_on_buffer()

        if self.aux_cluster_centers is None:
            # No centers yet; skip until we have them
            return region_prototypes.sum() * 0.0

        # 3) Assign each prototype to nearest cluster center (pseudo-labels)
        # Normalize features
        feats = F.normalize(region_prototypes, dim=1)            # [N, C]
        centers = F.normalize(self.aux_cluster_centers, dim=1)   # [K, C]

        # sim: [N, K]
        sim = feats @ centers.t()
        _, labels = sim.max(dim=1)  # [N], int in [0,K)

        # 4) Classifier prediction logits: [N, K]
        logits = self.aux_classifier(region_prototypes.detach())  # stop gradient into prototypes

        # 4a) Cross-entropy (confident assignments)
        ce_loss = F.cross_entropy(logits, labels)

        # 4b) Balance term: encourage uniform usage of clusters
        probs = F.softmax(logits, dim=1)         # [N, K]
        p_mean = probs.mean(dim=0)               # [K]
        uniform = torch.full_like(p_mean, 1.0 / self.aux_num_clusters)
        balance_loss = F.kl_div(
            p_mean.log(), uniform, reduction="batchmean"
        )  # KL(p_mean || uniform)

        loss_aux = ce_loss + balance_loss

        logger.debug(
            f"[aux_classifier] N_regions={N}, K={self.aux_num_clusters}, "
            f"ce_loss={float(ce_loss.item()):.6f}, bal_loss={float(balance_loss.item()):.6f}"
        )

        return loss_aux
    
    
    def _update_aux_buffer(self, region_prototypes):
        """
        Append region_prototypes [N_r, C] to aux_proto_buffer, keeping at most aux_buffer_size.
        """
        if region_prototypes.numel() == 0:
            return

        if self.aux_proto_buffer is None:
            self.aux_proto_buffer = region_prototypes.detach()
        else:
            buf = torch.cat([self.aux_proto_buffer, region_prototypes.detach()], dim=0)
            if buf.shape[0] > self.aux_buffer_size:
                buf = buf[-self.aux_buffer_size:]
            self.aux_proto_buffer = buf


    def _run_kmeans_on_buffer(self):
        """
        Run a simple K-means on aux_proto_buffer to update aux_cluster_centers.
        """
        if self.aux_proto_buffer is None or self.aux_proto_buffer.shape[0] < self.aux_num_clusters:
            return  # not enough data

        X = self.aux_proto_buffer  # [N_buf, C]
        X = F.normalize(X, dim=1)
        N, C = X.shape
        K = self.aux_num_clusters

        # Initialize centers from current buffer if centers are None
        if self.aux_cluster_centers is None or self.aux_cluster_centers.shape[1] != C:
            # Choose K random rows
            perm = torch.randperm(N, device=X.device)
            centers = X[perm[:K]]
        else:
            centers = self.aux_cluster_centers.clone()

        # Simple fixed-iter K-means
        num_iters = 5
        for _ in range(num_iters):
            # Assign each point to nearest center
            # sim: [N, K] = X @ centers^T
            sim = X @ centers.t()
            _, labels = sim.max(dim=1)  # [N], int in [0,K)

            # Recompute centers
            new_centers = torch.zeros_like(centers)
            for k in range(K):
                mask_k = (labels == k)
                if mask_k.any():
                    new_centers[k] = X[mask_k].mean(dim=0)
                else:
                    # If a cluster is empty, re-sample a random point
                    idx = torch.randint(0, N, (1,), device=X.device)
                    new_centers[k] = X[idx]

            centers = F.normalize(new_centers, dim=1)

        self.aux_cluster_centers = centers
        logger.debug(
            f"[aux_kmeans] Updated cluster centers from buffer: N_buf={N}, K={K}"
        )




































# # mask2former/modeling/future_aware/future_region_contrast.py

# import torch
# from torch import nn
# import torch.nn.functional as F

# from .helper_functions import (
#     select_future_like_masks,
#     compute_region_prototypes,
#     sample_pixels_from_regions,
#     info_nce_loss,
# )

# from . import get_future_aware_logger

# logger = get_future_aware_logger() #info, debug, error, warning

# #futcr_author1 2026-03-9
# class FutureRegionContrastModule(nn.Module):
#     """
#     Computes a pixel-to-region contrastive loss over predicted masks that
#     correspond to *future-like* classes.

#     Inputs (from Criterion.loss_future_aware):
#       - mask_features: Tensor [B, C, H, W]
#           Pixel embeddings from the pixel decoder (same H,W as pred_masks).
#       - pred_masks: Tensor [B, Q, H, W]
#           Query-wise mask logits from the transformer decoder (last layer).
#       - gt_semantic: list[Tensor] of length B
#           Each gt_semantic[b]: [H, W] int labels (constructed from targets).
#       - known_class_ids: set/list of ints
#           Class indices that are already known at current task (old + current).

#     Output:
#       - dict with key "loss_future_contrast": scalar tensor.
#     """

#     def __init__(self, cfg):
#         super().__init__()

#         # Weight used when adding this loss into the total loss
#         # loss_future_contrast = loss_weight * L_con
#         self.loss_weight = cfg.CONT.FUTURE_AWARE.LOSS_WEIGHT

#         # Max number of pixels to sample per region (for efficiency)
#         self.num_pixels_per_region = cfg.CONT.FUTURE_AWARE.NUM_SAMPLED_PIXELS_PER_REGION

#         # Temperature τ in InfoNCE
#         self.temperature = cfg.CONT.FUTURE_AWARE.TEMPERATURE

#         # Threshold on mask probabilities to define region support
#         # We will use (pred_masks.sigmoid() > mask_threshold) as region membership.
#         self.mask_threshold = cfg.CONT.FUTURE_AWARE.MASK_THRESHOLD

#     def forward(self, mask_features, pred_masks, gt_semantic, known_class_ids, ignore_id ):
#         """
#         mask_features: Tensor [B, C, H, W]
#         pred_masks   : Tensor [B, Q, H, W] (logits)
#         gt_semantic  : list[Tensor H×W] length B
#         known_class_ids: set/list[int]

#         Returns:
#           {"loss_future_contrast": scalar}  (or zero scalar if no regions)
#         """
#         # Sanity checks on shapes
#         assert mask_features.dim() == 4, f"mask_features must be [B,C,H,W], got {mask_features.shape}"
#         assert pred_masks.dim() == 4, f"pred_masks must be [B,Q,H,W], got {pred_masks.shape}"
#         B, C, H, W = mask_features.shape
#         B2, Q, H2, W2 = pred_masks.shape
#         assert B == B2 and H == H2 and W == W2, \
#             f"mask_features and pred_masks must share B,H,W; got {mask_features.shape} and {pred_masks.shape}"
#         assert len(gt_semantic) == B, f"gt_semantic length {len(gt_semantic)} != batch size {B}"
        
#         # 1) Select indices of predicted masks that are "future-like"
#         #    This uses gt_semantic and known_class_ids internally.
#         #futcr_author1*: list of lists (query indices of list of Q), for each image in B, that suggest a future class
#         # future_region_indices: list of length B;
#         #   future_region_indices[b]: list of mask indices q (0 <= q < Q) that are future-like.
#         future_region_indices = select_future_like_masks(
#             pred_masks=pred_masks,         # [B, Q, H, W], logits
#             gt_semantic=gt_semantic,       # list[H,W] (int labels)
#             known_class_ids=known_class_ids,
#             mask_threshold=self.mask_threshold,
#             ignore_id=ignore_id
#         )
        
#         # If no candidate future regions in the entire batch, return zero loss.
#         if all(len(idxs) == 0 for idxs in future_region_indices):
#             # Zero-loss tensor with correct device/dtype, contributes nothing to gradients.
#             # logger.warning(f'Loss 0 due to query indices: future_region_indices lenghts')
#             return {"loss_future_contrast": mask_features.sum() * 0.0}


#         # 2) Compute region prototypes (mean pixel feature per region).
#         #    Each region is defined by (b,q) with q in future_region_indices[b],
#         #    and support pixels where sigmoid(pred_masks[b,q]) > mask_threshold.
        
#         # region_prototypes: Tensor [N_regions, C] futcr_author1* Regions in the entire batch
#         #   Each row is mean feature over pixels in one future-like region.
#         #
#         # region_assignments: list of length N_regions
#         #   region_assignments[r] = (b_idx, pixel_indices)
#         #   where:
#         #     b_idx: int in [0, B)
#         #     pixel_indices: 1D LongTensor of positions in [0, H*W)
#         #
#         #   We don't need the query index anymore once prototypes are computed.
#         region_prototypes, region_assignments = compute_region_prototypes(
#             mask_features=mask_features,               # [B, C, H, W]
#             pred_masks=pred_masks,                     # [B, Q, H, W]
#             future_region_indices=future_region_indices,
#             mask_threshold=self.mask_threshold,
#         )
        
        
        
        
#         num_regions = region_prototypes.shape[0]
#         if num_regions == 0:
#             # This can happen if all future-like masks were empty after thresholding.
#             # logger.warning(f'Loss 0 due to num regions: {num_regions}')
#             return {"loss_future_contrast": mask_features.sum() * 0.0}




#         # 3) Sample pixels from regions to build InfoNCE pairs.
#         #    We sample up to num_pixels_per_region per region.
#         # sampled_feats    : Tensor [N_samples, C]
#         # sampled_region_ids: LongTensor [N_samples], each in [0, N_regions)
#         sampled_feats, sampled_region_ids = sample_pixels_from_regions(
#             mask_features=mask_features,         # [B, C, H, W]
#             region_assignments=region_assignments,
#             num_pixels_per_region=self.num_pixels_per_region,
#         )
        

#         if sampled_feats.shape[0] == 0:
#             # No sampled pixels (should be rare, but safe to handle)
#             # logger.warning(f'Loss 0 due to sampled features: {sampled_feats.shape[0]}')
            
#             return {"loss_future_contrast": mask_features.sum() * 0.0}

#         # 4) Build positives and negatives for InfoNCE.
#         #    For each sampled pixel i, the positive prototype is region_prototypes[region_id].
#         pos_prototypes = region_prototypes[sampled_region_ids]  # [N_samples, C]

#         # Negatives: all region prototypes within the batch.
#         # For now, we only use per-batch prototypes (no memory queue) for simplicity.
#         # denominator of infoNCE softmax
#         all_prototypes = region_prototypes                     # [N_regions, C]

#         # Compute InfoNCE loss:
#         #   For each pixel feature f_i and positive prototype p_i,
#         #   L_i = - log ( exp(sim(f_i, p_i)/τ) / sum_j exp(sim(f_i, r_j)/τ) )
#         # where sim is cosine similarity, τ = self.temperature.
#         loss_con = info_nce_loss(
#             pixel_feats=sampled_feats,      # [N_samples, C]
#             pos_prototypes=pos_prototypes,  # [N_samples, C]
#             all_prototypes=all_prototypes,  # [N_regions, C]
#             temperature=self.temperature,
#         )
        
#         """
#         what about the overclustering and merging, visual similarity vs feature affinity parts to make the loss as on the ptf paper
#         """

#         # 5) Scale by configured weight and return.
#         loss_con_weighted = self.loss_weight * loss_con  # scalar
#         logger.debug(
#             f"[future_region_contrast] resultant loss={loss_con_weighted} loss={float(loss_con.item()):.6f}"
#         )

#         return {"loss_future_contrast": loss_con_weighted}
