# mask2former/modeling/future_aware/helper_functions.py

import torch
import torch.nn.functional as F
import os
import torch
import matplotlib.pyplot as plt

from . import get_future_aware_logger

logger = get_future_aware_logger()


def select_future_like_masks(
    pred_masks,
    gt_semantic,
    known_class_ids,
    mask_threshold,
    ignore_id=None,
    min_region_pixels=20, #50 TODO: investigate this
    min_mean_conf=0.2,#0.5
):
    """
    Identify predicted masks that likely correspond to *future-like* regions.

    Inputs:
      pred_masks     : Tensor [B, Q, H, W] (logits)
      gt_semantic    : list[Tensor H×W] length B, int labels per pixel
                       (built from instance targets: labels + masks)
      known_class_ids: set/list[int], classes already known at this task
      mask_threshold : float, threshold on sigmoid(pred_masks) to define region support
      ignore_id      : int or None, label used for unlabeled pixels in gt_semantic
      min_region_pixels: int, minimum number of pixels in a region to be considered
      min_mean_conf  : float, minimum mean mask probability inside the region

    Output:
      future_region_indices: list of length B
        future_region_indices[b]: list[int] of query indices q (0 ≤ q < Q)
        for which the mask is future-like (mostly unlabeled / ignore).
    """
    B, Q, H, W = pred_masks.shape
    # Convert logits to probabilities once: [B, Q, H, W]
    prob_masks = pred_masks.sigmoid()

    known_set = set(known_class_ids)
    future_region_indices = []

    for b in range(B):
        gt_b = gt_semantic[b]  # [H_b, W_b], int

        # Ensure GT spatial size matches pred_masks; resize if needed
        if gt_b.shape[-2:] != (H, W):
            # gt_b: [H_gt, W_gt] -> [1,1,H_gt,W_gt] -> interpolate -> [1,1,H,W] -> squeeze
            gt_b = F.interpolate(
                gt_b[None, None].float(), size=(H, W), mode="nearest"
            ).long()[0, 0]

        img_indices = []

        for q in range(Q):
            # 1) Binary region support: [H, W] bool
            mask_q = prob_masks[b, q] > mask_threshold
            num_pixels = int(mask_q.sum().item())
            
            
            # logger.info(
            #     dict(
            #         num_pixels=num_pixels,
            #         min_region_pixels=min_region_pixels,
            #         mean_conf=float(prob_masks[b, q][mask_q].mean().item())
            #     )
            # )
            
            if num_pixels == 0:
                # logger.info(f'Num pixels is 0: {num_pixels}, some prob: {prob_masks[b, q][:5]}')
                # No confident pixels; skip this mask
                continue

            # Optional: minimum region size filter
            if num_pixels < min_region_pixels:
                # logger.info(f'Num pixels is low: {num_pixels}')
                continue

            # 2) Confidence filter: mean prob inside region
            mean_conf = float(prob_masks[b, q][mask_q].mean().item())
            if mean_conf < min_mean_conf:
                # logger.info(f'Mean conf is too low: {mean_conf}')
                continue

            # 3) Gather GT labels inside region: [num_pixels]
            gt_vals = gt_b[mask_q]

            # If ignore_id is used in gt_semantic, separate known vs ignore
            if ignore_id is not None:
                # Boolean masks for ignore and non-ignore pixels
                is_ignore = (gt_vals == ignore_id)
                num_ignore = int(is_ignore.sum().item())
                num_valid = num_pixels - num_ignore

                # -------------------------------------------------------------------
                # Case 1: region is dominated by labeled (known) pixels → treat as known
                # -------------------------------------------------------------------
                if num_valid > 0 and num_valid >= num_ignore :
                    # Among non-ignore pixels, find dominant label
                    valid_vals = gt_vals[~is_ignore]
                    bincount_valid = torch.bincount(valid_vals.flatten())
                    if bincount_valid.numel() == 0:
                        # logger.warning(f'Valid  bincount_valid.numel() is  0 {bincount_valid.numel()}')
                        continue
                    dominant_known = int(bincount_valid.argmax().item())
                    if dominant_known in known_set:
                        # Strongly overlaps some known class → not future-like
                        # logger.warning(f'Dominant known in known set {dominant_known}')
                        continue

                # Case B: region mostly ignore/unlabeled → treat as future-like
                # We require that ignore pixels dominate the region.
                if num_ignore > num_valid:
                    img_indices.append(q)
                # If neither case clearly holds, you can choose to skip or treat as future.
                # Here we skip ambiguous regions.
            else:
                # logger.error(f'No ignore id seen!')
                # No ignore_id: fall back to "label not in known_set" heuristic
                bincount = torch.bincount(gt_vals.flatten())
                if bincount.numel() == 0:
                    continue
                dominant_label = int(bincount.argmax().item())
                if dominant_label not in known_set:
                    img_indices.append(q)

        future_region_indices.append(img_indices)

    total_future = sum(len(x) for x in future_region_indices)
    # logger.debug(
    #     f"[select_future_like_masks] B={B}, Q={Q}, total_future_regions={total_future} "
    # )

    return future_region_indices



def select_future_like_masks_old(pred_masks, gt_semantic, known_class_ids, mask_threshold,ignore_id=None):
    """
    Identify predicted masks that likely correspond to *future* classes.

    Inputs:
      pred_masks    : Tensor [B, Q, H, W] (logits)
      gt_semantic   : list[Tensor H×W] length B, int labels per pixel
                      (constructed from targets: labels + instance masks)
      known_class_ids: set/list[int], classes already known at this task
      mask_threshold: float, threshold on sigmoid(pred_masks) to define region support

    Output:
      future_region_indices: list of length B
        future_region_indices[b]: list[int] of query indices q (0 ≤ q < Q)
        such that the dominant GT label in mask (b,q) is *not* in known_class_ids.
    """
    B, Q, H, W = pred_masks.shape
    # Convert to probabilities once
    prob_masks = pred_masks.sigmoid()  # [B, Q, H, W]

    # Robust set for membership test
    known_set = set(known_class_ids)

    future_region_indices = []

    for b in range(B):
        gt_b = gt_semantic[b]  # [H_gt, W_gt], int
        # logger.info(f"torch.unique(gt_b) : {torch.unique(gt_b)}")

        # Ensure GT spatial size matches pred_masks; resize if needed
        if gt_b.shape[-2:] != (H, W):
            # gt_b: [H_gt, W_gt] -> [1,1,H_gt,W_gt] -> interpolate -> [1,1,H,W] -> squeeze
            gt_b = F.interpolate(
                gt_b[None, None].float(), size=(H, W), mode="nearest"
            ).long()[0, 0]

        img_indices = []

        for q in range(Q):
            # Binary mask for region support: [H, W] bool
            mask_q = prob_masks[b, q] > mask_threshold
            num_pixels = mask_q.sum().item()
            if num_pixels == 0:
                # No confident pixels; skip this mask
                continue

            # Gather GT labels inside mask
            gt_vals = gt_b[mask_q]  # [num_pixels]

            if ignore_id is not None:
                # Remove ignore pixels from consideration
                valid = gt_vals != ignore_id
                if valid.sum().item() == 0:
                    # All pixels in this mask are ignore; skip this mask
                    continue
                gt_vals = gt_vals[valid]

            # Dominant (most frequent) GT label
            # torch.bincount requires non-negative ints
            bincount = torch.bincount(gt_vals.flatten())
            # Edge case: if all gt_vals are "ignore" greater than num_classes, bincount might be empty
            if bincount.numel() == 0:
                continue
            dominant_label = int(bincount.argmax().item())

            # If dominant label is not in known_set, treat as future-like
            if dominant_label not in known_set:
                img_indices.append(q)

        future_region_indices.append(img_indices)

    # Optional: log statistics for debugging
    total_future = sum(len(x) for x in future_region_indices)
    # logger.debug(
    #     f"[select_future_like_masks] B={B}, Q={Q}, total_future_regions={total_future} , known_class_ids ={known_set}"
    # )

    return future_region_indices


def compute_region_prototypes(mask_features, pred_masks, future_region_indices, mask_threshold):
    """
    Compute mean feature (prototype) for each future-like region.

    Inputs:
      mask_features        : Tensor [B, C, H, W]
      pred_masks           : Tensor [B, Q, H, W] (logits)
      future_region_indices: list[length B] of lists of query indices
      mask_threshold       : float

    Outputs:
      region_prototypes: Tensor [N_regions, C]
        Each row r is mean feature over pixels belonging to region r.
      region_assignments: list[length N_regions]
        region_assignments[r] = (b_idx, pixel_indices)
          - b_idx: int in [0, B)
          - pixel_indices: 1D LongTensor of indices in [0, H*W)
    """
    B, C, H, W = mask_features.shape
    B2, Q, H2, W2 = pred_masks.shape
    assert B == B2 and H == H2 and W == W2, "mask_features and pred_masks must share B,H,W"

    prob_masks = pred_masks.sigmoid()  # [B, Q, H, W]

    prototypes = []
    region_assignments = []

    for b in range(B):
        feats_b = mask_features[b]  # [C, H, W]
        for q in future_region_indices[b]:
            # Region support: [H, W] bool
            mask_q = prob_masks[b, q] > mask_threshold
            if mask_q.sum().item() == 0:
                continue

            # Flatten for indexing
            mask_flat = mask_q.view(-1)          #query's interest [H*W]
            feats_flat = feats_b.view(C, -1)     # [C, H*W]

            # Select features in region: [C, N_pix]
            region_feats = feats_flat[:, mask_flat]

            # Prototype: mean over pixels -> [C]
            proto = region_feats.mean(dim=1)     # [C]
            prototypes.append(proto)

            # Pixel indices in [0, H*W) that are nonzero
            pix_indices = mask_flat.nonzero(as_tuple=False).squeeze(1)  #[N_pix,1] -> [N_pix]
            region_assignments.append((b, pix_indices))

    if len(prototypes) == 0:
        logger.debug("[compute_region_prototypes] No valid future-like regions after thresholding.")
        return mask_features.new_zeros((0, C)), []

    region_prototypes = torch.stack(prototypes, dim=0)  # [N_regions, C]
    # logger.debug(
    #     f"[compute_region_prototypes] num_regions={region_prototypes.shape[0]}, "
    #     f"feat_dim={region_prototypes.shape[1]}"
    # )
    return region_prototypes, region_assignments


def sample_pixels_from_regions(mask_features, region_assignments, num_pixels_per_region):
    """
    Sample a fixed number of pixels from each region to build contrastive pairs.

    Inputs:
      mask_features        : Tensor [B, C, H, W]
      region_assignments   : list[length N_regions]
          Each element = (b_idx, pixel_indices)
            - b_idx: int in [0, B)
            - pixel_indices: 1D LongTensor of indices in [0, H*W)
      num_pixels_per_region: int, max pixels sampled per region

    Outputs:
      sampled_feats    : Tensor [N_samples, C]
      sampled_region_ids: LongTensor [N_samples], each in [0, N_regions)
    """
    B, C, H, W = mask_features.shape

    sampled_feats_list = []
    sampled_ids_list = []

    for region_id, (b_idx, pix_indices) in enumerate(region_assignments):
        if pix_indices.numel() == 0:
            continue

        # Randomly sample up to num_pixels_per_region indices
        if pix_indices.numel() > num_pixels_per_region:
            perm = torch.randperm(pix_indices.numel(), device=pix_indices.device)
            pix_indices_region = pix_indices[perm[:num_pixels_per_region]]
        else:
            pix_indices_region = pix_indices

        # Convert 1D indices [0, H*W) to features
        feats_b = mask_features[b_idx].view(C, -1)             # [C, H*W]
        feats_region = feats_b[:, pix_indices_region]          # [C, N_samples_region]
        feats_region = feats_region.transpose(0, 1)            # [N_samples_region, C]

        sampled_feats_list.append(feats_region)
        sampled_ids_list.append(
            torch.full(
                (feats_region.shape[0],),
                fill_value=region_id,
                dtype=torch.long,
                device=mask_features.device,
            )
        )

    if len(sampled_feats_list) == 0:
        # logger.debug("[sample_pixels_from_regions] No pixels sampled from any region.")
        return mask_features.new_zeros((0, C)), mask_features.new_zeros((0,), dtype=torch.long)

    sampled_feats = torch.cat(sampled_feats_list, dim=0)  # [N_samples, C]
    sampled_region_ids = torch.cat(sampled_ids_list, dim=0)  # [N_samples]

    # logger.debug(
    #     f"[sample_pixels_from_regions] N_regions={len(region_assignments)}, "
    #     f"N_samples={sampled_feats.shape[0]}, feat_dim={C}"
    # )

    return sampled_feats, sampled_region_ids


def info_nce_loss(pixel_feats, pos_prototypes, all_prototypes, temperature):
    """
    Compute InfoNCE loss for pixel-to-region contrast.

    Inputs:
      pixel_feats    : Tensor [N, C]
        Features of sampled pixels.
      pos_prototypes : Tensor [N, C]
        For each pixel i, the prototype of its own region.
      all_prototypes : Tensor [M, C]
        All region prototypes in this batch (negatives + some positives).
      temperature    : float τ

    Steps:
      - Normalize all vectors along C.
      - For each pixel i, compute logits_i[j] = sim(f_i, r_j) / τ,
        where sim is cosine similarity.
      - The positive logit is sim(f_i, p_i) / τ.
      - L_i = - log ( exp(pos_logit_i) / sum_j exp(logits_i[j]) )
      - L = mean_i L_i
    """
    assert pixel_feats.shape == pos_prototypes.shape, \
        f"pixel_feats {pixel_feats.shape} and pos_prototypes {pos_prototypes.shape} must match."

    # Normalize
    pixel_feats = F.normalize(pixel_feats, dim=1)       # [N, C]
    pos_prototypes = F.normalize(pos_prototypes, dim=1) # [N, C]
    all_prototypes = F.normalize(all_prototypes, dim=1) # [M, C]

    N, C = pixel_feats.shape #num_sample_pix x channels
    M = all_prototypes.shape[0] #num of regions/prototypes

	# Positive logits: sim(f_i, p_i) / τ = (f_i · p_i) / τ
    pos_logit = (pixel_feats * pos_prototypes).sum(dim=1, keepdim=True) / temperature  #∑_c(N,C ⊙ N,C ) -> [N, 1]


    # logits: [N, M] = f_i · r_j / τ
    logits = pixel_feats @ all_prototypes.t()  #N,C x C,M -> [N, M] futcr_author1*: each pixel and its region/prototype scores
    logits = logits / temperature

   
    # log softmax probability of positive class:
    # log_prob_i = pos_logit_i - logsumexp_j logits_i[j] futcr_author1*: not that lne^x = x, ln∑e^x -> logsumexpx
    log_prob = pos_logit - logits.logsumexp(dim=1, keepdim=True)  # [N, 1]

    loss = -log_prob.mean()  # scalar

    # logger.debug(
    #     f"[info_nce_loss] N={N}, M={M}, temperature={temperature}, loss={float(loss.item()):.6f}"
    # )

    return loss


def ignore_repulsion_loss(
    mask_features,
    gt_semantic,
    known_class_ids,
    ignore_id,
    max_ignore_pixels=1024,
    margin=0.0,
):
    """
    Push features of ignore/unlabeled pixels away from known-class prototypes.

    Inputs:
      mask_features : Tensor [B, C, H, W]
        Pixel embeddings from the pixel decoder.
      gt_semantic   : list[Tensor H×W] length B
        Per-pixel labels built from instance targets; known classes or ignore_id.
      known_class_ids: iterable[int]
        Class indices considered "known" at this task.
      ignore_id     : int
        Label assigned to unlabeled/unknown pixels in gt_semantic.
      max_ignore_pixels: int
        Max number of ignore pixels sampled per batch.
      margin        : float
        Cosine similarity margin; we penalize ignore pixels whose max
        similarity to any known prototype exceeds this margin.

    Output:
      loss_ignore: scalar Tensor
        Zero if no ignore or no known-class pixels exist.
    """
    B, C, H, W = mask_features.shape
    device = mask_features.device
    known_set = set(known_class_ids)

    class_feats = {}        # cid -> list of [N_c_chunk, C]
    ignore_feats_list = []  # list of [N_ignore_chunk, C]

    for b in range(B):
        gt_b = gt_semantic[b]  # [H_b, W_b]

        # Resize GT to match feature map if needed
        if gt_b.shape[-2:] != (H, W):
            gt_b = F.interpolate(
                gt_b[None, None].float(), size=(H, W), mode="nearest"
            ).long()[0, 0]

        feats_b = mask_features[b]              # [C, H, W]
        feats_flat = feats_b.view(C, -1).t()    # [H*W, C]
        gt_flat = gt_b.view(-1)                 # [H*W]

        # Collect known-class pixel features
        for cid in known_set:
            mask_c = (gt_flat == cid)          # [H*W] bool
            if mask_c.any():
                feats_c = feats_flat[mask_c]   # [N_c, C]
                if cid not in class_feats:
                    class_feats[cid] = [feats_c]
                else:
                    class_feats[cid].append(feats_c)

        # Collect ignore pixel features
        mask_ignore = (gt_flat == ignore_id)
        if mask_ignore.any():
            feats_ignore = feats_flat[mask_ignore]  # [N_ignore_b, C]
            ignore_feats_list.append(feats_ignore)

    # If no ignore pixels or no known-class pixels, no repulsion needed
    if len(ignore_feats_list) == 0 or len(class_feats) == 0:
        return mask_features.sum() * 0.0

    # Build prototypes: mean feature per known class across batch
    prototypes = []
    for cid, chunks in class_feats.items():
        feats_cat = torch.cat(chunks, dim=0)   # [N_c_total, C]
        proto_c = feats_cat.mean(dim=0)        # [C]
        prototypes.append(proto_c)
    known_prototypes = torch.stack(prototypes, dim=0)  # [N_known_in_batch, C]

    # Gather ignore features, sample up to max_ignore_pixels
    ignore_feats = torch.cat(ignore_feats_list, dim=0)  # [N_ignore_total, C]
    if ignore_feats.shape[0] > max_ignore_pixels:
        perm = torch.randperm(ignore_feats.shape[0], device=ignore_feats.device)
        ignore_feats = ignore_feats[perm[:max_ignore_pixels]]
    N_ignore = ignore_feats.shape[0]
    if N_ignore == 0:
        return mask_features.sum() * 0.0

    # Normalize features and prototypes
    ignore_feats = F.normalize(ignore_feats, dim=1)         # [N_ignore, C]
    known_prototypes = F.normalize(known_prototypes, dim=1) # [N_known, C]

    # Cosine similarities: sim[i, j] = f_i · mu_j
    sim = ignore_feats @ known_prototypes.t()  # [N_ignore, N_known]

    # For each ignore feature, get max similarity over known classes
    max_sim, _ = sim.max(dim=1)  # [N_ignore]

    # Hinge: max(0, max_sim - margin)
    loss_ignore = F.relu(max_sim - margin).mean()

    logger.debug(
        f"[ignore_repulsion_loss] N_ignore={N_ignore}, "
        f"N_known={known_prototypes.shape[0]}, margin={margin}, "
        f"loss_ignore={float(loss_ignore.item()):.6f}"
    )

    return loss_ignore




def debug_plot_known_vs_ignore(
    gt_semantic_b,
    known_class_ids,
    ignore_id,
    save_dir,
    prefix="debug_known_ignore",
    step_idx=0,
):
    os.makedirs(save_dir, exist_ok=True)
    known_set = set(known_class_ids)

    # Move to CPU for plotting
    gt_b = gt_semantic_b.detach()
    device = gt_b.device

    # Build boolean masks first
    known_mask_bool = torch.zeros_like(gt_b, dtype=torch.bool, device=device)
    for cid in known_set:
        known_mask_bool |= (gt_b == cid)

    ignore_mask_bool = (gt_b == ignore_id)

    # Convert to numpy for imshow (0/1 float)
    gt_np = gt_b.cpu().numpy()
    known_mask_np = known_mask_bool.float().cpu().numpy()
    ignore_mask_np = ignore_mask_bool.float().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax0, ax1, ax2 = axes

    im0 = ax0.imshow(gt_np, cmap="tab20")
    ax0.set_title("gt_semantic (class ids)")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    im1 = ax1.imshow(known_mask_np, cmap="gray")
    ax1.set_title("known classes (1) vs others (0)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(ignore_mask_np, cmap="gray")
    ax2.set_title("ignore_id pixels (1)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    out_path = os.path.join(save_dir, f"{prefix}_step{step_idx}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

    logger.info(
        f"[debug_plot_known_vs_ignore] Saved comparison plot to {out_path}; "
        f"unique gt labels={torch.unique(gt_b)}, ignore_id={ignore_id}"
    )
    
    
    




# # mask2former/modeling/future_aware/helper_functions.py

# import torch
# import torch.nn.functional as F
# import os
# import torch
# import matplotlib.pyplot as plt

# from . import get_future_aware_logger

# logger = get_future_aware_logger()


# def select_future_like_masks(
#     pred_masks,
#     gt_semantic,
#     known_class_ids,
#     mask_threshold,
#     ignore_id=None,
#     min_region_pixels=2, #50 TODO: investigate this
#     min_mean_conf=0.1,#0.5
# ):
#     """
#     Identify predicted masks that likely correspond to *future-like* regions.

#     Inputs:
#       pred_masks     : Tensor [B, Q, H, W] (logits)
#       gt_semantic    : list[Tensor H×W] length B, int labels per pixel
#                        (built from instance targets: labels + masks)
#       known_class_ids: set/list[int], classes already known at this task
#       mask_threshold : float, threshold on sigmoid(pred_masks) to define region support
#       ignore_id      : int or None, label used for unlabeled pixels in gt_semantic
#       min_region_pixels: int, minimum number of pixels in a region to be considered
#       min_mean_conf  : float, minimum mean mask probability inside the region

#     Output:
#       future_region_indices: list of length B
#         future_region_indices[b]: list[int] of query indices q (0 ≤ q < Q)
#         for which the mask is future-like (mostly unlabeled / ignore).
#     """
#     B, Q, H, W = pred_masks.shape
#     # Convert logits to probabilities once: [B, Q, H, W]
#     prob_masks = pred_masks.sigmoid()

#     known_set = set(known_class_ids)
#     future_region_indices = []

#     for b in range(B):
#         gt_b = gt_semantic[b]  # [H_b, W_b], int

#         # Ensure GT spatial size matches pred_masks; resize if needed
#         if gt_b.shape[-2:] != (H, W):
#             # gt_b: [H_gt, W_gt] -> [1,1,H_gt,W_gt] -> interpolate -> [1,1,H,W] -> squeeze
#             gt_b = F.interpolate(
#                 gt_b[None, None].float(), size=(H, W), mode="nearest"
#             ).long()[0, 0]

#         img_indices = []

#         for q in range(Q):
#             # 1) Binary region support: [H, W] bool
#             mask_q = prob_masks[b, q] > mask_threshold
#             num_pixels = int(mask_q.sum().item())
            
            
#             # logger.info(
#             #     dict(
#             #         num_pixels=num_pixels,
#             #         min_region_pixels=min_region_pixels,
#             #         mean_conf=float(prob_masks[b, q][mask_q].mean().item())
#             #     )
#             # )
            
#             if num_pixels == 0:
#                 # logger.info(f'Num pixels is 0: {num_pixels}, some prob: {prob_masks[b, q][:5]}')
#                 # No confident pixels; skip this mask
#                 continue

#             # Optional: minimum region size filter
#             if num_pixels < min_region_pixels:
#                 # logger.info(f'Num pixels is low: {num_pixels}')
#                 continue

#             # 2) Confidence filter: mean prob inside region
#             mean_conf = float(prob_masks[b, q][mask_q].mean().item())
#             if mean_conf < min_mean_conf:
#                 # logger.info(f'Mean conf is too low: {mean_conf}')
#                 continue

#             # 3) Gather GT labels inside region: [num_pixels]
#             gt_vals = gt_b[mask_q]

#             # If ignore_id is used in gt_semantic, separate known vs ignore
#             if ignore_id is not None:
#                 # Boolean masks for ignore and non-ignore pixels
#                 is_ignore = (gt_vals == ignore_id)
#                 num_ignore = int(is_ignore.sum().item())
#                 num_valid = num_pixels - num_ignore

#                 # -------------------------------------------------------------------
#                 # Case 1: region is dominated by labeled (known) pixels → treat as known
#                 # -------------------------------------------------------------------
#                 if num_valid > 0 and num_valid >= num_ignore :
#                     # Among non-ignore pixels, find dominant label
#                     valid_vals = gt_vals[~is_ignore]
#                     bincount_valid = torch.bincount(valid_vals.flatten())
#                     if bincount_valid.numel() == 0:
#                         # logger.warning(f'Valid  bincount_valid.numel() is  0 {bincount_valid.numel()}')
#                         continue
#                     dominant_known = int(bincount_valid.argmax().item())
#                     if dominant_known in known_set:
#                         # Strongly overlaps some known class → not future-like
#                         # logger.warning(f'Dominant known in known set {dominant_known}')
#                         continue

#                 # Case B: region mostly ignore/unlabeled → treat as future-like
#                 # We require that ignore pixels dominate the region.
#                 if num_ignore > num_valid:
#                     img_indices.append(q)
#                 # If neither case clearly holds, you can choose to skip or treat as future.
#                 # Here we skip ambiguous regions.
#             else:
#                 # logger.error(f'No ignore id seen!')
#                 # No ignore_id: fall back to "label not in known_set" heuristic
#                 bincount = torch.bincount(gt_vals.flatten())
#                 if bincount.numel() == 0:
#                     continue
#                 dominant_label = int(bincount.argmax().item())
#                 if dominant_label not in known_set:
#                     img_indices.append(q)

#         future_region_indices.append(img_indices)

#     total_future = sum(len(x) for x in future_region_indices)
#     # logger.debug(
#     #     f"[select_future_like_masks] B={B}, Q={Q}, total_future_regions={total_future} "
#     # )

#     return future_region_indices



# def select_future_like_masks_old(pred_masks, gt_semantic, known_class_ids, mask_threshold,ignore_id=None):
#     """
#     Identify predicted masks that likely correspond to *future* classes.

#     Inputs:
#       pred_masks    : Tensor [B, Q, H, W] (logits)
#       gt_semantic   : list[Tensor H×W] length B, int labels per pixel
#                       (constructed from targets: labels + instance masks)
#       known_class_ids: set/list[int], classes already known at this task
#       mask_threshold: float, threshold on sigmoid(pred_masks) to define region support

#     Output:
#       future_region_indices: list of length B
#         future_region_indices[b]: list[int] of query indices q (0 ≤ q < Q)
#         such that the dominant GT label in mask (b,q) is *not* in known_class_ids.
#     """
#     B, Q, H, W = pred_masks.shape
#     # Convert to probabilities once
#     prob_masks = pred_masks.sigmoid()  # [B, Q, H, W]

#     # Robust set for membership test
#     known_set = set(known_class_ids)

#     future_region_indices = []

#     for b in range(B):
#         gt_b = gt_semantic[b]  # [H_gt, W_gt], int
#         # logger.info(f"torch.unique(gt_b) : {torch.unique(gt_b)}")

#         # Ensure GT spatial size matches pred_masks; resize if needed
#         if gt_b.shape[-2:] != (H, W):
#             # gt_b: [H_gt, W_gt] -> [1,1,H_gt,W_gt] -> interpolate -> [1,1,H,W] -> squeeze
#             gt_b = F.interpolate(
#                 gt_b[None, None].float(), size=(H, W), mode="nearest"
#             ).long()[0, 0]

#         img_indices = []

#         for q in range(Q):
#             # Binary mask for region support: [H, W] bool
#             mask_q = prob_masks[b, q] > mask_threshold
#             num_pixels = mask_q.sum().item()
#             if num_pixels == 0:
#                 # No confident pixels; skip this mask
#                 continue

#             # Gather GT labels inside mask
#             gt_vals = gt_b[mask_q]  # [num_pixels]

#             if ignore_id is not None:
#                 # Remove ignore pixels from consideration
#                 valid = gt_vals != ignore_id
#                 if valid.sum().item() == 0:
#                     # All pixels in this mask are ignore; skip this mask
#                     continue
#                 gt_vals = gt_vals[valid]

#             # Dominant (most frequent) GT label
#             # torch.bincount requires non-negative ints
#             bincount = torch.bincount(gt_vals.flatten())
#             # Edge case: if all gt_vals are "ignore" greater than num_classes, bincount might be empty
#             if bincount.numel() == 0:
#                 continue
#             dominant_label = int(bincount.argmax().item())

#             # If dominant label is not in known_set, treat as future-like
#             if dominant_label not in known_set:
#                 img_indices.append(q)

#         future_region_indices.append(img_indices)

#     # Optional: log statistics for debugging
#     total_future = sum(len(x) for x in future_region_indices)
#     # logger.debug(
#     #     f"[select_future_like_masks] B={B}, Q={Q}, total_future_regions={total_future} , known_class_ids ={known_set}"
#     # )

#     return future_region_indices


# def compute_region_prototypes(mask_features, pred_masks, future_region_indices, mask_threshold):
#     """
#     Compute mean feature (prototype) for each future-like region.

#     Inputs:
#       mask_features        : Tensor [B, C, H, W]
#       pred_masks           : Tensor [B, Q, H, W] (logits)
#       future_region_indices: list[length B] of lists of query indices
#       mask_threshold       : float

#     Outputs:
#       region_prototypes: Tensor [N_regions, C]
#         Each row r is mean feature over pixels belonging to region r.
#       region_assignments: list[length N_regions]
#         region_assignments[r] = (b_idx, pixel_indices)
#           - b_idx: int in [0, B)
#           - pixel_indices: 1D LongTensor of indices in [0, H*W)
#     """
#     B, C, H, W = mask_features.shape
#     B2, Q, H2, W2 = pred_masks.shape
#     assert B == B2 and H == H2 and W == W2, "mask_features and pred_masks must share B,H,W"

#     prob_masks = pred_masks.sigmoid()  # [B, Q, H, W]

#     prototypes = []
#     region_assignments = []

#     for b in range(B):
#         feats_b = mask_features[b]  # [C, H, W]
#         for q in future_region_indices[b]:
#             # Region support: [H, W] bool
#             mask_q = prob_masks[b, q] > mask_threshold
#             if mask_q.sum().item() == 0:
#                 continue

#             # Flatten for indexing
#             mask_flat = mask_q.view(-1)          #query's interest [H*W]
#             feats_flat = feats_b.view(C, -1)     # [C, H*W]

#             # Select features in region: [C, N_pix]
#             region_feats = feats_flat[:, mask_flat]

#             # Prototype: mean over pixels -> [C]
#             proto = region_feats.mean(dim=1)     # [C]
#             prototypes.append(proto)

#             # Pixel indices in [0, H*W) that are nonzero
#             pix_indices = mask_flat.nonzero(as_tuple=False).squeeze(1)  #[N_pix,1] -> [N_pix]
#             region_assignments.append((b, pix_indices))

#     if len(prototypes) == 0:
#         logger.debug("[compute_region_prototypes] No valid future-like regions after thresholding.")
#         return mask_features.new_zeros((0, C)), []

#     region_prototypes = torch.stack(prototypes, dim=0)  # [N_regions, C]
#     # logger.debug(
#     #     f"[compute_region_prototypes] num_regions={region_prototypes.shape[0]}, "
#     #     f"feat_dim={region_prototypes.shape[1]}"
#     # )
#     return region_prototypes, region_assignments


# def sample_pixels_from_regions(mask_features, region_assignments, num_pixels_per_region):
#     """
#     Sample a fixed number of pixels from each region to build contrastive pairs.

#     Inputs:
#       mask_features        : Tensor [B, C, H, W]
#       region_assignments   : list[length N_regions]
#           Each element = (b_idx, pixel_indices)
#             - b_idx: int in [0, B)
#             - pixel_indices: 1D LongTensor of indices in [0, H*W)
#       num_pixels_per_region: int, max pixels sampled per region

#     Outputs:
#       sampled_feats    : Tensor [N_samples, C]
#       sampled_region_ids: LongTensor [N_samples], each in [0, N_regions)
#     """
#     B, C, H, W = mask_features.shape

#     sampled_feats_list = []
#     sampled_ids_list = []

#     for region_id, (b_idx, pix_indices) in enumerate(region_assignments):
#         if pix_indices.numel() == 0:
#             continue

#         # Randomly sample up to num_pixels_per_region indices
#         if pix_indices.numel() > num_pixels_per_region:
#             perm = torch.randperm(pix_indices.numel(), device=pix_indices.device)
#             pix_indices_region = pix_indices[perm[:num_pixels_per_region]]
#         else:
#             pix_indices_region = pix_indices

#         # Convert 1D indices [0, H*W) to features
#         feats_b = mask_features[b_idx].view(C, -1)             # [C, H*W]
#         feats_region = feats_b[:, pix_indices_region]          # [C, N_samples_region]
#         feats_region = feats_region.transpose(0, 1)            # [N_samples_region, C]

#         sampled_feats_list.append(feats_region)
#         sampled_ids_list.append(
#             torch.full(
#                 (feats_region.shape[0],),
#                 fill_value=region_id,
#                 dtype=torch.long,
#                 device=mask_features.device,
#             )
#         )

#     if len(sampled_feats_list) == 0:
#         # logger.debug("[sample_pixels_from_regions] No pixels sampled from any region.")
#         return mask_features.new_zeros((0, C)), mask_features.new_zeros((0,), dtype=torch.long)

#     sampled_feats = torch.cat(sampled_feats_list, dim=0)  # [N_samples, C]
#     sampled_region_ids = torch.cat(sampled_ids_list, dim=0)  # [N_samples]

#     # logger.debug(
#     #     f"[sample_pixels_from_regions] N_regions={len(region_assignments)}, "
#     #     f"N_samples={sampled_feats.shape[0]}, feat_dim={C}"
#     # )

#     return sampled_feats, sampled_region_ids


# def info_nce_loss(pixel_feats, pos_prototypes, all_prototypes, temperature):
#     """
#     Compute InfoNCE loss for pixel-to-region contrast.

#     Inputs:
#       pixel_feats    : Tensor [N, C]
#         Features of sampled pixels.
#       pos_prototypes : Tensor [N, C]
#         For each pixel i, the prototype of its own region.
#       all_prototypes : Tensor [M, C]
#         All region prototypes in this batch (negatives + some positives).
#       temperature    : float τ

#     Steps:
#       - Normalize all vectors along C.
#       - For each pixel i, compute logits_i[j] = sim(f_i, r_j) / τ,
#         where sim is cosine similarity.
#       - The positive logit is sim(f_i, p_i) / τ.
#       - L_i = - log ( exp(pos_logit_i) / sum_j exp(logits_i[j]) )
#       - L = mean_i L_i
#     """
#     assert pixel_feats.shape == pos_prototypes.shape, \
#         f"pixel_feats {pixel_feats.shape} and pos_prototypes {pos_prototypes.shape} must match."

#     # Normalize
#     pixel_feats = F.normalize(pixel_feats, dim=1)       # [N, C]
#     pos_prototypes = F.normalize(pos_prototypes, dim=1) # [N, C]
#     all_prototypes = F.normalize(all_prototypes, dim=1) # [M, C]

#     N, C = pixel_feats.shape #num_sample_pix x channels
#     M = all_prototypes.shape[0] #num of regions/prototypes

# 	# Positive logits: sim(f_i, p_i) / τ = (f_i · p_i) / τ
#     pos_logit = (pixel_feats * pos_prototypes).sum(dim=1, keepdim=True) / temperature  #∑_c(N,C ⊙ N,C ) -> [N, 1]


#     # logits: [N, M] = f_i · r_j / τ
#     logits = pixel_feats @ all_prototypes.t()  #N,C x C,M -> [N, M] futcr_author1*: each pixel and its region/prototype scores
#     logits = logits / temperature

   
#     # log softmax probability of positive class:
#     # log_prob_i = pos_logit_i - logsumexp_j logits_i[j] futcr_author1*: not that lne^x = x, ln∑e^x -> logsumexpx
#     log_prob = pos_logit - logits.logsumexp(dim=1, keepdim=True)  # [N, 1]

#     loss = -log_prob.mean()  # scalar

#     # logger.debug(
#     #     f"[info_nce_loss] N={N}, M={M}, temperature={temperature}, loss={float(loss.item()):.6f}"
#     # )

#     return loss


# def ignore_repulsion_loss(
#     mask_features,
#     gt_semantic,
#     known_class_ids,
#     ignore_id,
#     max_ignore_pixels=1024,
#     margin=0.0,
# ):
#     """
#     Push features of ignore/unlabeled pixels away from known-class prototypes.

#     Inputs:
#       mask_features : Tensor [B, C, H, W]
#         Pixel embeddings from the pixel decoder.
#       gt_semantic   : list[Tensor H×W] length B
#         Per-pixel labels built from instance targets; known classes or ignore_id.
#       known_class_ids: iterable[int]
#         Class indices considered "known" at this task.
#       ignore_id     : int
#         Label assigned to unlabeled/unknown pixels in gt_semantic.
#       max_ignore_pixels: int
#         Max number of ignore pixels sampled per batch.
#       margin        : float
#         Cosine similarity margin; we penalize ignore pixels whose max
#         similarity to any known prototype exceeds this margin.

#     Output:
#       loss_ignore: scalar Tensor
#         Zero if no ignore or no known-class pixels exist.
#     """
#     B, C, H, W = mask_features.shape
#     device = mask_features.device
#     known_set = set(known_class_ids)

#     class_feats = {}        # cid -> list of [N_c_chunk, C]
#     ignore_feats_list = []  # list of [N_ignore_chunk, C]

#     for b in range(B):
#         gt_b = gt_semantic[b]  # [H_b, W_b]

#         # Resize GT to match feature map if needed
#         if gt_b.shape[-2:] != (H, W):
#             gt_b = F.interpolate(
#                 gt_b[None, None].float(), size=(H, W), mode="nearest"
#             ).long()[0, 0]

#         feats_b = mask_features[b]              # [C, H, W]
#         feats_flat = feats_b.view(C, -1).t()    # [H*W, C]
#         gt_flat = gt_b.view(-1)                 # [H*W]

#         # Collect known-class pixel features
#         for cid in known_set:
#             mask_c = (gt_flat == cid)          # [H*W] bool
#             if mask_c.any():
#                 feats_c = feats_flat[mask_c]   # [N_c, C]
#                 if cid not in class_feats:
#                     class_feats[cid] = [feats_c]
#                 else:
#                     class_feats[cid].append(feats_c)

#         # Collect ignore pixel features
#         mask_ignore = (gt_flat == ignore_id)
#         if mask_ignore.any():
#             feats_ignore = feats_flat[mask_ignore]  # [N_ignore_b, C]
#             ignore_feats_list.append(feats_ignore)

#     # If no ignore pixels or no known-class pixels, no repulsion needed
#     if len(ignore_feats_list) == 0 or len(class_feats) == 0:
#         return mask_features.sum() * 0.0

#     # Build prototypes: mean feature per known class across batch
#     prototypes = []
#     for cid, chunks in class_feats.items():
#         feats_cat = torch.cat(chunks, dim=0)   # [N_c_total, C]
#         proto_c = feats_cat.mean(dim=0)        # [C]
#         prototypes.append(proto_c)
#     known_prototypes = torch.stack(prototypes, dim=0)  # [N_known_in_batch, C]

#     # Gather ignore features, sample up to max_ignore_pixels
#     ignore_feats = torch.cat(ignore_feats_list, dim=0)  # [N_ignore_total, C]
#     if ignore_feats.shape[0] > max_ignore_pixels:
#         perm = torch.randperm(ignore_feats.shape[0], device=ignore_feats.device)
#         ignore_feats = ignore_feats[perm[:max_ignore_pixels]]
#     N_ignore = ignore_feats.shape[0]
#     if N_ignore == 0:
#         return mask_features.sum() * 0.0

#     # Normalize features and prototypes
#     ignore_feats = F.normalize(ignore_feats, dim=1)         # [N_ignore, C]
#     known_prototypes = F.normalize(known_prototypes, dim=1) # [N_known, C]

#     # Cosine similarities: sim[i, j] = f_i · mu_j
#     sim = ignore_feats @ known_prototypes.t()  # [N_ignore, N_known]

#     # For each ignore feature, get max similarity over known classes
#     max_sim, _ = sim.max(dim=1)  # [N_ignore]

#     # Hinge: max(0, max_sim - margin)
#     loss_ignore = F.relu(max_sim - margin).mean()

#     logger.debug(
#         f"[ignore_repulsion_loss] N_ignore={N_ignore}, "
#         f"N_known={known_prototypes.shape[0]}, margin={margin}, "
#         f"loss_ignore={float(loss_ignore.item()):.6f}"
#     )

#     return loss_ignore





# def debug_plot_known_vs_ignore(
#     gt_semantic_b,
#     known_class_ids,
#     ignore_id,
#     save_dir,
#     prefix="debug_known_ignore",
#     step_idx=0,
# ):
#     os.makedirs(save_dir, exist_ok=True)
#     known_set = set(known_class_ids)

#     # Move to CPU for plotting
#     gt_b = gt_semantic_b.detach()
#     device = gt_b.device

#     # Build boolean masks first
#     known_mask_bool = torch.zeros_like(gt_b, dtype=torch.bool, device=device)
#     for cid in known_set:
#         known_mask_bool |= (gt_b == cid)

#     ignore_mask_bool = (gt_b == ignore_id)

#     # Convert to numpy for imshow (0/1 float)
#     gt_np = gt_b.cpu().numpy()
#     known_mask_np = known_mask_bool.float().cpu().numpy()
#     ignore_mask_np = ignore_mask_bool.float().cpu().numpy()

#     fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#     ax0, ax1, ax2 = axes

#     im0 = ax0.imshow(gt_np, cmap="tab20")
#     ax0.set_title("gt_semantic (class ids)")
#     fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

#     im1 = ax1.imshow(known_mask_np, cmap="gray")
#     ax1.set_title("known classes (1) vs others (0)")
#     fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

#     im2 = ax2.imshow(ignore_mask_np, cmap="gray")
#     ax2.set_title("ignore_id pixels (1)")
#     fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

#     for ax in axes:
#         ax.axis("off")

#     out_path = os.path.join(save_dir, f"{prefix}_step{step_idx}.png")
#     plt.tight_layout()
#     plt.savefig(out_path)
#     plt.close(fig)

#     logger.info(
#         f"[debug_plot_known_vs_ignore] Saved comparison plot to {out_path}; "
#         f"unique gt labels={torch.unique(gt_b)}, ignore_id={ignore_id}"
#     )
