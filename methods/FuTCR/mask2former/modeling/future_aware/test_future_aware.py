import torch
import sys
import os
from types import SimpleNamespace as NS
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


from mask2former.modeling.future_aware.future_region_contrast_update import (
    FutureRegionContrastModule,
)

hparam_matrix = [(0.5 ,0.5 ,0.5),(0.35 ,0.45, 0.5),(100 ,80, 48),(0.8 ,0.5, 0.1),(4 ,3 ,4)]

def build_dummy_cfg(hparam_vec=None):
    lw, mt, nppr, rw, ncm = [float(f"{i.item():.4f}") for i in hparam_vec]
    print(f"param: {lw, mt, nppr, rw, ncm}" )
    
    cfg = NS()
    cfg.CONT = NS()
    cfg.CONT.FUTURE_AWARE = NS()
    cfg.CONT.INC_CLS = 5
    
    fa = cfg.CONT.FUTURE_AWARE
    fa.ENABLE = True

    # Region contrast
    fa.REGION_CONTRAST_ENABLE = True
    fa.LOSS_WEIGHT = lw
    fa.NUM_SAMPLED_PIXELS_PER_REGION = int(nppr)
    fa.TEMPERATURE = 0.07
    fa.MASK_THRESHOLD = mt

    # Ignore repulsion
    fa.IGNORE_REPULSION_ENABLE = True
    fa.IGNORE_REPULSION_WEIGHT = rw
    fa.IGNORE_REPULSION_MARGIN = 0.0
    fa.MAX_IGNORE_PIXELS = 1024

    # Aux classifier
    fa.AUX_CLS_ENABLE = True
    fa.AUX_CLS_NUM_CLUSTERS = int(ncm *cfg.CONT.INC_CLS)
    fa.AUX_CLS_HIDDEN_DIM = 256
    fa.AUX_CLS_LOSS_WEIGHT = 0.1
    fa.AUX_CLS_UPDATE_FREQ = 100
    fa.AUX_CLS_BUFFER_SIZE = 4096

    return cfg


def main():
    for hparam_vec in torch.tensor(hparam_matrix).T:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = build_dummy_cfg(hparam_vec)

        module = FutureRegionContrastModule(cfg).to(device)

        # Dummy shapes
        B, C, H, W = 2, 256, 32, 32    # batch, channels, spatial
        Q = 10                         # queries per image
        num_classes = 150
        ignore_id = 150

        # mask_features: pixel embeddings [B, C, H, W]
        mask_features = torch.randn(B, C, H, W, device=device)

        # pred_masks: logits [B, Q, H, W]
        pred_masks = torch.randn(B, Q, H, W, device=device)

        # gt_semantic: list of [H, W] with ints in known classes or ignore
        gt_semantic = []
        for b in range(B):
            sem = torch.full((H, W), ignore_id, dtype=torch.long, device=device)
            # Paint a few known-class blobs
            sem[8:16, 8:16] = 3
            sem[20:28, 20:28] = 42
            gt_semantic.append(sem)

        # known classes: some subset of 0..num_classes-1
        known_class_ids = list(range(100))  # 0..49 known, others treated as future/unlabeled

        module.train()
        out = module(
            mask_features=mask_features,
            pred_masks=pred_masks,
            gt_semantic=gt_semantic,
            known_class_ids=known_class_ids,
            ignore_id=ignore_id,
        )

        print("loss_future_contrast:", out["loss_future_contrast"].item())


if __name__ == "__main__":
    main()
    #aux class head for all remaining classes or next set of classes?
