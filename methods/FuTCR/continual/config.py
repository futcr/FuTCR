from detectron2.config import CfgNode as CN


def add_continual_config(cfg):
    cfg.CONT = CN()
    cfg.CONT.OLD_MODEL = True
    cfg.CONT.TOT_CLS = 150
    cfg.CONT.BASE_CLS = 100
    cfg.CONT.INC_CLS = 10
    cfg.CONT.SETTING = 'overlapped'
    cfg.CONT.TASK = 1
    cfg.CONT.WEIGHTS = None
    cfg.CONT.OLD_WEIGHTS = None
    cfg.CONT.MED_TOKENS_WEIGHT = 1.0
    cfg.CONT.MEMORY = False
    cfg.CONT.PSD_LABEL_THRESHOLD = 0.35
    cfg.CONT.PSD_OVERLAP_THRESHOLD = 0.8
    cfg.CONT.COLLECT_QUERY_MODE = False
    cfg.CONT.CUMULATIVE_PSDNUM = False
    cfg.CONT.WEIGHTED_SAMPLE = True
    cfg.CONT.LIB_SIZE = 80
    cfg.CONT.VQ_NUMBER = 3
    cfg.CONT.VQ_STORE = False
    cfg.CONT.FREEZE_LABEL = False
    cfg.CONT.KL_ALL = True
    cfg.CONT.KL_WEIGHT = 2.0
    
    cfg.CONT.KD_TYPE = 'kl'
    cfg.CONT.DISTRIBUTION_ALPHA = 0.5
    cfg.CONT.KD_TEMPERATURE = 0.1
    cfg.CONT.KD_TEMPERATURE2 = 0.1
    cfg.CONT.KD_DECODER = True
    cfg.CONT.FILTER_KD = False
    cfg.CONT.COMBINE_PSDLABEL = False
    cfg.CONT.ADD_POS = False
    
    #futcr_author1 added
    cfg.CONT.USE_PCL= False
    cfg.CONT.PCL_USE_SUPERVISED= False
    cfg.CONT.PCL_USE_AUTO_OVERLAP_SCALING= False #Only set True if you have severe overlap and model can't learn new classes
    cfg.CONT.PCL_TEMPERATURE= 0.1
    cfg.CONT.PCL_OVERLAP_THRESHOLD= 0.5
    cfg.CONT.KD_LAMBDA = 0.1
    
    #futcr_author1 2026-03-9
    cfg.CONT.FUTURE_AWARE = CN()
    cfg.CONT.FUTURE_AWARE.ENABLE = False          # master switch for future-aware module
    cfg.CONT.FUTURE_AWARE.LOSS_WEIGHT = 0.5      # small relative to main CE/mask losses
    
    # # Region-based pixel-to-region InfoNCE (on future-like masks)
    cfg.CONT.FUTURE_AWARE.REGION_CONTRAST_ENABLE = False
    cfg.CONT.FUTURE_AWARE.MASK_THRESHOLD = 0.40   # 0.5 classic mask prob threshold
    cfg.CONT.FUTURE_AWARE.NUM_SAMPLED_PIXELS_PER_REGION = 80  # 64 32–128 is a good range
    cfg.CONT.FUTURE_AWARE.TEMPERATURE = 0.07     # standard InfoNCE temperature
    
    #futcr_author1 2026-03-16
    # Ignore-repulsion branch (push ignore pixels away from known-class prototypes)
    cfg.CONT.FUTURE_AWARE.IGNORE_REPULSION_ENABLE = False
    cfg.CONT.FUTURE_AWARE.IGNORE_REPULSION_WEIGHT = 0.5
    cfg.CONT.FUTURE_AWARE.IGNORE_REPULSION_MARGIN = 0.0
    cfg.CONT.FUTURE_AWARE.MAX_IGNORE_PIXELS = 1024
    
    # Auxiliary classifier on region prototypes (PTF-style)
    cfg.CONT.FUTURE_AWARE.AUX_CLS_ENABLE = False      # turn on/off auxiliary branch
    cfg.CONT.FUTURE_AWARE.AUX_CLS_NUM_CLUSTERS = 32   # K (pseudo-classes / slots)
    cfg.CONT.FUTURE_AWARE.AUX_CLS_HIDDEN_DIM = 256    # hidden layer size in MLP
    cfg.CONT.FUTURE_AWARE.AUX_CLS_LOSS_WEIGHT = 0.1   # weight for auxiliary loss
    cfg.CONT.FUTURE_AWARE.AUX_CLS_UPDATE_FREQ = 100   # iterations between reclustering
    cfg.CONT.FUTURE_AWARE.AUX_CLS_BUFFER_SIZE = 4096  # max region prototypes in buffer

