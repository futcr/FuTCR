# FuTCR: Future-Targeted Contrast and Repulsion for Continual Panoptic Segmentation

FuTCR is a query-based continual panoptic segmentation framework that treats unlabeled future-class evidence as a first-class training signal. It builds on Mask2Former-style decoders and SimCIS-style continual setups, adding future-aware region contrast and known-class repulsion.

---

## 🔧 Environment & Installation

1. Clone the repository and enter the FuTCR codebase:
```bash
git clone <this-repo-url> FuTCR
cd FuTCR/methods/FuTCR
```

2. Create a conda environment and install core dependencies:
```bash
conda create --name futcr python=3.8 -y
conda activate futcr

# PyTorch + CUDA (adapt versions to your system)
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch -c nvidia

# Detectron2 (match to your CUDA / PyTorch)
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Panoptic API and other utilities
pip install git+https://github.com/cocodataset/panopticapi.git
pip install -r requirements.txt
```

3. Compile the MSDeformAttn CUDA ops (required for Mask2Former-style decoders):
```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
```

---

## 📁 Data Preparation (ADE20K Continual Panoptic)

FuTCR assumes ADE20K is prepared in the standard Mask2Former format and then reorganized for continual segmentation.

From `methods/FuTCR`:

1. Download ADE20K and instance annotations and place them under `datasets/`:
```bash
cd datasets
# Download ADEChallengeData2016 and instance annotations (see original ADE20K links)
# After extraction, you should have:
# ADEChallengeData2016/
#   images/
#   annotations/
#   annotations_instance/
#   objectInfo150.txt
#   sceneCategories.txt
cd ..
```

2. Generate semantic / instance / panoptic annotations in Detectron2 format (Mask2Former-style):
```bash
python datasets/prepare_ade20k_sem_seg.py
python datasets/prepare_ade20k_pan_seg.py
python datasets/prepare_ade20k_ins_seg.py
```

3. Prepare continual splits and JSONs:
```bash
python continual/prepare_datasets.py
# Reorganized annotations are written to ./json
```

---

## 🧪 Controlled Data Subset Generation (Overlap / Disjoint Streams)

FuTCR uses controlled overlap/disjoint streams (e.g., 100–5, 100–10, 100–50 with (IMG_OV, CLS_OV) ≈ (100,75) and (0,75)).

At the repository root (`FuTCR/`):

1. Summarize dataset distributions and select subset indices (e.g., for 100–50):
   - Open and run the notebook  
     `dataset_summary_distrib_main_100-50.ipynb`  
     to generate the desired image subsets for base and incremental steps.

2. Generate JSON split files for the chosen protocol:
```bash
bash start.sh
# start.sh internally calls split_paths_resolver.py / shared/standard_prepare_datasets.py
# and writes protocol-specific JSONs under methods/FuTCR/json
```

This pipeline yields consistent overlap / disjoint-image splits shared by all experiments.

---

## 🚀 Training

All training scripts are launched from `methods/FuTCR`. Example (100–5 panoptic, overlap stream):

```bash
cd methods/FuTCR

# Example: train FuTCR on ADE20K 100–5 overlap stream
bash scripts/pan_100-5_continual_future_aware_seed.sh
```

Typical script usage:

- Base step (step 1): train Mask2Former-style panoptic backbone with the base class set.
- Incremental steps (step 2…T): call `train_continual.py` with the desired protocol (`100-5`, `100-10`, `100-50`), stream type (overlap vs disjoint-image), and FuTCR options (region contrast, known-future repulsion).

Each script reads configuration from `configs/` and JSON splits from `json/`, and writes logs and checkpoints to `output/`.

---

## ✅ Evaluation

Final-step evaluation is also performed from `methods/FuTCR`:

```bash
cd methods/FuTCR

# Example: evaluate last step (e.g., step 11 for 100–5)
bash scripts/panoptic_eval.sh 11
```

Evaluation scripts:

- Load the trained checkpoints from `output/`.
- Run panoptic inference (PQ\_old, PQ\_new, PQ\_all) under the chosen protocol and stream.
- Optionally generate qualitative panoptic visualizations into a paper/qualitative path.

---

## 🧩 Future-Aware Module and Configs

FuTCR’s core components live in:

```bash
methods/FuTCR/mask2former/modeling/future_aware/
```

Future-like region discovery, region contrast, and repulsion are enabled and tuned via the YAML configs in `methods/FuTCR/continual/config.py`, using the hyperparameters described in the paper (e.g., loss weights, thresholds, sampling settings). No direct edits to `future_aware/*.py` are required for standard runs.

---

## 📊 Reproducing Main Experiments

From `methods/FuTCR`:

- **100–5 / 100–10 / 100–50**:
  - Train and evaluate FuTCR and baselines (e.g., SimCIS-style) using the corresponding scripts under `scripts/`.
  - Use the same base splits and overlap/disjoint JSONs produced in the data-preparation stage.


---

## 🙏 Acknowledgements

FuTCR is implemented on top of:

- [Mask2Former](https://github.com/facebookresearch/Mask2Former) for panoptic segmentation.
- SimCIS-style continual segmentation infrastructure for query-based training and evaluation.

We gratefully acknowledge these projects and related continual segmentation work (e.g., SimCIS, BalConpas) for their foundational contributions.