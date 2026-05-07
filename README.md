# Supplementary Materials for *DD-CAM: Minimal Sufficient Explanations for Vision Models Using Delta Debugging*

This archive contains the source code and datasets required to reproduce
the three research questions reported in the paper.

* RQ1. Faithfulness of saliency maps on the ImageNet classification task.
* RQ2. Localization performance against radiologist-annotated regions on
  the NIH ChestX-ray14 dataset.
* RQ3. Stability of the recovered minimal sufficient sets under different
  orderings of the channel index.

Third-party libraries and datasets bundled with this archive are
documented in `NOTICE.md`.

## Directory layout

```
ddcam_supplement/
├── README.md                          This file.
├── NOTICE.md                          Third-party attribution.
├── BBox_List_2017.csv                 Radiologist bounding-box annotations
│                                          (884 boxes across 791 images).
├── ddcam/                             Core implementation of the proposed
│                                          method.
├── pytorch_grad_cam/                  Vendored copy of the pytorch-grad-cam
│                                          library, used for the CAM-family
│                                          baselines and for the ADCC and
│                                          ROAD metric implementations.
├── reciprocam/                        Vendored copy of the Recipro-CAM
│                                          baseline.
├── test_folder/                       Ten ImageNet validation images for
│                                          the RQ1 experiments.
├── med_img/                           Seventeen ChestX-ray14 images for
│                                          the RQ2 experiments. The
│                                          bounding-box CSV determines the
│                                          nineteen image-finding pairs
│                                          actually evaluated.
├── comparison.py                      RQ1 main script.
├── baselines_only.py                  RQ1 non-CAM baselines.
├── run_sis_imagenet.py                RQ1 Sufficient Input Subsets
│                                          baseline.
├── rollout_with_metrics.py            RQ1 Attention Rollout baseline.
├── attention_rollout_vit.py           Attention Rollout module.
├── sis_module.py                      Sufficient Input Subsets module.
├── minimality_localization.py         RQ2 main script.
├── minimality_localization_baselines.py  RQ2 non-CAM baselines.
├── run_sis_medical.py                 RQ2 SIS baseline.
├── stability_analysis.py              RQ3 main script.
└── stability_determinism_check.py     RQ3 control.
```

All experiment scripts are located at the top level of the archive and
must be executed with this directory as the current working directory.
This ensures that relative dataset paths and the import of the local
`ddcam` package both resolve correctly.

## Software requirements

The code has been tested with Python 3.10 and PyTorch 2.0. The following
packages are required.

```
pip install torch torchvision numpy scipy pillow opencv-python matplotlib pandas tqdm
pip install captum lime shap
pip install torchxrayvision
```

The libraries `pytorch_grad_cam` and `reciprocam` are bundled inside the
archive and must not be installed separately.

## Reproducing the experiments

The following commands assume the current working directory is the root
of this archive.

### RQ1 — Faithfulness on ImageNet

```
python comparison.py
python baselines_only.py --out_dir results_baselines
python run_sis_imagenet.py
python rollout_with_metrics.py
```

The metrics reported in Table 1 of the paper are Average Drop, Coherency,
Complexity, Average DCC, Increase in Confidence, and Average Drop in
Deletion. Their definitions are given in the appendix of the paper.

### RQ2 — Localization on ChestX-ray14

```
python minimality_localization.py --random-samples 200
python minimality_localization_baselines.py
python run_sis_medical.py
```

The metrics reported in Table 2 are Intersection over Union, Precision,
Recall, F1, and the number of connected regions, with binarization
threshold 0.5 and 8-way connectivity.

### RQ3 — Stability of the minimal sufficient set

```
python stability_analysis.py
python stability_determinism_check.py
```

The reported quantities are the Jaccard similarity of the channel sets,
the Intersection over Union of the resulting saliency maps, the
coefficient of variation of the subset size, and the standard deviation
of the retained prediction probability.

## Programmatic use

```python
import torch
import torchvision.models as M
from ddcam import DDCAMExplainer

model = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2).eval()

explainer = DDCAMExplainer(
    model=model,
    target_layer=model.layer4[-1],
    device="cuda",
    is_vit=False,
    strategy="auto",
)

heatmap, overlay = explainer.explain(image_tensor, target_class=None)
```

For Vision Transformers, set `is_vit=True` and use the last encoder
block, for example `model.encoder.layers[-1].ln_1`, as the target layer.
Setting `strategy="auto"` selects iterated single-unit removal for
classifier heads with a single linear layer and the standard
partition-and-reduce delta-debugging procedure for non-linear
classifier heads and for self-attention.

## Citation

Please cite the paper when using this code, and acknowledge the
third-party libraries and datasets enumerated in `NOTICE.md`.
