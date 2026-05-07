# Third-Party Code and Data

This archive redistributes the following third-party material for the
sole purpose of reviewer convenience and experimental reproducibility.

## Source code

* `pytorch_grad_cam/` — implementations of Grad-CAM, Grad-CAM++, XGrad-CAM,
  Layer-CAM, Score-CAM, Ablation-CAM, and the ROAD and ADCC evaluation
  metrics. Distributed under the MIT License. Upstream project:
  *jacobgil/pytorch-grad-cam*.

* `reciprocam/` — implementation of Recipro-CAM for both convolutional
  networks and Vision Transformers, accompanying the original paper of
  the same name. Used here as a baseline.

The proposed method itself, contained in the `ddcam/` package, is the
authors' own work and is released for review with this supplement.

## Datasets

* `med_img/` — seventeen radiographs from the NIH ChestX-ray14 dataset.
  ChestX-ray14 is released by the United States National Institutes of
  Health for free use in research; redistribution is permitted with
  attribution to the dataset authors.

* `BBox_List_2017.csv` — the radiologist-drawn bounding-box annotation
  file released alongside ChestX-ray14.

* `test_folder/` — ten validation images from the ILSVRC-2012 ImageNet
  classification subset. ImageNet is licensed for non-commercial
  research use; the small number of images included here is intended
  solely for executing the supplied scripts during peer review and is
  not intended for any other distribution.

## Vendored python libraries

The libraries listed under "Source code" are bundled inside this archive
to remove any internet dependency and to fix the exact versions used in
the experiments. No modifications were made to the upstream code.
