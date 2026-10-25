<div align="center">

<h1>HypCL: Adapting CLIP in Hyperbolic Space for Continual Learning</h1>


[![ICML 2026](https://img.shields.io/badge/ICML-2026-4b44ce.svg)](https://proceedings.mlr.press/v306/cheng26p.html) [![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**[Paper](https://proceedings.mlr.press/v306/cheng26p.html)** &nbsp;·&nbsp; **[Overview](#overview)** &nbsp;·&nbsp; **[Results](#main-results)** &nbsp;·&nbsp; **[Getting Started](#getting-started)** &nbsp;·&nbsp; **[Citation](#citation)**

</div>

---

> **TL;DR:** HypCL adapts CLIP in hyperbolic space for exemplar-free continual learning.

---

## Overview

Existing CLIP-based continual learning methods usually freeze the backbone, which limits their ability to learn discriminative features. HypCL instead adapts CLIP in hyperbolic space, whose exponentially growing volume naturally accommodates an expanding class space.

<div align="center">
<img src="assets/framework.png" alt="HypCL framework" width="900">
</div>

- **Hyperbolic adapters:** task-specific adapters on the visual encoder, composed sequentially in the Poincaré ball.
- **Hyperbolic classification:** predictions based on hyperbolic distances between image and text features.
- **Prototype calibration:** visual prototypes that calibrate predictions at inference.

## Main results

<div align="center">
<img src="assets/curves.png" alt="Per-task accuracy curves" width="900">
</div>

## Getting started

### 1. Installation

```bash
conda create -n cl python=3.10 pip -y --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda activate cl
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

The backbone is OpenCLIP `ViT-B-16` with `laion400m_e32` weights.
The model first loads `checkpoints/open_clip_pytorch_model.bin` when present;
otherwise OpenCLIP downloads the pre-trained weights on the first run.
The locally prepared checkpoint was downloaded through `hf-mirror.com`, and its
source revision and verified SHA256 are recorded in
`checkpoints/open_clip_pytorch_model.source.json`.

### 2. Data preparation

HypCL uses the same class subsets and train/test splits as ENGINE. Download the prepared datasets from the [ENGINE data instructions](https://github.com/RenaissCode/ENGINE#-requirements) and arrange them under one root:

```text
data/
├── cifar-100-python/          # downloaded automatically with --download
├── aircraft/{train,test}/
├── cars/{train,test}/
├── cub/{train,test}/
├── food101/{train,test}/
├── imagenetr/{train,test}/
├── objectnet/{train,test}/
├── sun/{train,test}/
└── ucf/{train,test}/
```

Local download sources are recorded in `data/sources.json`: the SJTU mirror
provides CIFAR100; the author's mainland China OneDrive mirrors provide Cars,
UCF, Aircraft, Food, and SUN. CUB, ImageNet-R, and ObjectNet use the author's
overseas OneDrive links. Downloaded archives remain in `data/downloads/` with
completion records and SHA256 digests. CIFAR100 is checked against its official
MD5. A locally calculated SHA256 records file identity; it is only an independent
integrity check when an expected digest is available from the source.

To resume dataset downloads and prepare the extracted splits:

```bash
conda activate cl
python -u scripts/download_assets.py
python -u scripts/prepare_datasets.py cifar100 ucf101 cub imagenetr cars aircraft food101 sun objectnet
```

Large archives use byte-range downloads with persistent progress records.
`data/<dataset>.prepared.json` records the class and image counts after validation
with the project's data loader and image decoding checks for each class.

### 3. Usage Example

```bash
python main.py --config configs/cifar_0_10.json \
               --data-root /path/to/data \
               --output outputs/cifar_0_10
```

## Experiment tracking (W&B)

Training logs to the W&B project `hypcli` by default. Each invocation creates
one run, grouped by dataset and tagged with the incremental protocol, so the
18 main experiments remain separate. Local W&B files are stored inside the
experiment's output directory.

Authenticate once with `wandb login`, or provide `WANDB_API_KEY` in the process
environment. Keep credentials outside the repository. Then run:

```bash
conda activate cl
python main.py --config configs/cifar_0_10.json --data-root ./data \
  --device cuda:0 --wandb-project hypcli
```

Recorded metrics include `train/loss`, `train/accuracy_uncalibrated`, and the
learning rate against global epoch; `task_01/loss`, etc. against each task's
own epoch; and `eval/accuracy`, `eval/top5`, old/new class accuracy, and running
average accuracy against task number. Epoch accuracy uses the distance
classifier; task-end evaluation includes prototype calibration. Accuracy values
are percentages. Run summaries contain final and average accuracy, the task
accuracy curve, completion status, and elapsed time.

Use `--run-name` to name a run, `--wandb-entity` to choose an account/team,
`--wandb-mode offline` to retain tracking locally, or `--wandb-mode disabled`
to disable tracking. Offline runs can later be uploaded with `wandb sync`
using the directory printed by W&B.

## Citation

If you find our work useful for your research, please star our project and cite our work.

```bibtex
@inproceedings{cheng2026hypcl,
  title     = {{H}yp{CL}: Adapting {CLIP} in Hyperbolic Space for Continual Learning},
  author    = {Quan Cheng and Hao Yu and Da-Wei Zhou and Lijun Zhang},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  pages     = {19099--19111},
  year      = {2026}
}
```


## Acknowledgements

This project builds upon excellent open-source work:

- [C3Box](https://github.com/LAMDA-CL/C3Box)
- [ENGINE](https://github.com/RenaissCode/ENGINE)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)


## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
