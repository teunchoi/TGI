<div align="center">

# Anchoring and Rescaling Attention for Semantically Coherent Inbetweening

<h3>CVPR 2026</h3>

<h3>
Tae Eun Choi*&nbsp;&nbsp;Sumin Shim*&nbsp;&nbsp;Junhyeok Kim&nbsp;&nbsp;Seong Jae Hwang
</h3>

<h3>Yonsei University</h3>

<br>

[![Project Website](https://img.shields.io/badge/Project-Page-blue)](https://teunchoi.github.io/TGI-project-page/) 
[![arXiv](https://img.shields.io/badge/arXiv-2603.17651-b31b1b)](https://arxiv.org/abs/2603.17651) 
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-ffcc4d)](https://huggingface.co/datasets/use08174/TGI-Benchmark)

</div>

<br>
<br>
<p align="center">
  <img src="figure.png" width="100%">
</p>

## Overview

This repository contains the official implementation of **Anchoring and Rescaling Attention for Semantically Coherent Inbetweening**, a training-free approach for **text-conditioned generative inbetweening** that improves semantic fidelity, frame consistency, and pace stability. Given the **first frame**, **last frame**, and a **text prompt**, our method generates semantically coherent intermediate frames while enhancing semantic alignment, temporal consistency, and motion pacing without additional model training. We also introduce **TGI-Bench**, a benchmark for evaluating text-conditioned generative inbetweening across diverse sequence lengths and motion scenarios.

___

## Dataset

The **TGI-Bench** dataset is available on Hugging Face [here](https://huggingface.co/datasets/use08174/TGI-Benchmark).

## Installation

We recommend using a **conda** environment.

### Create environment

Python **3.10 or higher** is required.

```bash
conda create -n tgi python=3.10
conda activate tgi
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Once this is done, the environment setup is complete.


## Run Inference

To run inference with the default settings:

```bash
python inference.py
```


## Optional Arguments

You can customize inference with additional arguments:

```bash
python inference.py \
  --prompt "A freight train moves forward through heavy falling snow." \
  --img_first example/first.jpg \
  --img_last example/last.jpg \
  --seed 0 \
  --num_frames 81 \
  --w_edge 8 \
  --s_edge 1.06 \
  --s_mid 0.94 \
  --beta_end 0.7 \
  --beta_mid 0.3
```

### Argument description

- `--prompt`: text prompt
- `--img_first`: path to the first frame
- `--img_last`: path to the last frame
- `--seed`: random seed
- `--num_frames`: number of frames (`25`, `33`, `65`, `81`)
- `--w_edge`: width of the fast region near both ends
- `--s_edge`: scaling parameter near keyframes
- `--s_mid`: scaling parameter for middle frames
- `--beta_end`: endpoint weighting parameter
- `--beta_mid`: middle-region weighting parameter

If not specified, default example values are used.

