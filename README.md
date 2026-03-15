# Anchoring and Rescaling Attention for Semantically Coherent Inbetweening

[![Project Page](https://img.shields.io/badge/Project-Coming_Soon-blue)](#)
[![Paper](https://img.shields.io/badge/Paper-CVPR_2026-green)](#)
[![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b)](#)

> Training-free generative inbetweening with improved semantic fidelity, frame consistency, and pace stability.

## Overview

This repository contains the official project page for:

**Anchoring and Rescaling Attention for Semantically Coherent Inbetweening**  
*Tae Eun Choi, Sumin Shim, Junhyeok Kim, Seong Jae Hwang*  
Yonsei University

We propose a **training-free** approach for **text-conditioned generative inbetweening (GI)**, where intermediate frames are generated from the **first frame**, **last frame**, and a **text prompt**.

Our method addresses three key challenges in generative inbetweening:

- **Semantic fidelity**: generated frames should follow the text prompt accurately
- **Frame consistency**: object identity and scene structure should remain stable across time
- **Pace stability**: motion should progress smoothly without abrupt accelerations or stalls

To achieve this, we introduce:

- **Keyframe-anchored Attention Bias (KAB)**  
  Guides intermediate frames using semantic and temporal cues derived from the first/last keyframes and text.

- **Rescaled Temporal RoPE (ReTRo)**  
  Rescales temporal positional encoding to better preserve keyframes while improving overall temporal consistency.

- **TGI-Bench**  
  A new benchmark for **text-conditioned generative inbetweening**, designed to evaluate GI models across multiple sequence lengths and challenge categories.

---

## Method

### 1. Keyframe-anchored Attention Bias (KAB)

KAB extracts anchor signals from the model's own cross-attention maps of the first and last keyframes, then interpolates them across time to guide intermediate frames.

This helps the model:

- better follow the intended motion path
- maintain stronger alignment with the text prompt
- produce more stable motion pacing

### 2. Rescaled Temporal RoPE (ReTRo)

ReTRo modifies temporal RoPE scaling inside self-attention:

- **higher scale near keyframes** to preserve endpoint fidelity
- **lower scale in the middle frames** to widen temporal attention and improve consistency

This simple adjustment reduces blur and artifacts while stabilizing the whole sequence.

---

## TGI-Bench

We introduce **TGI-Bench**, the first benchmark specifically designed for **text-conditioned generative inbetweening**.

### Features

- multiple sequence lengths: **25 / 33 / 65 / 81 frames**
- text descriptions paired with keyframe sequences
- challenge-based evaluation across:
  - **dynamic motion**
  - **linear motion**
  - **occlusion**
  - **near-static**

TGI-Bench enables more fine-grained diagnosis of GI models beyond conventional video metrics.

---

## Main Results

Without additional training, our method achieves state-of-the-art performance on multiple evaluation axes, including:

- video generation quality
- semantic fidelity
- frame consistency
- pace stability

### 81-frame results

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD ↓ | VBench ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Wan | 17.63 | 0.6179 | 0.3945 | 82.90 | 0.2769 | 9.904 |
| **Ours** | **18.17** | **0.6269** | **0.3818** | **77.59** | **0.2458** | **10.022** |

### Human evaluation (81-frame)

| Method | Frame Consistency ↑ | Semantic Fidelity ↑ | Pace Stability ↑ |
|---|---:|---:|---:|
| Wan | 3.50 | 3.69 | 3.65 |
| **Ours** | **4.38** | **4.27** | **4.34** |

---

## Why it works

Existing GI models often struggle when:

- the two keyframes are far apart
- motion is large or non-linear
- the text prompt specifies subtle semantics such as direction or action style

Our method explicitly strengthens both:

- **cross-attention guidance** from keyframes and text
- **self-attention temporal structure** across the whole sequence

This leads to more coherent inbetweening under difficult scenarios such as **dynamic motion** and **occlusion**.

---

## Coming Soon

- [ ] Project page
- [ ] arXiv link
- [ ] Code release
- [ ] TGI-Bench release
- [ ] Demo videos
- [ ] BibTeX

---

## Citation

```bibtex
@inproceedings{choi2026anchoring,
  title={Anchoring and Rescaling Attention for Semantically Coherent Inbetweening},
  author={Choi, Tae Eun and Shim, Sumin and Kim, Junhyeok and Hwang, Seong Jae},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
