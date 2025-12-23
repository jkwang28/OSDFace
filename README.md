# OSDFace: One-Step Diffusion Model for Face Restoration
[[Jingkai Wang](https://github.com/jkwang28), [Jue Gong](https://github.com/gobunu), [Lin Zhang](https://github.com/wanliyungui), [Zheng Chen](https://zhengchen1999.github.io/), Xing Liu, Hong Gu, [Yutong Liu](https://isabelleliu630.github.io/), [Yulun Zhang](http://yulunzhang.com/), and [Xiaokang Yang](https://scholar.google.com/citations?user=yDEavdMAAAAJ), "One-Step Diffusion Model for Face Restoration", CVPR, 2025

[![page](https://img.shields.io/badge/Project-Page-blue?logo=github&logoSvg)](https://jkwang28.github.io/OSDFace-web/)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&logoSvg)](https://arxiv.org/abs/2411.17163)
[![supp](https://img.shields.io/badge/Supplementary_material-Paper-orange.svg)](https://github.com/jkwang28/OSDFace/releases/download/v2/supp.pdf)
[![releases](https://img.shields.io/github/downloads/jkwang28/OSDFace/total.svg)](https://github.com/jkwang28/OSDFace/releases)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=jkwang28.OSDFace&right_color=violet)](https://github.com/jkwang28/OSDFace)
[![GitHub Stars](https://img.shields.io/github/stars/jkwang28/OSDFace?style=social)](https://github.com/jkwang28/OSDFace)

#### 🔥🔥🔥 News

- **2025-12-23:** Inference code and pretrained models are released. 
- **2025-04-23:** Results are released. (Synthetic dataset: CelebA-Test; Real-world datasets: Wider-Test, LFW-Test, and WebPhoto-Test)
- **2025-02-27:** Congratulations! OSDFace is accepted to CVPR 2025.
- **2024-11-25:** This repo is released.


---

> **Abstract:** Diffusion models have demonstrated impressive performance in face restoration. Yet, their multi-step inference process remains computationally intensive, limiting their applicability in real-world scenarios. Moreover, existing methods often struggle to generate face images that are harmonious, realistic, and consistent with the subject’s identity. In this work, we propose OSDFace, a novel one-step diffusion model for face restoration. Specifically, we propose a visual representation embedder (VRE) to better capture prior information and understand the input face. In VRE, low-quality faces are processed by a visual tokenizer and subsequently embedded with a vector-quantized dictionary to generate visual prompts. Additionally, we incorporate a facial identity loss derived from face recognition to further ensure identity consistency. We further employ a generative adversarial network (GAN) as a guidance model to encourage distribution alignment between the restored face and the ground truth. Experimental results demonstrate that OSDFace surpasses current state-of-the-art (SOTA) methods in both visual quality and quantitative metrics, generating high-fidelity, natural face images with high identity consistency.

![](images/overall-osdface.png)

---

<!-- ![vis-main-top](images/vis-main-top.png) -->

[<img src="images/CAT-0012.png" height="200"/>](https://imgsli.com/MzIxNTU3) [<img src="images/CAT-0051.png" height="200"/>](https://imgsli.com/MzIxNTU5) [<img src="images/CAT-0054.png" height="200"/>](https://imgsli.com/MzIxNTYw) [<img src="images/CAT-1093.png" height="200"/>](https://imgsli.com/MzIxNTYy)

[<img src="images/LFW_Abdullatif_Sener.png" height="200"/>](https://imgsli.com/MzIxNTYz) [<img src="images/WebPhoto_0101.png" height="200"/>](https://imgsli.com/MzIxNTY4) [<img src="images/WT_0011.png" height="200"/>](https://imgsli.com/MzIxNTY5) [<img src="images/WT_0000.png" height="200"/>](https://imgsli.com/MzIxNTcz)

---

## ⚒️ TODO

* [x] Release inference code and pretrained models
* [ ] Release training code

## 🔗 Contents

- [x] [Environment](#datasets)
- [x] [Datasets](#datasets)
- [x] [Models](https://sjtueducn-my.sharepoint.com/:f:/g/personal/jingkaiwang_sjtu_edu_cn/EhIFpGRj6GxLoHG54TyeMT0Bg0wkiDHCcZ4B674k9veCiA?e=zQYHze)
- [x] [Inference](#Inference)
- [ ] Training
- [x] [Results](#results)
- [x] [Citation](#Citation)
- [ ] [Acknowledgements](#Acknowledgements)

## <a name="environment"></a>⚒️ Environment

1. Create conda environment with Python 3.10. (`conda create -n osdface python=3.10`)
2. Install [PyTorch 2.4.0](https://pytorch.org/get-started/previous-versions/#v240). (`pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121`)
3. Install `requirements.txt`. (`pip install -r requirements.txt`)

## <a name="datasets"></a>📊 Datasets

**Training Dataset:**

We use [FFHQ](https://github.com/NVlabs/ffhq-dataset) and [FFHQR](https://github.com/skylab-tech/ffhqr-dataset) for training. All of them are resized to 512x512. 

**Testing Dataset:**

We use CelebA-Test as the synthetic dataset. The download link can refer to [DAEFR](https://github.com/LIAGM/DAEFR). 

We use Wider-Test, LFW-Test, and WebPhoto-Test as the real-world datasets. The download links can refer to [DAEFR](https://github.com/LIAGM/DAEFR) and [VQFR](https://github.com/TencentARC/VQFR). 

Thanks a lot for their contribution to Face Restoration. 

## <a name="inference"></a>🧮 Inference

1. Download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1Nci6KufB8t2Uj-6tobrw3S7kkfQUTLHV?usp=sharing). Put models in `pretrained/`, and put the test data in `data/`
2. Run the inference bash `test.sh`. 

## 💬 Tips 

Since the offcial Stable Diffusion 2.1 is not avaliable, you could visit [ModelScope](https://modelscope.cn/models/stabilityai/stable-diffusion-2-1-base) to download it and use it locally. 

If you meet an error from `diffusers`: 
```sh
File "... lib/python3.10/site-packages/diffusers/utils/dynamic_modules_utils.py", line 28, in <module>
    from huggingface_hub import cached_download, hf_hub_download, model_info
```

you can edit the `dynamic_modules_utils.py` like: 

1. Replace line 24 from `from huggingface_hub import cached_download, hf_hub_download, model_info` to `from huggingface_hub import hf_hub_download, model_info`. 
2. Replace line 287 from `resolved_module_file = cached_download(` to `resolved_module_file = hf_hub_download(`. 

## <a name="results"></a>🔎 Results

We achieved state-of-the-art performance on synthetic and real-world datasets. Detailed results can be found in the paper, and the full results could be downloaded here: [Google Drive](https://drive.google.com/file/d/1BQkh6mLGtmSSk0DvH_fpX2xx99UoSIcR/view?usp=sharing) or [OneDrive](https://sjtueducn-my.sharepoint.com/:u:/g/personal/jingkaiwang_sjtu_edu_cn/EWxudCoxhSVGng5FRawnzdYBhzROUfxCgBa2chUZFYfYkQ?e=yISwFp). 

<details>
<summary>&ensp;Quantitative Comparisons (click to expand) </summary>
<li> Results in Table 1 on synthetic dataset (CelebA-Test) from the main paper. 
<p align="center">
<img src="images/quan-celeba.png" >
</p>
</li>
<li> Results in Table 2 on real-world datasets (Wider-Test, LFW-Test, WebPhoto-Test) from the main paper. 
<p align="center">
<img src="images/quan-realworld.png" >
</p>
</li>
</details>
<details open>
<summary>&ensp;Visual Comparisons (click to expand) </summary>
<li> Results in Figure 5 on synthetic dataset (CelebA-Test) from the main paper.
<p align="center">
<img src="images/vis-main-celeba.png" >
</p>
</li>
<li> Results in Figure 6 on real-world dataset (Wider-Test, LFW-Test, WebPhoto-Test) from the main paper.
<p align="center">
<img src="images/vis-main-realworld.png" >
</p>
</li>
</details>
<details>
<summary style="margin-left: 2rem;">&ensp;More Comparisons on Synthetic Dataset... </summary>
<li style="margin-left: 2rem;"> Results in Figure 4, 5, 6 on synthetic dataset (CelebA-Test) from supplemental material.
<p align="center">
<img src="images/vis-supp-celeba-1.png" >
</p>
<p align="center">
<img src="images/vis-supp-celeba-2.png" >
</p>
<p align="center">
<img src="images/vis-supp-celeba-3.png" >
</p>
</li>
</details>
<details>
<summary style="margin-left: 2rem;">&ensp;More Comparisons on Real-World Dataset... </summary>
<li style="margin-left: 2rem;"> Results in Figure 7, 8, 9, 10 on real-world datasets (Wider-Test, LFW-Test, WebPhoto-Test) from supplemental material.
<p align="center">
<img src="images/vis-supp-wider.png" >
</p>
<p align="center">
<img src="images/vis-supp-lfw.png" >
</p>
<p align="center">
<img src="images/vis-supp-webphoto.png" >
</p>
<p align="center">
<img src="images/vis-supp-realworld.png" >
</p>
</li>
</details>

## <a name="citation"></a>📎 Citation

If you find the code helpful in your research or work, please cite the following paper(s).

```
@InProceedings{wang2025osdface,
    author    = {Wang, Jingkai and Gong, Jue and Zhang, Lin and Chen, Zheng and Liu, Xing and Gu, Hong and Liu, Yutong and Zhang, Yulun and Yang, Xiaokang},
    title     = {{OSDFace}: One-Step Diffusion Model for Face Restoration},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {12626-12636}
}
```

## <a name="acknowledgements"></a>💡 Acknowledgements

[TBD]

<a href="https://www.star-history.com/#jkwang28/OSDFace&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jkwang28/OSDFace&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jkwang28/OSDFace&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jkwang28/OSDFace&type=date&legend=top-left" />
 </picture>
</a>
