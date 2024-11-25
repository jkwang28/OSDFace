# OSDFace: One-Step Diffusion Model for Face Restoration

[Jingkai Wang](https://github.com/jkwang28), [Jue Gong](https://github.com/gobunu), [Lin Zhang](https://github.com/wanliyungui), [Zheng Chen](https://zhengchen1999.github.io/), Xing Liu, Hong Gu, [Yutong Liu](https://isabelleliu630.github.io/), [Yulun Zhang](http://yulunzhang.com/), and [Xiaokang Yang](https://scholar.google.com/citations?user=yDEavdMAAAAJ), "One-Step Diffusion Model for Face Restoration", arXiv, 2024

[arXiv] [[supplementary material](https://github.com/jkwang28/OSDFace/releases/tag/v1)] [visual results] [pretrained models]

#### 🔥🔥🔥 News

- **2024-11-25:** This repo is released.

---

> **Abstract:** Diffusion models have demonstrated impressive performance in face restoration. Yet, their multi-step inference process remains computationally intensive, limiting their applicability in real-world scenarios. Moreover, existing methods often struggle to generate face images that are harmonious, realistic, and consistent with the subject’s identity. In this work, we propose OSDFace, a novel one-step diffusion model for face restoration. Specifically, we propose a visual representation embedder (VRE) to better capture prior information and understand the input face. In VRE, low-quality faces are processed by a visual tokenizer and subsequently embedded with a vector-quantized dictionary to generate visual prompts. Additionally, we incorporate a facial identity loss derived from face recognition to further ensure identity consistency. We further employ a generative adversarial network (GAN) as a guidance model to encourage distribution alignment between the restored face and the ground truth. Experimental results demonstrate that OSDFace surpasses current state-of-the-art (SOTA) methods in both visual quality and quantitative metrics, generating high-fidelity, natural face images with high identity consistency.

![](images/overall-osdface.png)

---

![vis-main-top](images/vis-main-top.png)

[<img src="figs/C1.png" height="216"/>](https://imgsli.com/MjIyMjk5) [<img src="figs/C2.png" height="216"/>](https://imgsli.com/MjIyMzAy) [<img src="figs/C3.png" height="216"/>](https://imgsli.com/MjIyMzEw) [<img src="figs/C4.png" height="216"/>](https://imgsli.com/MjIyMzA4)

[<img src="figs/C5.png" height="215"/>](https://imgsli.com/MjIyMzA0) [<img src="figs/C6.png" height="215"/>](https://imgsli.com/MjIyMzAw) [<img src="figs/C7.png" height="215"/>](https://imgsli.com/MjIyMjk3)

---

## ⚒️ TODO

* [ ] Release code and pretrained models

## 🔗 Contents

- [ ] Datasets
- [ ] Models
- [ ] Testing
- [ ] Training
- [x] [Results](#Results)
- [x] [Citation](#Citation)
- [ ] [Acknowledgements](#Acknowledgements)

## <a name="results"></a>🔎 Results

We achieved state-of-the-art performance on synthetic and real-world blur dataset. Detailed results can be found in the paper.

<details>
<summary>Quantitative Comparisons (click to expand)</summary>
<li> Results in Table 1 on synthetic dataset (CelebA-Test) from main paper. 
<p align="center">
<img src="images/quan-celeba.png" >
</p>
</li>
<li> Results in Table 2 on real-world datasets (Wider-Test, LFW-Test, WebPhoto-Test) from main paper. 
<p align="center">
<img src="images/quan-realworld.png" >
</p>
</li>
</details>
<details open>
<summary>Visual Comparisons (click to expand)</summary>
<li> Results in Figure 5 on synthetic dataset (CelebA-Test) from main paper.
<p align="center">
<img src="images/vis-main-celeba.png" >
</p>
</li>
<li> Results in Figure 6 on real-world dataset (Wider-Test, LFW-Test, WebPhoto-Test) from main paper.
<p align="center">
<img src="images/vis-main-realworld.png" >
</p>
</li>
</details>
<details>
<summary  style="margin-left: 2rem;">More Comparisons on Synthetic Dataset...</summary>
<li style="margin-left: 2rem;">Results in Figure 4, 5, 6 on synthetic dataset (CelebA-Test) from supplemental material.
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
<summary  style="margin-left: 2rem;">More Comparisons on Real-World Dataset...</summary>
<li style="margin-left: 2rem;">Results in Figure 7, 8, 9, 10 on real-world datasets (Wider-Test, LFW-Test, WebPhoto-Test) from supplemental material.
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

If you find the code helpful in your resarch or work, please cite the following paper(s).

```
@article{wang2024osdface,
    title={One-Step Diffusion Model for Face Restoration},
    author={Wang, Jingkai and Gong, Jue and Zhang, Lin and Chen, Zheng and Liu, Xing and Gu, Hong and Liu, Yutong and Zhang, Yulun and Yang, Xiaokang},
    journal={arXiv preprint arXiv:2409.08384},
    year={2024}
}
```

## <a name="acknowledgements"></a>💡 Acknowledgements

[TBD]