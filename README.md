# FFG-benchmarks

This repository provides an unified frameworks to train and test the state-of-the-art few-shot font generation (FFG) models.

## What is Few-shot Font Generation (FFG)?

Few-shot font generation tasks aim to generate a new font library using only a few reference glyphs, e.g., less than 10 glyph images, without additional model fine-tuning at the test time [[ref]](https://arxiv.org/abs/2104.00887).

In this repository, we do not consider methods fine-tuning on the unseen style fonts.

## Sub-documents

```
docs
├── Dataset.md
├── FTransGAN-Dataset.md
├── Inference.md
├── Evaluator.md
└── models
    ├── DM-Font.md
    ├── FUNIT.md
    ├── LF-Font.md
    └── MX-Font.md
```

## Available models

- FUNIT (Liu, Ming-Yu, et al. ICCV 2019) [[pdf]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Few-Shot_Unsupervised_Image-to-Image_Translation_ICCV_2019_paper.pdf) [[github]](https://github.com/NVlabs/FUNIT): not originally proposed for FFG tasks, but we modify the unpaired i2i framework to the paired i2i framework for FFG tasks.
- DM-Font (Cha, Junbum, et al. ECCV 2020) [[pdf]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640715.pdf) [[github]](https://github.com/clovaai/dmfont): proposed for complete compositional scripts (e.g., Korean). If you want to test DM-Font in Chinese generation tasks, you have to modify the code (or use other models).
- LF-Font (Park, Song, et al. AAAI 2021) [[pdf]](https://www.aaai.org/AAAI21Papers/AAAI-1379.ParkS.pdf) [[github]](https://github.com/clovaai/lffont/): originally proposed to solve the drawback of DM-Font, but it still require component labels for generation. Our implementation allows to generate characters with unseen component.
- MX-Font (Park, Song, et al. ICCV 2021) [[pdf]](https://arxiv.org/abs/2104.00887) [[github]](https://github.com/clovaai/mxfont): generating fonts by employing multiple experts where each expert focuses on different local concepts.

### Not available here, but you may also consider

- EMD (Zhang, Yexun, et al. CVPR 2018) [[pdf]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Separating_Style_and_CVPR_2018_paper.pdf) [[github]](https://github.com/zhyxun/Separating-Style-and-Content-for-Generalized-Style-Transfer)
- AGIS-Net (Yue, Gao, et al. SIGGRAPH Asia 2019) [[pdf]](https://arxiv.org/abs/1910.04987) [[github]](https://github.com/hologerry/AGIS-Net)
- FTransGAN (Li, Chenhao, et al. WACV 2021) [[pdf]](https://openaccess.thecvf.com/content/WACV2021/papers/Li_Few-Shot_Font_Style_Transfer_Between_Different_Languages_WACV_2021_paper.pdf) [[github]](https://github.com/ligoudaner377/font_translator_gan)

### Model overview

| Model                       | Provided in this repo? | Chinese generation? | Need component labels? |
|-----------------------------|------------------------|---------------------|------------------------|
| EMD (CVPR'18)               | X                      | O                   | X                      |
| FUNIT (ICCV'19)             | O                      | O                   | X                      |
| AGIS-Net (SIGGRAPH Asia'19) | X                      | O                   | X                      |
| DM-Font (ECCV'20)           | O                      | X                   | O                      |
| LF-Font (AAAI'21)           | O                      | O                   | O                      |
| FTransGAN (WACV'21)         | X                      | O                   | X                      |
| MX-Font (ICCV'21)           | O                      | O                   | Only for training      |

## Preparing Environments

### Requirements

Our code is tested on `Python >= 3.6` (we recommend [conda](https://docs.anaconda.com/anaconda/install/linux/)) with the following libraries

```
torch >= 1.5
sconf
numpy
scipy
scikit-image
tqdm
jsonlib-python3
fonttools
```

### Datasets

Korean / Chinese / ...


The full description is in [docs/Dataset.md](docs/Dataset.md)

We allow two formats for datasets:

- TTF: We allow using the native true-type font (TTF) formats for datasets. It is storage-efficient and easy-to-use, particularly if you want to build your own dataset.
- Images: We also allow rendered images for datasets, similar to [ImageFoler](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder) (but a modified version). It is convenient when you want to generate a full font library from the un-digitalized characters (e.g., handwritings).

You can collect your own fonts from the following web sites (for non-commercial purpose):

- [https://www.foundertype.com/index.php/FindFont/index](https://www.foundertype.com/index.php/FindFont/index) (acknowledgement: [DG-Font](https://github.com/ecnuycxie/DG-Font) refers this web site)
- [https://chinesefontdesign.com/](https://chinesefontdesign.com/)
- Any other web sites providing non-commercial fonts

Note that fonts are protected intellectual property and it is unable to release the collected font datasets unless license is cleaned-up. Many font generation papers do not publicly release their own datasets due to this license issue. We also face the same issue here. Therefore, we encourage the users to collect their own datasets from the web, or using the publicly avaiable datasets.

FTransGAN (Li, Chenhao, et al. WACV 2021) [[pdf]](https://openaccess.thecvf.com/content/WACV2021/papers/Li_Few-Shot_Font_Style_Transfer_Between_Different_Languages_WACV_2021_paper.pdf) [[github]](https://github.com/ligoudaner377/font_translator_gan) released the rendered image files for training and evaluating FFG models. We also make our repository able to use the font dataset provided by FTransGAN. More details can be found in [docs/FTransGAN-Dataset.md](docs/FTransGAN-Dataset.md).

## Training

We separately provide model documents in [docs/models](docs/models) as follows

- [docs/models/FUNIT.md](docs/models/FUNIT.md)
- [docs/models/DM-Font.md](docs/models/DM-Font.md)
- [docs/models/LF-Font.md](docs/models/LF-Font.md)
- [docs/models/MX-Font.md](docs/models/MX-Font.md)


## Generation

### Preparing reference images

Detailed instruction for preparing reference images is decribed in [here](docs/Reference.md).
    
### Run test

Please refer following documents to train the model:

* DM-Font: [docs/models/DM-Font.md](docs/models/DM-Font.md)
* LF-Font(Phase 1, 2): [docs/models/LF-Font.md](docs/models/LF-Font.md)
* MX-Font: [docs/models/MX-Font.md](docs/models/MX-Font.md)
* FUNIT(modified for fonts): [docs/models/FUNIT.md](docs/models/FUNIT.md)


## Evaluation

Detailed instructions for preparing evaluator and testing the generated images are decribed in [here](docs/Evaluation.md).

## License

This project is distributed under [MIT license](LICENSE), except [FUNIT](FUNIT) and [base/modules/modules.py](base/modules/modules.py) which is adopted from https://github.com/NVlabs/FUNIT.

```
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
