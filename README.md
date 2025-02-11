# AniLines - Anime Line Extractor

![Teaser](./assets/gifs/teaser.gif)
*©Oniichan wa Oshimai!*

⭐ **AniLines** is a simple tool to extract lineart from anime images. 

## Contents
- [Features](#features)
- [How to use](#how-to-use)
  - [Environment](#environment)
  - [Binarized output](#Binarization)
  - [Inference acceleration](#inference-acceleration)
- [License](#license)
- [Citation](#citation)
- [Related Works](#related-works)


## Updates
- [x] Init Repo - 2025/02/10
- [ ] Support for video

## Features
### What does AniLines do?

Like [🚀MangaLineExtraction](https://github.com/ljsabc/MangaLineExtraction_PyTorch) and [👓Anime2Sketch](https://github.com/Mukosame/Anime2Sketch), **AniLines** generates sketch from anime(🎞️cel animation🎞️ in particular) images, it generally provides more details and clearer lines with fewer artifacts on characters. 

| Input | [MangaLine](https://github.com/ljsabc/MangaLineExtraction_PyTorch) | [Anime2Sketch](https://github.com/Mukosame/Anime2Sketch) | DoG(Binary) | AniLines(Basic) | AniLines(Detail) |
| --- | --- | --- | --- | --- | --- |
| <img src="./assets/imgs/input/img1.jpg" width="100"/> | <img src="./assets/imgs/mangaline/img1.jpg" width="100"/> | <img src="./assets/imgs/anime2sketch/img1.jpg" width="100"/> | <img src="./assets/imgs/dog/img1.jpg" width="100"/> | <img src="./assets/imgs/anilines_basic/img1.jpg" width="100"/> | <img src="./assets/imgs/anilines_detail/img1.jpg" width="100"/> |
| <img src="./assets/imgs/input/img2.jpg" width="100"/> | <img src="./assets/imgs/mangaline/img2.jpg" width="100"/> | <img src="./assets/imgs/anime2sketch/img2.jpg" width="100"/> | <img src="./assets/imgs/dog/img2.jpg" width="100"/> | <img src="./assets/imgs/anilines_basic/img2.jpg" width="100"/> | <img src="./assets/imgs/anilines_detail/img2.jpg" width="100"/> |
| <img src="./assets/imgs/input/img4.jpg" width="100"/> | <img src="./assets/imgs/mangaline/img4.jpg" width="100"/> | <img src="./assets/imgs/anime2sketch/img4.jpg" width="100"/> | <img src="./assets/imgs/dog/img4.jpg" width="100"/> | <img src="./assets/imgs/anilines_basic/img4.jpg" width="100"/> | <img src="./assets/imgs/anilines_detail/img4.jpg" width="100"/> |
| <img src="./assets/imgs/input/img6.jpg" width="100"/> | <img src="./assets/imgs/mangaline/img6.jpg" width="100"/> | <img src="./assets/imgs/anime2sketch/img6.jpg" width="100"/> | <img src="./assets/imgs/dog/img6.jpg" width="100"/> | <img src="./assets/imgs/anilines_basic/img6.jpg" width="100"/> | <img src="./assets/imgs/anilines_detail/img6.jpg" width="100"/> |
| <img src="./assets/imgs/input/img7.jpg" width="100"/> | <img src="./assets/imgs/mangaline/img7.jpg" width="100"/> | <img src="./assets/imgs/anime2sketch/img7.jpg" width="100"/> | <img src="./assets/imgs/dog/img7.jpg" width="100"/> | <img src="./assets/imgs/anilines_basic/img7.jpg" width="100"/> | <img src="./assets/imgs/anilines_detail/img7.jpg" width="100"/> |
| <img src="./assets/imgs/input/img8.jpg" width="100"/> | <img src="./assets/imgs/mangaline/img8.jpg" width="100"/> | <img src="./assets/imgs/anime2sketch/img8.jpg" width="100"/> | <img src="./assets/imgs/dog/img8.jpg" width="100"/> | <img src="./assets/imgs/anilines_basic/img8.jpg" width="100"/> | <img src="./assets/imgs/anilines_detail/img8.jpg" width="100"/> |
| <img src="./assets/imgs/input/img9.jpg" width="100"/> | <img src="./assets/imgs/mangaline/img9.jpg" width="100"/> | <img src="./assets/imgs/anime2sketch/img9.jpg" width="100"/> | <img src="./assets/imgs/dog/img9.jpg" width="100"/> | <img src="./assets/imgs/anilines_basic/img9.jpg" width="100"/> | <img src="./assets/imgs/anilines_detail/img9.jpg" width="100"/> |


*©Urusei Yatsura 2022, ©Violet Evergarden, ©Miss Kobayashi's Dragon Maid, ©Little Witch Academia*

All input images shown above were inferenced at a 1080p resolution. It is also advised to use 1080p images for better performance.

### How does AniLines work?

**AniLines** has two modes: `basic` and `detail`. While `basic` model extracts the main structure of a drawing, `detail` model provides the sketch on more elements like backgrounds and cel edges. Personally speaking, `detail` mode has better performance as it has a higher 'recall' score, and consequently miss less lines.

| input | basic mode | detail mode |
| --- | --- | --- |
| ![Input](./assets/imgs/input/img3.jpg) | ![Basic](./assets/imgs/anilines_basic/img3.jpg) | ![Detail](./assets/imgs/anilines_detail/img3.jpg) |

*©Watashi ni Tenshi ga Maiorita!*


## How to use
### Environment

First, clone this repo, create a new conda environment and install the requirements:

```bash
git clone https://github.com/zhenglinpan/AniLines-Anime-Line-Extractor.git
cd AniLines-Anime-Line-Extractor

conda create -n anilines python=3.12 -y
conda activate anilines

pip install -r requirements.txt
```

Download the pre-trained models from the links below and put them in the `./weights` folder:
- [Basic model](https://drive.google.com/file/d/14Bp8mbQAbiR1rQrEsFp-uNdOou8hoCFr/view?usp=sharing)
- [Detail model](https://drive.google.com/file/d/12U1Mwlonoipk2Yvr12mNaFB30foy420o/view?usp=sharing)

and then run the following command:

```bash
python infer.py --dir_in ./input --dir_out ./output --mode detail --binarize -1 --fp16 True --device cuda:0
```

You can either pass a single image or a folder to `dir_in`, the extracted lineart will be saved to the `output` folder by default.

### Binarized output
Binarized lines are often used in animation production. AniLines provides an API for this feature. By default, binarization is disabled (set to `-1`). You can enable binarization by setting the `--binarize` parameter to any value between 0 and 1. This will adjust the threshold for binarizing the output.
| input | no binarize | binarize 0.5 | binarize 0.95 |
| --- | --- | --- | --- |
| ![Input](./assets/imgs/binarize/input.jpg) | ![No Binarize](./assets/imgs/binarize/no_binary.jpg) | ![Binarize 0.5](./assets/imgs/binarize/binary_50.jpg) | ![Binarize 0.95](./assets/imgs/binarize/binary_95.jpg) |

*©RainbowSea*

### Inference acceleration
By default, AniLines uses `fp16` to accelerate inference, trading marginal performance for a considerable speed boost. You can set `--fp16` to `False` if your GPU does not support it or if you prefer full precision. When using `fp16`, the model runs up to `5 times faster`, as tested with 1080p images.

For further accleration, you can play with **TensorRT**/**TorchTRT** with **onnx** to achieve its maximum potential speed.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
Feel free to cite this work if you find it useful for your research:

```bibtex
@misc{AniLines,
  author = {Zhenglin Pan},
  title = {AniLines - Anime Line Extractor},
  publisher = {GitHub},
  year = {2025},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zhenglinpan/AniLines-Anime-Line-Extractor}}
}
