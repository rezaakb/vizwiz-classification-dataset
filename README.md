# VizWiz-Classification

![VizWiz-Classification Cover Image](http://drive.google.com/uc?export=view&id=17T2WF2uT2r_MfTEzub_zfU5IoSAtS2ne)

## Introduction

VizWiz-Classification is a new test set using 8,900 images taken by people who are blind for which we collected metadata to indicate the presence versus absence of 200 ImageNet object categories. Please read our paper to learn more:

[A New Dataset Based on Images Taken by Blind People for Testing the Robustness of Image Classification Models Trained for ImageNet Categories.
Reza Akbarian Bafghi, and Danna Gurari. arXiv, 2023.](#)


## Dataset download

Before you can use our API, you must download the dataset. Links to download the images and annotations are available [here](#).

Alternatively, you can run the following series of commands. This will create a directory `dataset` with all the files in your current directory.
```
$ mkdir -p dataset/images dataset/annotations
$ cd dataset
$ wget X/test.zip \
       X/annotations.zip
$ unzip -o test.zip -d images
$ unzip -o annotations.zip -d annotations
$ rm test.zip annotations.zip
```

## Installation

Please run the following command to install the library for testing pre-trained models from [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) on our dataset.
```
$ pip install timm
```

## Evaluation
After downloading the dataset, you can use `eval.py` code to evaluate your model. Please make sure that you set `IMAGE_PATH` (e.g. `dataset/images`), `ANN_PATH` (e.g. `dataset/annotations`), and `MODEL_NAME` (e.g. `vgg19`). 

```
$ python eval.py -m MODEL_NAME -i IMAGE_PATH -a ANN_PATH
```

## Citation

If you make use of our dataset for your research, please be sure to cite our work with the following BibTeX citation.
```
...
```
