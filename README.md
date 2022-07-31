<div align="center">

<h1><img src="logo.png" height="28px" /> BMInf </h1>

**Efficient Inference for Big Models**

</div>

<p align="center">
  <a href="#overview">Overview</a> • <a href="#install">Installation</a> • <a href="#quick-start">Quick Start</a> • <a href="./README-ZH.md" target="_blank">简体中文</a>
<br>
</p>

<p align="center">
	<a href='https://bminf.readthedocs.io/en/latest/'>
	    <img src='https://readthedocs.org/projects/bminf/badge/?version=latest' alt='doc' />
	</a>
	<a href="https://github.com/OpenBMB/BMInf/blob/main/LICENSE">
	    <img alt="github" src="https://img.shields.io/github/license/OpenBMB/BMInf">
	</a>
	<a>
		 <img alt="version" src="https://img.shields.io/badge/version-1.0.0-blue">
	</a>
</p>    


## What's New
- 2022/07/31 (**BMInf 2.0.0**) BMInf can now be applied to any transformer-based model.
- 2021/12/21 (**BMInf 1.0.0**) Now the package no more depends on ``cupy`` and supports PyTorch backpropagation.
- 2021/10/18 We updated the ``generate`` interface and added a new CPM 2.1 demo.
- 2021/09/24 We publicly released BMInf on the 2021 Zhongguancun Forum (AI and Multidisciplinary Synergy Innovation Forum).

**Note:** README for `BMInf-1` can be found in `old_docs` directory. Examples of CPM-1/2 and EVA will be published soon.

<div id="overview"></div>

## Overview

BMInf (Big Model Inference) is a low-resource inference package for large-scale pretrained language models (PLMs). 

BMInf supports running models with more than 10 billion parameters on a single NVIDIA GTX 1060 GPU in its minimum requirements. Running with better GPUs leads to better performance. In cases where the GPU memory supports the large model inference (such as V100 or A100), BMInf still has a significant performance improvement over the existing PyTorch implementation.

<div id="install"></div>

## Installation

- From pip: ``pip install bminf``

- From source code: download the package and run ``python setup.py install``


### Hardware Requirement

Here we list the minimum and recommended configurations for running BMInf. 

| | Minimum Configuration | Recommended Configuration |
|-|-|-|
| Memory | 16GB | 24GB
| GPU | NVIDIA GeForce GTX 1060 6GB | NVIDIA Tesla V100 16GB
| PCI-E |  PCI-E 3.0 x16 |  PCI-E 3.0 x16

GPUs with compute
capability 6.1 or higher are supported by BMInf. Refer to the [table](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) to check whether your GPU is supported. 

### Software Requirement

BMInf requires CUDA version >= 10.1 and all the dependencies can be automaticlly installed by the installation process.

- **python** >= 3.6
- **torch** >= 1.7.1
- **cpm_kernels** >= 1.0.9

<div id="quick-start"></div>

## Quick Start

Use `bminf.wrapper` to automatically convert your model.

```python
import bminf

# initialize your model on CPU
model = MyModel()

# load state_dict before using wrapper
model.load_state_dict(model_checkpoint)

# apply wrapper
with torch.cuda.device(CUDA_DEVICE_INDEX):
    model = bminf.wrapper(model)
```

If `bminf.wrapper` does not fit your model well, you can use the following method to replace it manually.

* Replace `torch.nn.ModuleList` with `bminf.TransformerBlockList`.
```python
module_list = bminf.TransformerBlockList([
	# ...
], [CUDA_DEVICE_INDEX])
```

* Replace `torch.nn.Linear` with `bminf.QuantizedLinear`.
```python
linear = bminf.QuantizedLinear(torch.nn.Linear(...))
```

## Performances

Here we report the speeds of CPM2 encoder and decoder we have tested on different platforms. You can also run ``benchmark/cpm2/encoder.py`` and ``benchmark/cpm2/decoder.py`` to test the speed on your machine!

Implementation | GPU | Encoder Speed (tokens/s) | Decoder Speed (tokens/s) |
|-|-|-|-|
BMInf | NVIDIA GeForce GTX 1060 | 718 | 4.4
BMInf | NVIDIA GeForce GTX 1080Ti | 1200 | 12
BMInf | NVIDIA GeForce GTX 2080Ti | 2275 | 19
BMInf | NVIDIA Tesla V100 | 2966 | 20
BMInf | NVIDIA Tesla A100 | 4365 | 26
PyTorch | NVIDIA Tesla V100 | - | 3
PyTorch | NVIDIA Tesla A100 | - | 7

## Community
We welcome everyone to contribute codes following our [contributing guidelines](https://github.com/OpenBMB/BMInf/blob/master/CONTRIBUTING.md).

You can also find us on other platforms:
- QQ Group: 735930538
- WeChat Official Account: OpenBMB
- Website: https://www.openbmb.org
- Weibo: http://weibo.cn/OpenBMB
- Twitter: https://twitter.com/OpenBMB

## License

The package is released under the [Apache 2.0](https://github.com/OpenBMB/BMInf/blob/master/LICENSE) License.

## References
<div id="ref"></div>

1. [CPM-2: Large-scale Cost-efficient Pre-trained Language Models.](https://arxiv.org/abs/2106.10715) Zhengyan Zhang, Yuxian Gu, Xu Han, Shengqi Chen, Chaojun Xiao, Zhenbo Sun, Yuan Yao, Fanchao Qi, Jian Guan, Pei Ke, Yanzheng Cai, Guoyang Zeng, Zhixing Tan, Zhiyuan Liu, Minlie Huang, Wentao Han, Yang Liu, Xiaoyan Zhu, Maosong Sun.
2. [CPM: A Large-scale Generative Chinese Pre-trained Language Model.](https://arxiv.org/abs/2012.00413) Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.
3. [EVA: An Open-Domain Chinese Dialogue System with Large-Scale Generative Pre-Training.](https://arxiv.org/abs/2108.01547) Hao Zhou, Pei Ke, Zheng Zhang, Yuxian Gu, Yinhe Zheng, Chujie Zheng, Yida Wang, Chen Henry Wu, Hao Sun, Xiaocong Yang, Bosi Wen, Xiaoyan Zhu, Minlie Huang, Jie Tang.
4. [Language Models are Unsupervised Multitask Learners.](http://www.persagen.com/files/misc/radford2019language.pdf) Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever.
