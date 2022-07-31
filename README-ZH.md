<div align="center">

<h1><img src="logo.png" height="28px" /> BMInf </h1>

**大模型高效推理工具包**

</div>

<p align="center">
  <a href="#overview">总览</a> • <a href="#install">安装</a> • <a href="#quick-start">快速上手</a> • <a href="./README.md" target="_blank">English</a>
<br>
</p>

<p align="center">
	<a href='https://bminf.readthedocs.io/zh_CN/latest/'>
	    <img src='https://readthedocs.org/projects/bminf/badge/?version=latest' alt='doc' />
	</a>
	<a href="https://github.com/OpenBMB/BMInf/blob/main/LICENSE">
	    <img alt="github" src="https://img.shields.io/github/license/OpenBMB/BMInf">
	</a>
	<a>
		 <img alt="version" src="https://img.shields.io/badge/version-1.0.0-blue">
	</a>
</p>    


## 最新动态
- 2022/07/31 (**BMInf 2.0.0**) BMInf现在支持任意transformer类型的模型了。
- 2021/12/21 (**BMInf 1.0.0**) 现在工具包不再依赖``cupy``，新增了对于PyTorch反向传播的支持。
- 2021/10/18 更新了``generate``接口并且增加了一个CPM 2.1的新demo。
- 2021/09/24 BMInf于2021年中关村论坛-人工智能与多学科协同论坛正式发布了！

<div id="overview"></div>

## 总览

BMInf (Big Model Inference) 是一个用于大规模预训练语言模型（pretrained language models, PLM）推理阶段的低资源工具包。

BMInf最低支持在NVIDIA GTX 1060单卡运行百亿大模型。在此基础上，使用更好的gpu运行会有更好的性能。在显存支持进行大模型推理的情况下（如V100或A100显卡），BMInf的实现较现有PyTorch版本仍有较大性能提升。

<div id="install"></div>

## 安装

- 用pip安装：``pip install bminf``

- 从源代码安装: 下载工具包并在目录中运行 ``python setup.py install``

### 硬件要求

在下表中我们列出了运行BMInf的最低配置与推荐配置：

| | 最低配置 | 推荐配置 |
|-|-|-|
| 内存 | 16GB | 24GB
| GPU | NVIDIA GeForce GTX 1060 6GB | NVIDIA Tesla V100 16GB
| PCI-E |  PCI-E 3.0 x16 |  PCI-E 3.0 x16

BMInf支持计算能力6.1或更高的GPU，查看[对照表](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)来明确你的GPU是否被支持。

### 软件要求

BMInf需要安装CUDA 10.1及以上版本，所有的依赖包都会在安装过程中自动被安装。

- **python** >= 3.6
- **torch** >= 1.7.1
- **cpm_kernels** >= 1.0.9

<div id="quick-start"></div>

## 快速上手

使用`bminf.wrapper`来自动的转换你的模型。

```python
import bminf

# 在CPU上初始化你的模型
model = MyModel()

# 加载模型参数
model.load_state_dict(model_checkpoint)

# 使用bminf.wrapper
with torch.cuda.device(CUDA_DEVICE_INDEX):
    model = bminf.wrapper(model)
```

如果`bminf.wrapper`不能很好的适配你的模型，你可以用以下的方法来进行手动适配。

* 将 `torch.nn.ModuleList` 替换为 `bminf.TransformerBlockList`.
```python
module_list = bminf.TransformerBlockList([
	# ...
], [CUDA_DEVICE_INDEX])
```

* 将 `torch.nn.Linear` 替换为 `bminf.QuantizedLinear`.
```python
linear = bminf.QuantizedLinear(torch.nn.Linear(...))
```

<div id="supported-models"></div>

## 运行性能

下表汇报了我们在不同平台上运行CPM2编码器和解码器的速度。用户可以在本机运行``benchmark/cpm2/encoder.py``和``benchmark/cpm2/decoder.py``来测试本机运行速度。

实现 | GPU | 编码速度 (tokens/s) | 解码速度 (tokens/s) |
|-|-|-|-|
BMInf | NVIDIA GeForce GTX 1060 | 718 | 4.4
BMInf | NVIDIA GeForce GTX 1080Ti | 1200 | 12
BMInf | NVIDIA GeForce GTX 2080Ti | 2275 | 19
BMInf | NVIDIA Tesla V100 | 2966 | 20
BMInf | NVIDIA Tesla A100 | 4365 | 26
PyTorch | NVIDIA Tesla V100 | - | 3
PyTorch | NVIDIA Tesla A100 | - | 7

## 开源社区
欢迎贡献者参照我们的[贡献指南](https://github.com/OpenBMB/BMInf/blob/master/CONTRIBUTING.md)贡献相关代码。

您也可以在其他平台与我们沟通交流:
- QQ群: 735930538
- 微信公众号: OpenBMB
- 官方网站: https://www.openbmb.org
- 微博: http://weibo.cn/OpenBMB
- Twitter: https://twitter.com/OpenBMB

## 开源许可

该工具包使用[Apache 2.0](https://github.com/OpenBMB/BMInf/blob/master/LICENSE)开源许可证。

## 参考文献
<div id="ref"></div>

1. [CPM-2: Large-scale Cost-efficient Pre-trained Language Models.](https://arxiv.org/abs/2106.10715) Zhengyan Zhang, Yuxian Gu, Xu Han, Shengqi Chen, Chaojun Xiao, Zhenbo Sun, Yuan Yao, Fanchao Qi, Jian Guan, Pei Ke, Yanzheng Cai, Guoyang Zeng, Zhixing Tan, Zhiyuan Liu, Minlie Huang, Wentao Han, Yang Liu, Xiaoyan Zhu, Maosong Sun.
2. [CPM: A Large-scale Generative Chinese Pre-trained Language Model.](https://arxiv.org/abs/2012.00413) Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.
3. [EVA: An Open-Domain Chinese Dialogue System with Large-Scale Generative Pre-Training.](https://arxiv.org/abs/2108.01547) Hao Zhou, Pei Ke, Zheng Zhang, Yuxian Gu, Yinhe Zheng, Chujie Zheng, Yida Wang, Chen Henry Wu, Hao Sun, Xiaocong Yang, Bosi Wen, Xiaoyan Zhu, Minlie Huang, Jie Tang.
4. [Language Models are Unsupervised Multitask Learners.](http://www.persagen.com/files/misc/radford2019language.pdf) Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever.
