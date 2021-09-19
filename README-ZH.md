# BMInf

[English] | 简体中文

[English]: ./README.md

BMInf (Big Model Inference) 是一个用于大规模预训练语言模型（pretrained language models, PLM）推理阶段的低资源工具包。

- **硬件友好** BMInf最低支持在NVIDIA GTX 1060单卡运行百亿大模型。在此基础上，使用更好的gpu运行会有更好的性能。
- **开源共享** 模型参数开源共享，用户在本地即可部署运行，无需访问或申请API。
- **能力全面** 支持生成模型CPM1 [[1](#ref)]、通用模型CPM2 [[2](#ref)]、对话模型EVA2 [[3](#ref)]，模型能力覆盖文本补全、文本生成与对话场景。
- **模型升级** 基于持续学习推出百亿模型新升级CPM2.1 [[2](#ref)]，文本生成能力大幅提高。
- **应用便捷** 基于工具包可以快速开发大模型相关下游应用。

## Demo
![demo](./docs/source/images/demo.gif)

## 安装
- 用pip安装：``pip install bminf``

- 从源代码安装: 下载工具包并在目录中运行 ``python setup.py install``

- 从Docker安装: ``docker run -it --gpus 1 -v $HOME/.cache/bigmodels:/root/.cache/bigmodels --rm openbmb/bminf:0.0.2 python3 examples/fill_blank.py``

运行BMInf的最低配置与推荐配置：

| | 最低配置 | 推荐配置 |
|-|-|-|
| 内存 | 16GB | 24GB
| GPU | NVIDIA GeForce GTX 1060 6GB | NVIDIA Tesla V100 16GB
| PCI-E |  PCI-E 3.0 x16 |  PCI-E 3.0 x16

## 快速上手

这里我们给出了一个使用BMInf的简单脚本。

首先，从模型库中导入一个想要使用的模型（如CPM1，CPM2或EVA2）。
```python
import bminf
cpm2 = bminf.models.CPM2()
```

定义输入文本，使用``<span>``标签来表示需要填入文本的位置。
```python
text = "北京环球度假区相关负责人介绍，北京环球影城指定单日门票将采用<span>制度，即推出淡季日、平季日、旺季日和特定日门票。<span>价格为418元，<span>价格为528元，<span>价格为638元，<span>价格为<span>元。北京环球度假区将提供90天滚动价格日历，以方便游客提前规划行程。"
```

使用``generate``函数获取结果，将文本中的``<span>``标签替换为得到的结果。

```python
for result in cpm2.generate(text, 
    top_p=1.0,
    top_n=10, 
    temperature=0.9,
    frequency_penalty=0,
    presence_penalty=0
):
    value = result["text"]
    text = text.replace("<span>", "\033[0;32m" + value + "\033[0m", 1)
print(text)
```
最终我们就得到了预测文本。更多的使用脚本详见``examples``文件夹。

## 支持模型

BMInf目前支持下列模型：

- **CPM2.1**. CPM2.1是CPM2 [[1](#ref)] 的升级版本。CPM2是一个拥有110亿参数的通用中文预训练语言模型。基于CPM2，CPM2.1新增了一个生成式的预训练任务并基于持续学习范式进行训练。实验结果证明CPM2.1比CPM2具有更好的生成能力。
- **CPM1.** CPM1 [[2](#ref)] 是一个拥有26亿参数的生成式中文预训练语言模型。CPM1的模型架构与GPT [[4](#ref)] 类似，它能够被应用于广泛的自然语言处理任务，如对话、文章生成、完形填空和语言理解。
- **EVA2.** EVA2是EVA [[3](#ref)] 模型的升级版本。EVA是一个有着28亿参数的中文预训练对话模型。EVA在很多对话任务上表现优异，尤其是在多轮人机交互对话任务上。

除了这些模型，我们目前致力于导入更多的预训练语言模型，尤其是大规模预训练语言模型。我们欢迎每一位贡献者通过提交issue来添加他们的模型。

## 运行性能

下表汇报了我们在不同平台上运行CPM2编码器和解码器的速度。用户可以在本机运行``benchmark/cpm2/encoder.py``和``benchmark/cpm2/decoder.py``来测试本机运行速度。

| GPU | 编码速度 (tokens/s) | 解码速度 (tokens/s) |
|-|-|-|
| NVIDIA GeForce GTX 1060 | 533 | 1.6
| NVIDIA GeForce GTX 1080Ti | 1200 | 12
| NVIDIA GeForce GTX 2080Ti | 2275 | 19

## 参与贡献
请参考开源社区链接和[贡献指南](./CONTRIBUTING.md)。

## 开源许可

该工具包使用[Apache 2.0](./LICENSE)开源许可证。

## 参考文献
<div id="ref"></div>

1. [CPM-2: Large-scale Cost-efficient Pre-trained Language Models.](https://arxiv.org/abs/2106.10715) Zhengyan Zhang, Yuxian Gu, Xu Han, Shengqi Chen, Chaojun Xiao, Zhenbo Sun, Yuan Yao, Fanchao Qi, Jian Guan, Pei Ke, Yanzheng Cai, Guoyang Zeng, Zhixing Tan, Zhiyuan Liu, Minlie Huang, Wentao Han, Yang Liu, Xiaoyan Zhu, Maosong Sun.
2. [CPM: A Large-scale Generative Chinese Pre-trained Language Model.](https://arxiv.org/abs/2012.00413) Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.
3. [EVA: An Open-Domain Chinese Dialogue System with Large-Scale Generative Pre-Training.](https://arxiv.org/abs/2108.01547) Hao Zhou, Pei Ke, Zheng Zhang, Yuxian Gu, Yinhe Zheng, Chujie Zheng, Yida Wang, Chen Henry Wu, Hao Sun, Xiaocong Yang, Bosi Wen, Xiaoyan Zhu, Minlie Huang, Jie Tang.
4. [Language Models are Unsupervised Multitask Learners.](http://www.persagen.com/files/misc/radford2019language.pdf) Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever.
