# BMInference

[English] | 简体中文

[English]: ./README.md

BMInference (Big Model Inference) 是一个用于大规模预训练语言模型（pretrained language models, PLM）推理阶段的低资源工具包。

- **低资源** 无需在大规模GPU集群上运行，该工具包允许用户在个人电脑上使用大规模预训练语言模型进行推理！
- **开放** 模型参数和配置全部开放，用户无需通过在线API使用预训练语言模型，用户可以直接在本地运行。
- **绿色** 使用更少的机器和GPU、更少的能源运行预训练语言模型。

## Demo
Todo：CPM2演示示例
![demo](./docs/images/demo.gif)

## 安装
- 用pip安装：``pip install bminference``

- 从源代码安装: 下载工具包并在目录中运行 ``python setup.py install``

- 从Docker安装: ``docker build . -f docker/base.Dockerfile``

运行BMInference的最低配置与推荐配置：

| | 最低配置 | 推荐配置 |
|-|-|-|
| 内存 | 16GB | 24GB
| GPU | NVIDIA GeForce GTX 1060 6GB | NVIDIA Tesla V100 16GB
| PCI-E |  PCI-E 3.0 x16 |  PCI-E 3.0 x16

## 快速上手

这里我们给出了一个使用BMInference的简单脚本。

首先，从模型库中导入一个想要使用的模型（如CPM1，CPM2或EVA2）。
```python
import bminference
cpm2 = bminference.models.CPM2()
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

## 运行性能

下表汇报了我们在不同平台上运行CPM2编码器和解码器的速度。用户可以在本机运行``benchmark/cpm2/encoder.py``和``benchmark/cpm2/decoder.py``来测试本机运行速度。

| GPU | 编码速度 (tokens/s) | 解码速度 (tokens/s) |
|-|-|-|
| NVIDIA GeForce GTX 1060 | 533 | 1.6
| NVIDIA GeForce GTX 1080Ti | 1200 | 12
| NVIDIA GeForce GTX 2080Ti | 2275 | 19

## 参与贡献
Todo：开源社区链接和贡献指南

## 开源许可

该工具包使用[Apache 2.0](./LICENSE)开源许可证。

