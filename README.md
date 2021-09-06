# BMInference

English | [简体中文]

[简体中文]: ./README-ZH.md

BMInference (Big Model Inference) is a low-resource inference package for large-scale pretrained language models (PLMs).


- **Low Resource.** Instead of running on large-scale GPU clusters, the package enables the running of the inference process for large-scale pretrained language models on personal computers with only one GPU!
- **Open.** Model parameters and configurations are all publicly released, you don't need to access a PLM via online APIs, just run it on your computer! 
- **Green.** Run pretrained language models with fewer machines and GPUs, also with less energy consumption.

## Demo
![demo](./docs/images/demo.gif)

## Install

- From pip: ``pip install bminference``

- From source code: download the package and run ``python setup.py install``

- From docker: ``docker run -it --gpus 1 -v $HOME/.cache/bigmodels:/root/.cache/bigmodels --rm openbmb/bminference:0.0.2 python3 examples/fill_blank.py``

Here we list the minimum and recommended configurations for running BMInference. 

| | Minimum Configuration | Recommended Configuration |
|-|-|-|
| Memory | 16GB | 24GB
| GPU | NVIDIA GeForce GTX 1060 6GB | NVIDIA Tesla V100 16GB
| PCI-E |  PCI-E 3.0 x16 |  PCI-E 3.0 x16

## Quick Start

Here we provide a simple script for using BMInference. 

Firstly, import a model from the model base (e.g. CPM1, CPM2, EVA2).
```python
import bminference
cpm2 = bminference.models.CPM2()
```

Then define the text and use the ``<span>`` token to denote the blank to fill in.
```python
text = "北京环球度假区相关负责人介绍，北京环球影城指定单日门票将采用<span>制度，即推出淡季日、平季日、旺季日和特定日门票。<span>价格为418元，<span>价格为528元，<span>价格为638元，<span>价格为<span>元。北京环球度假区将提供90天滚动价格日历，以方便游客提前规划行程。"
```

Use the ``generate`` function to obtain the results and replace ``<span>`` tokens with the results.

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
Finally, you can get the predicted text. For more examples, go to the ``examples`` folder.

## Performances

Here we report the speeds of CPM2 encoder and decoder we have tested on different platforms. You can also run ``benchmark/cpm2/encoder.py`` and ``benchmark/cpm2/decoder.py`` to test the speed on your machine!

| GPU | Encoder Speed (tokens/s) | Decoder Speed (tokens/s) |
|-|-|-|
| NVIDIA GeForce GTX 1060 | 533 | 1.6
| NVIDIA GeForce GTX 1080Ti | 1200 | 12
| NVIDIA GeForce GTX 2080Ti | 2275 | 19

## Contributing
Links to the user community and contributing guidelines.

## License

The package is released under the [Apache 2.0](./LICENSE) License.

