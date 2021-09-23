# Introduction
BMInf (Big Model Inference) is a low-resource inference package for large-scale pretrained language models (PLMs). It has following features:

- **Hardware Friendly.** BMInf supports running models with more than 10 billion parameters on a single NVIDIA GTX 1060 GPU in its minimum requirements. Running with better GPUs leads to better performance.
- **Open.** The parameters of models are open. Users can access large models locally with their own machines without applying or accessing an online API.  
- **Comprehensive Ability.**  BMInf supports generative model CPM1 [[1](#ref)], general language model CPM2.1 [[2](#ref)], and dialogue model EVA [[3](#ref)]. The abilities of these models cover text completion, text generation, and dialogue generation.
- **Upgraded Model.** Based on CPM2 [[2](#ref)], the newly upgraded model CPM2.1 is currently supported. Based on continual learning, the text generation ability of CPM2.1 is greatly improved compared to CPM2.
- **Convenient Deployment.** Using BMInf, it will be fast and convenient to develop interesting downstream applications.

## Supported Models

BMInf currently supports these models:

- **CPM2.1.** CPM2.1 is an upgraded version of CPM2 [[1](#ref)], which is a general Chinese pre-trained language model with 11 billion parameters. Based on CPM2, CPM2.1 introduces a generative pre-training task and was trained via the continual learning paradigm. In experiments, CPM2.1 has a better generation ability than CPM2.

- **CPM1.** CPM1 [[2](#ref)] is a generative Chinese pre-trained language model with 2.6 billion parameters. The architecture of CPM1 is similar to GPT [[4](#ref)] and it can be used in various NLP tasks such as conversation, essay generation, cloze test, and language understanding.

- **EVA.** EVA [[3](#ref)] is a Chinese pre-trained dialogue model with 2.8 billion parameters. EVA performs well on many dialogue tasks, especially in the multi-turn interaction of human-bot conversations.

Besides these models, we are now working on adding more PLMs especially large-scale PLMs. We welcome every contributor to add their models to this project by proposing an issue.

## Performances

Here we report the speeds of CPM2 encoder and decoder we have tested on different platforms. You can also run ``benchmark/cpm2/encoder.py`` and ``benchmark/cpm2/decoder.py`` to test the speed on your machine!

| GPU | Encoder Speed (tokens/s) | Decoder Speed (tokens/s) |
|-|-|-|
| NVIDIA GeForce GTX 1060 | 533 | 1.6
| NVIDIA GeForce GTX 1080Ti | 1200 | 12
| NVIDIA GeForce GTX 2080Ti | 2275 | 19

## Contributing
Links to the user community and [contributing guidelines](./CONTRIBUTING.md).

## License

The package is released under the [Apache 2.0](./LICENSE) License.

## References
<div id="ref"></div>

1. [CPM-2: Large-scale Cost-efficient Pre-trained Language Models.](https://arxiv.org/abs/2106.10715) Zhengyan Zhang, Yuxian Gu, Xu Han, Shengqi Chen, Chaojun Xiao, Zhenbo Sun, Yuan Yao, Fanchao Qi, Jian Guan, Pei Ke, Yanzheng Cai, Guoyang Zeng, Zhixing Tan, Zhiyuan Liu, Minlie Huang, Wentao Han, Yang Liu, Xiaoyan Zhu, Maosong Sun.
2. [CPM: A Large-scale Generative Chinese Pre-trained Language Model.](https://arxiv.org/abs/2012.00413) Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.
3. [EVA: An Open-Domain Chinese Dialogue System with Large-Scale Generative Pre-Training.](https://arxiv.org/abs/2108.01547) Hao Zhou, Pei Ke, Zheng Zhang, Yuxian Gu, Yinhe Zheng, Chujie Zheng, Yida Wang, Chen Henry Wu, Hao Sun, Xiaocong Yang, Bosi Wen, Xiaoyan Zhu, Minlie Huang, Jie Tang.
4. [Language Models are Unsupervised Multitask Learners.](http://www.persagen.com/files/misc/radford2019language.pdf) Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever.