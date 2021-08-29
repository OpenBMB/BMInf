EasyInf

------------------------

English | [简体中文]

[简体中文]: http://www.baidu.com

EasyInf is a low-resource inference package for large-scale pretrained language models (PLMs).

- **Low resource.** Instead of running on large-scale GPU clusters, the package enables the running of the inference process for large-scale pretrained language models on personal computers!
- **Open.** Model parameters and configurations are all released, you don't need to access a PLM via online APIs, just run it on your computer! 
- **Green.** Run pretrained language models with fewer machines and GPUs, also with less energy consumption.

## Demo

Here we provide an online demo based on the package with CPM2.

## Installation

### Configurations

| | Mimimum Configurations | Recommended Configureation |
|-|-|-|
| Memory | 16GB | 24GB
| GPU | NVIDIA GeForce GTX 1060 6GB | NVIDIA Tesla V100 16GB
| PCI-E |  PCI-E 3.0 x16 |  PCI-E 3.0 x16



### From Source 
```
python3 setup.py install
```

### From Docker
```
docker build . -f docker/base.Dockerfile
```

## Quick Start

Load the model.
```
import bigmodels
model = bigmodels.models.CPM2()
```

```
model.text_to_id()
model.id_to_text()
model.get_token_id()
model.get_id_token()
```

```
model.encode()
model.decode()
```

## Performances

Performances on different platforms.


## Contributing
Links to the user community and contributing guidelines.

## License

The package is released under the Apache 2.0 License.

