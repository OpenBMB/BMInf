# Installation

## From pip (Recommended)
```
pip install bminf
```

## From Source
```
git clone https://github.com/OpenBMB/BMInf.git
cd BMInf
python setup.py install
```

## From Docker 
```
docker run -it --gpus 1 -v $HOME/.cache/bigmodels:/root/.cache/bigmodels --rm openbmb/bminf python3 examples/fill_blank.py
```

After installation, you can run an example in the ``examples`` folder to find if it is installed correctly.

```
python examples/fill_blank.py
```

![demo](./images/demo.gif)

## Hardware Requirement

Here we list the minimum and recommended configurations for running BMInf. 

| | Minimum Configuration | Recommended Configuration |
|-|-|-|
| Memory | 16GB | 24GB
| GPU | NVIDIA GeForce GTX 1060 6GB | NVIDIA Tesla V100 16GB
| PCI-E |  PCI-E 3.0 x16 |  PCI-E 3.0 x16

## Software Requirement

BMInf requires CUDA 10 or CUDA 11 installed and all the dependencies can be automaticlly installed by the installation process.

- **python** >= 3.6
- **requests**
- **tqdm** 
- **jieba**
- **numpy** 
- **cupy-cuda<your_cuda_version>** >= 9, <10

Here is the table to find the corresponding cupy package for your CUDA version. Don't worry, this process will also be automatically done.

| CUDA Version | Package Name |
|-|-|
v10.0 | cupy-cuda100
v10.1 | cupy-cuda101
v10.2 | cupy-cuda102
v11.0 | cupy-cuda110
v11.1 | cupy-cuda111
v11.2 | cupy-cuda112
v11.3 | cupy-cuda113
v11.4 | cupy-cuda114

