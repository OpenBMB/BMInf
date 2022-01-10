# BMInf-Demos

[Demo Code](https://github.com/OpenBMB/BMInf-demos)

BMInf(Big Model Inference)-Demos is three examples designed according to three models in BMInf. These three examples are:
+ **Fill Blank.** It is a use case designed according to CPM2.1 model. It can support arbitrary input of a paragraph of text and generate corresponding blank content according to context semantics.
+ **Generate Story.** It is an example based on CPM1 model. You only need to write the beginning of a paragraph, and it can create a beautiful essay for you。
+ **Dialogue.** It is an example we created based on EVA model. Here, you can talk freely with the machine.


## Demonstrations
+ Fill Blank 
<div  align="center">    
<img src="./demo1.jpg" align=center />
</div>


+ Generate Story

<div  align="center">    
<img src="./demo2.jpg" align=center />
</div>

+ Dialogue
<div  align="center">    
<img src="./demo3.jpg" align=center />
</div>

## Installation

+ From docker: 
1. Install `nvidia-docker2` and run:

```console
$ docker run -it --gpus 1 -v $HOME/.cache/bigmodels:/root/.cache/bigmodels -p 0.0.0.0:8000:8000 --rm openbmb/bminf-demos
```

2. Visit http://localhost:8000/ with your browser.
