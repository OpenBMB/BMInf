# Demo Introduction

[Demo Code](https://github.com/OpenBMB/BMInf-demos)

BMInf-Demos includes application examples designed according to models in BMInf. These examples are:
+ **Fill Blank.** It is a use case based on CPM2.1. It supports arbitrary input of a paragraph and can generate corresponding content in the blank according to the context.
+ **Generate Story.** It is an example based on CPM1. You only need to write the beginning of a paragraph, and it can create a coherent essay for you.
+ **Dialogue.** It is an example based on the EVA model. Here, you can talk freely with the machine.

## Demonstration

+ Fill Blank
<div  align="center">    
<img src="https://raw.githubusercontent.com/OpenBMB/BMInf/master/docs/source/note/demo1.jpg" align=center />
</div>
<br/>

+ Generate Story

<div  align="center">    
<img src="https://raw.githubusercontent.com/OpenBMB/BMInf/master/docs/source/note/demo2.jpg" align=center />
</div>
<br/>

+ Dialogue
<div  align="center">    
<img src="https://raw.githubusercontent.com/OpenBMB/BMInf/master/docs/source/note/demo3.jpg" align=center />
</div>
<br/>

# **Install**

1. Run the follow command after installing `nvidia-docker2`:

```console
$ docker run -it --gpus 1 -v $HOME/.cache/bigmodels:/root/.cache/bigmodels -p 0.0.0.0:8000:8000 --rm openbmb/bminf-demos
```

2. Open your browser and visit http://localhost:8000/ to access the demo.
