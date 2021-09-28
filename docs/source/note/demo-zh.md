# Demo介绍

[Demo Code](https://github.com/OpenBMB/BMInf-demos)

BMInf(Big Model Inference)-Demos 是根据BMInf中的三个模型设计的三个示例。它们分别是：
+ **文本填空** 它是根据CPM2.1模型设计的用例，它可以支持文本段落的任意输入，并根据上下文语义生成相应的空白内容。
+ **故事生成** 它是根据CPM1模型设计的例子，用户只需要写一段开头，它就可以为你创造一篇精彩的文章。
+ **智能对话** 这是我们基于EVA模型创建的一个示例。在这里，你可以和AI进行无障碍对话。

## 演示

+ 文本填空演示
<div  align="center">    
<img src="https://raw.githubusercontent.com/OpenBMB/BMInf/master/docs/source/note/demo1.jpg" align=center />
</div>
<br/>

+ 故事生成演示

<div  align="center">    
<img src="https://raw.githubusercontent.com/OpenBMB/BMInf/master/docs/source/note/demo2.jpg" align=center />
</div>
<br/>

+ 智能对话演示
<div  align="center">    
<img src="https://raw.githubusercontent.com/OpenBMB/BMInf/master/docs/source/note/demo3.jpg" align=center />
</div>
<br/>


## 安装

1. 安装`nvidia-docker2`后运行下面的命令：

```console
$ docker run -it --gpus 1 -v $HOME/.cache/bigmodels:/root/.cache/bigmodels -p 0.0.0.0:8000:8000 --rm openbmb/bminf-demos
```

2. 打开浏览器访问 http://localhost:8000/ 即可使用。

