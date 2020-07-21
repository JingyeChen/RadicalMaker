# RadicalMaker

![](https://img.shields.io/badge/version-1.4.0-green)
![](https://img.shields.io/badge/build-pass-green)


## 项目简介
RadicalMaker可以随机生成一串偏旁部首序列，并组合成新字，即字典中不存在的字。一个汉字可以解析成一颗树，例如
‘㐻:⿰ 亻 内’表示‘㐻’由左右结构组成，包含左侧的‘亻’与右侧的‘内’，偏旁部首序列即为树的前序表示。

如下图所示，左侧为汉字的树形表示，右侧为汉字常见的几种结构符号

<div align=center><img width="636" height="374" src="https://s1.ax1x.com/2020/07/21/UIu6Et.jpg"></div>

## 使用说明
### pip安装
该项目已上传至pip，可以使用pip install的方式安装，建议使用`python3.6`
```python
pip install cjy==1.4
```
使用cjy包需要网络连接，连接后会自动下载相关资源
```python
import cjy
from cjy.generator import Generator
tool = Generator()
tool.help() # 查看generator接收的参数
tool.generate() # 执行生成指令
```

### git clone安装
```python
git clone https://github.com/JinGyeSetBirdsFree/RadicalMaker
python generator.py # 可修改文件底部的传入参数
```

## 效果展示
默认情况下，生成的图片会在当前路径的image文件夹下，标签文件为image/label.txt

<div align=center><img width="636" height="374" src="https://s1.ax1x.com/2020/07/21/UIusHI.png"></div>

<div align=center><img width="636" height="374" src="https://s1.ax1x.com/2020/07/21/UIucUP.jpg"></div>

## 写在最后
该项目由FudanOCR团队研发，启发于Zhang Jianshu博士的研究成果，在此表示感激！

