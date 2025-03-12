# 1 什么是深度学习模型
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;深度学习模型是一种机器学习模型，它由多个**神经网络层(layer)** 组成，这些层之间存在着多层的**非线性转换关系**。深度学习模型通过学习大量数据来提取和学习数据的高级特征表示，从而对输入数据进行分类、回归、生成等任务。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;工程上，常将**神经网络层(layer)** 抽象成计算机上可执行的**算子**如Conv2d、matmul、relu、sigmoid等，这些算子通过张量(Tensor)相互连接，组合成一张有向无环图，这个图就是我们常说的深度学习网络图，也称为深度学习模型图。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;指的主要的是，深度学习网络图中的算子有很多种类，有些算子并不能和 传统的 MLP中的神经网络层相对应，但也是很重要的，如reshape、permute、add、sconcat等。<br>

# 2 下载一个预训练好的深度学习模型
- [深度学习预训练模型下载](https://github.com/onnx/models)

# 3 可视化这个深度学习模型
- [深度学习模型可视化](https://netron.app/)
