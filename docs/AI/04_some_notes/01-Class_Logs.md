# 01-Class_Logs

## 0301-0302 Transformer🌟🌟🌟

- 0301:input序列长度大于embedding时候的seq_len时, input的输入序列会按照seq_len进行切割拼接到batch上吗? (老师讲了encoder时候input不足seq_len时候使用mask然后想问的另一个问题)
- 0302:`K-V cache`时候当预测下一个时间步的时候与之前的做Attention的时候, 中途会取出cache里的K—V吗还是只取出里面的K还是只在最后一个结束后才整体取一次 (我想问的也就是在一个batch或者一个seq的访存情况, 每一个时间步都需要访问cache一次吗)
- 是直接使用缓存的填充矩阵还是需要拿出缓存数据(读还是取)



## 0308-0309——PyTorch


> 提及: 混合精度训练

### 1.1 Tensor 中数据的连续性

reshape, transpose, view, T(转置), permute

transpose会让raw data不变(共用),  mata data的stride和shape等属性就变了 is_contiguous()不连续, 但reshape和permute这些是不会变的,因为他们会发生data copy,  contiguous()会发生copy raw data数据

view和reshape的区别

view更加安全, 不会重新拷贝数据, 但数据不连续不能使用view,也就是stride不协调, reshape不会错误, 会重新拷贝数据, 数据也连续

permute和transpose会让stride属性改变, 从而发生数据不连续, 通常使用后要加一个contiguous()让数据连续

### 1.2 pytorch autograd

<img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250308163009045.png" alt="image-20250308163009045" style="zoom:36%;" />

……………………

叶子结点+requests_grad=True才有最终的grad, 非叶子结点中途可能会计算grad, 但用了就会丢弃(requests_grad=True的)



梯度累加也有可能, 多个step的梯度累加, 隐式增加batch

若没进行xxx.grad.zero_()或者xxx.grad = None, 则会进行accumulate()累加grad, 这两种方法有一点区别, zero__()会置零,会占用显存, 但=None的话会释放显存, 两者各有好坏



### 1.3 inplace-op

![image-20250309102559857](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250309102559857.png)

叶子结点的Tensor变量不能进行in-place操作, 因为要更新梯度的时候要用叶子结点

<img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250309113212792.png" alt="image-20250309113212792" style="zoom: 50%;" />

<img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250309114217344.png" alt="image-20250309114217344" style="zoom:50%;" />

no_grad()底层是基于set_grad_enable(Flase)的

### 1.4 自动微分机制(auto grad) 重点：

- pytorch中 正向forward 对我们用户是可见的，但是backward对我们用户是不可见的；
- 一般情况下，每一个正向的函数，都对应一个反向的函数（grad_fn--> Tensor中）；
- tensor：requires_grad = True
- tensor: grad --> tensor 中存储grad的地方；
- tensor: grad_fn --> 存储我们反向函数的地方
- tesnor: is_leaf --> 这个tensor 是不是 叶子节点；
- net::all weight --> 都是leaf
- 叶子节点的梯度会自动保存下来的（weight）；
- 中间的 activation 的梯度会计算，但是不保留；
- pytorch 动态图 vs tensorflow 静态图；
- 我们不能改变一个非叶子节点的 requires_grad;
- 非叶子（一个函数的output）节点它的 requires_grad 自动推导的；
- 非叶子节点对应函数的inputs 中只要有一个 requires_grad = True, 那么这个非叶子节点的requires_grad = True;
- torch.no_grad() 会使得里面的新的tensor requires_grad = False
- inplace的操作，非常大的风险：覆盖了原来的值，导致反向传播时计算不准确；
- 标量的梯度才能被隐式创建，隐式创建（.backward(1)）；
- 一般情况下，.backward(gradient)是有输入的: ;



### 2.1 torch.nn.Module

train模式和veal模式不会对grad的情况做修改,只是对训练和推理的对应的算子做不同的处理(等价处理)

常用算子dropout和BachNorm 

xxx.cuda()的时候搬迁的是_parameters到cuda, 还有buffer也搬迁到cuda, 并没有将模型结构进行搬迁.

按照深度优先遍历sub module,将里面的_parameters和buffer到cuda, 数据类型转换也是一样的操作

c++底层实现了一个dispather分发机制,按照device属性分发, 对应device会调用对应的fn算子, 计算部分才执行

_parameters()送参数给优化器的时候将所有的parameters送到optim, 但数据共用, 同时更新

钩子函数(没太懂)

---



## 0315-0316（续PyTorch）

### 1.1 回顾

1.Tensor类和重要属性
2.autograd，动态图
3.Module以及属性和方法

> training,_parameters,_buffers,_modules(hooks是主要用二次开发等情况)

子模块啥时候定义的呢？

_parameters,_buffers哪些有哪些没有

将module里的parameters传给optim，会通过调用parameters()进行

一系列方法具体情况

### 1.2 问题合集

1. 在讲transformer的padding mask的时候想到，如果输入seq_len大于了定义的seq_len，会直接截断还是截断再拼接到下一个batch
2. 在sequence mask的时候，忘了要问啥了
3. 在normalization层的时候不是有两个学习的参数吗，这俩参数是一次forward训练一次还是单独有自己的训练？还有，这俩参数是咋更新的？
4. dataset会迭代的将所有数据加载到内存吗，然后dataloader再一批次的提取吗

with  torch.no_grad():
	eval时候用，计算图不再进行，对require_grads=True的不进行梯度计算，显存占用量会减少，activation的就会丢弃

dataset会迭代的将所有数据加载到内存吗，然后dataloader再一批次一批次的提取吗？还是说dataloader准备拿一个batch，然后dataset根据batch_size迭代获取size条。

> 是后者，也就是I/O的时候，batch_size太小的话会增加I/O负担

### 2.1 torch.optim

参数传param的时候的传递和打包方式

self.param_groups

==self.state==：训练时候显存消耗的主要项（优化器的动量项有关）
他是一个dict，keys是tensor，values也是
模型

>移动指数平均是啥忘了

def load_state_dict

### 2.2 learning rate 调整方案

Torch.optim.lr_scheduler

震荡类型的学习率调整是减少进入局部最优解的情况

==状态字典==，三个地方见过，都类似，模型保存时候需要有

### 2.3 模型保存和加载

==动态图==

1.save state_dict的时候只有参数，save model的时候无法直接保存整个网络，但是他的材料（init）的那些会保存，模型加载的时候能通过，但runing time时候，forward并没有，必须导入或者自己实现，需要原来Net的签名（具体定义可以不一致，会放入_modules）

2.如果是自己写的算子，在init时候也放入_modules吗？

![img](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/%7B6fbad3cc-1899-4404-b3b3-d91f7da5cb95%7D.png)

3.==onnx==模型保存必须输入对应的input，自己run一遍，是一个静态图

4.训练中的保存和加载（check point）==模型保存的几种参数类型==）

![image-20250316113313785](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250316113313785.png)

### 3.1 Dataset and Dataloader

> 只学习pytorch的，后续自己补hf的那些

### 4.1 NLP

GPT：自监督训练得到预训练模型（采用迁移学习）

Bert：完形填空

迁移学习：预训练+微调（微调的数据集就是专业领域的数据集）

### 4.2 Bert

1.两个任务：MLM和NSP

2.Embedding，词嵌入

词，句子（分段），位置 嵌入

![image-20250316165604714](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250316165604714.png)

> transformer的词嵌入式用三角位置嵌入

![image-20250316175009793](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250316175009793.png)



未讲知识：分词器tokenizer



## 0322

### 1.1 回顾

`bert4torch`的ner项目讲解和debug

### 2.1 T5讲解

### 2.2 position embedding🌟🌟🌟🌟🌟

绝对位置编码

- 三角函数式(Sinusoidal)
- 可学习(Learnable)

相对位置编码

- **是在Attention的时候才位置编码**
- 只对q和k做位置编码，对value不做，value是结果或者说是token本身的特征信息
- T5的分桶思想

==旋转位置编码==（大模型使用的方法）

- 根据数学原理推导

![image-20250322144241104](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250322144241104.png)

- 想要得到的效果=>反推

![image-20250322144600625](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250322144600625.png)

![image-20250322145503812](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250322145503812.png)

### 3.1 GPT

GPT-1 已经出现zero-shot迹象，层归一化还是之前的post-norm

GPT-2 零样本学习，即zero-shot，层归一化有点变化，改成per-Norm

> 相当于纯预训练

![image-20250322170432945](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250322170432945.png)

GPT-3 few-shot（给案例），发现模型规模可以提高能力，最后实现了无需微调到达一些较好的任务处理，架构基本和GPT-2一致，但加了一个新的‘交替的稠密和稀疏的’Attention，余弦衰减的学习率策略，batch-size从小变大，再加上0.1的权重衰减正则化

> Few-shot, one-shot, zero-shot
>
> • **Few-Shot（FS）：** 模型在推理时给出K个任务示例作为上下文信息，同时提供任务的自然语言描述，但不允许模型进行权重更新。通常将K设置在10到100的范围内，以适应模型的上下文窗口。
>
> • **One-Shot（1S）：** 模型在推理时通过提供一个任务示例作为上下文信息，同时还有任务的自然语言描述。这种方式最接近于人类在解决某些任务时所使用的方式。
>
> • **Zero-Shot（0S）：** 不提供任何上下文信息，模型只给出一个描述任务的自然语言指令。





## 0323

### 1.1 课前准备

- T5模型数据集下载并修改代码

- tumx和终端不后台从训练

>  sh Train.sh > ./xxx.log &

> tumx



### 2.1 Scaling Laws

> tip：模型规模搞大可以提高自己的能力？

- 实验变量：

> C, D, N
>
> ![image-20250323103615212](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250323103615212.png)

- 数据规模与模型规模扩大比：5/8
- 一些超参数的设定



----

### 3.1 分布式训练

1. 并行可以并行哪些？拆哪些？
2. 多卡并行范式

- **数据并行性**(DP)：将模型（所有weight）复制到别的Worker中，所以模型大于单个显存的时候使用这种方式无法很好工作

- 模型并行性(MP)，存在Bubble问题

- MP优化：管线并行性（MP --> PP），又叫**流水线并行**

  > pp传播的是activation（前向）和对应的grad（反向）
  >
  > GPipe的不足：最后一个执行完才能backward
  >
  > PipeDream：前反向穿插，调度问题很难，工程上=>Pipeline flash。实现one F one B

- **张量并行性（TP）**

  > 前面是纵向分割，现在提出用横向分割
  >
  > 将一个算子的Tenser分到多节点计算

- **专家混合（EP，MoE）**

  > G shard
  >
  > switch Transformer

  ![image-20250323144057238](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250323144057238.png)

- 后面还有CP，xxxxP

3. 分布式框架

   > pytorch的
   >
   > deepspeed



### 4. 显存占用问题

#### 4.1 解决方案

> 之前有多个batch的grad累加

1. 重计算（recompute）

Pytorch2.6开始更加新的重计算

2. offload ：用完就放到CPU

   eg:

   <img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250323152635318.png" alt="image-20250323152635318" style="zoom:150%;" />

3. gradient accumulate

#### 4.2 显存分析

1.API

<img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250323152133484.png" alt="image-20250323152133484"  />

2.显存高峰期

在第一个step不会，理论上是在第二个step的forward之后

### 5. 混合精度训练（AMP）🌟🌟🌟🌟🌟

![alt text](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image.png)

大模型必用，加速训练

> ==下一个热点FP8==

1.权重副本fp32

<img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250323155428008.png" alt="image-20250323155428008" style="zoom:50%;" />

2.损失缩放

3.输出存储到单精度，最终变半精度



舍入误差（下溢）

### 6. Apex





### 7.

1.之前看那个state_dict最后存储的key和value都是tenser的设计策略是不是跟这儿有关系

2.多机的时候是不是也需要ssh 密钥组网