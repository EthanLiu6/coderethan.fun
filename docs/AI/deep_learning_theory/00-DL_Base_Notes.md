[TOC]



# 一. DL_Base_Notes

## 1. Normalization（还未coding）🌟🌟🌟🌟🌟

![figure4](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/op-figure4.jpg)

### 1.1 Norm功能

Batch Norm，Layer Norm，Instance Norm，Group Norm
$$
y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
$$
- 去量纲，把数据调整到更强烈的数据分布
- 减少梯度消失和梯度爆炸
- 主要是有一个计算期望和方差的过程
- 做Norm的粒度不同，应用场景不同

其他资料：[Batch Norm的技术博客](https://blog.csdn.net/LoseInVain/article/details/86476010)

### 1.2 为啥不用BN来做NLP？

- NLP的主要数据格式是[batch_szie, seq_len, embedding_dim]
- Nlp 每个seq都是基本独立的，所以不能用Batch Norm
- LN对应的维度就是对embedding_dim进行的

**Batch Norm是逐channel（每个batch的同一个channel）进行标准化**，也就是垮batch的。图片恰好需要这种方式。

LN是逐batch进行标准化的。NLP中往往是一个一个的seq进行训练的，而且长度不同，更适合这种。**这让我想起了Attention的soft max操作是对一个行向量进行归一化的**

LayerNorm有助于稳定训练过程并提高收敛性。它的工作原理是对输入的各个特征进行归一化，确保激活的均值和方差一致。**普遍认为这种归一化有助于缓解与内部协变量偏移相关的问题，使模型能够更有效地学习并降低对初始权重的敏感性。**从架构图上看，LayerNorm在每个Transformer 块中应用两次，一次在自注意力机制之后，一次在FFN层之后，但是在实际工作中不一定如此。

文本长度不确定，而在LN层可以。

### 1.3 思考

- 在训练和推理时有何不同？？？

  > pytorch的模型有两种模式，在module模块里面有个‘training’属性，也有对应的API，里面明确指出了这个
  >
  > 在BatchNorm采用训练计算的结果（E和Var），应用到测试或者推理的时候
  >
  > 在Dropout后续会说，训练会drop掉，但推理不会，会成（1-rate）

  ```python
      def train(self: T, mode: bool = True) -> T:
          r"""Set the module in training mode.
  
          This has any effect only on certain modules. See documentations of
          particular modules for details of their behaviors in training/evaluation
          mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
          etc.
  
          Args:
              mode (bool): whether to set training mode (``True``) or evaluation
                           mode (``False``). Default: ``True``.
  ```

  

- 对于期望和方差计算策略？？？

  > `采用移动指数平均`，有点类似RNN了



### 1.4 RMS Norm(大模型使用)🌟🌟🌟

对LN做简化，对于NLP，对缩放敏感，对平移不敏感，所以分子不减$E_x$，减少了很大计算量

![image-20250326224954810](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250326224954810.png)

## 2. Activation🌟🌟🌟

### 2.1 Non-linear Activations的两种类型

一种是逐元素操作（Element wise 或者Point wise），eg:ReLU,Sigmoid,Tanh,等，另一种是操作对象（元素）之间具有相关性，eg.Softmax

> element wise是操作的数据间彼此独立，并且输入与输出大小一致

### 2.2





## 3. Loss Function🌟



## 4. Optimizer🌟🌟🌟🌟

> 动量后面的Admw那些据估计忘了



## 5. Transformer🌟🌟🌟🌟🌟

深入理解请阅读Transformer系列文章[Transformer](/AI/Transformer/)

### 5.1 为啥Attention的时候要除以$\sqrt{d_k}$？

$$
Attention(Q, K, V ) = softmax(\frac{Q·K^T}{\sqrt{d_k}})·V
$$

- 个人感觉跟Normalization的作用类似

当 $d_k$ 的值比较小的时候，两种点积机制(additive 和 Dot-Product)的性能相差相近，当 dk*d**k* 比较大时，additive attention 比不带scale 的点积attention性能好。 我们怀疑，对于很大的 dk*d**k* 值，点积大幅度增长，将softmax函数推向具有极小梯度的区域。 为了抵消这种影响，我们缩小点积 1dk√*d**k*1 倍。

### 5.2 为啥拆多头？为啥效果好了？

- 提取到了更多的信息（类似CNN的multi- kernel），数据分布组与组（子空间）之间独立

  > Multi-head attention允许模型的不同表示子空间联合**关注不同位置**的信息。 如果只有一个attention head，它的平均值会削弱这个信息。
- 减少计算量（应该可以在这一层减少原来的$\frac{1}{Num_{head}}$倍）

### 5.3 Cross Multi-Head Attention？

首先，Self- Attention与传统的Attention机制非常的不同：传统的Attention是基于source端和target端的隐变量（hidden state）计算Attention的，得到的结果是源端（source端）的每个词与目标端（target端）每个词之间的依赖关系。
    其次，Self-Attention首先分别在source端和target端进行自身的attention，仅与source input或者target input自身相关的Self -Attention，以捕捉source端或target端自身的词与词之间的依赖关系；然后再把source端的得到的self -Attention加入到target端得到的Attention中，称作为**Cross-Attention**，以捕捉source端和target端词与词之间的依赖关系。

###  5.4 Mask Multi-Head Attention

​    与Encoder的Multi-Head Attention计算原理一样，只是多加了一个mask码。mask 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果。Transformer 模型里面涉及两种 mask，分别是 padding mask 和 sequence mask。

### 5.5 Masking实现机理

具体的做法是，把**这些位置**的值**加上一个非常大的负数(负无穷)**，这样的话，经过 softmax，这些位置的概率就会接近0！

### 5.6 MHA、MQA和GQA

MQA多头共用K，V

GQA将头分组，组内共用KV

## 5.7 position embedding🌟🌟🌟🌟🌟

- attention并不会获取到输入序列（token与token之间）的位置信息，只有俩俩的相关性

- 最初的Transformer使用的是**绝对位置编码**，即：直接对第$batch$个输入序列的第$k$个token向量$X^{batch}_k$加上一个位置编码$P_k$，然后做同样的attention

- 关于编码向量（或者矩阵）的生成，采用三角位置编码方式，对于第奇数($2i+1$)个token采用cos，对于偶数采用sin

  > $p_{k,2i+1}=cos(\frac{k}{10000^{2i/d}})$
  >
  > $p_{k,2i}=sin(\frac{k}{10000^{2i/d}})$
  >
  > **其中：d表示位置向量的维度（应该也是输入次向量的embedding维度）**

- 其实提取到位置信息是靠变量$k$获取的，相当于是seq index

- **自己绘图发现该方案存在一定问题：** 当embedding dim**大于某一范围的**的时候，对应编码信息会向0-1分布。我测试的[2, 32, 64]，当dim 在[12, 20]发生某种变化，然后超过20开始趋向0-1分布，其实从公式也可以明显发现这种现象

  <img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250326170819283.png" alt="image-20250326170819283" style="zoom:40%;" />

  ![image-20250326170943850](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250326170943850.png)

- 该位置编码是在做Attention之前进行的，与embedding后的向量合并，再做attention

### x.x 其他

- 工程中将QKV的权重矩阵直接放在一块，shape就是原来$(embedding\_dim, embedding\_dim)$到 $(embedding\_dim, embedding\_dim \times 3)$

  ![img](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/1850883-20250209144655019-1620461538.jpg)



- Attention的时候是否需要对自身做？自回归的时候应当下一次token尽可能不是上一个词，所以矩阵对角线是否应当是趋于零的？

- 句子间的相似度计算方式有哪些？Attention为啥采用点积？

  > addivation
  >
  > dot-dart （为啥点积运算可以计算相似度）
  >
  > ![image-20250325155249034](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250325155249034.png)

- FFN的时候为啥dim要✖️4放大纬度
- encoder其实是特征提取的一个过程
- 为啥encoder的input和out（context vector）的纬度大小要一样？（因为有原始论文有6个encoder Layer或者cell）
- encoder层的attention会注意到前面的词吗？
- encoder的特征聚合是在什么时候
- 特征压缩
- softmax的时候是对attention结果的哪个纬度进行归一化（词与词相关性考虑，应该是掩码后矩阵分行向量），为啥归一化？（词与词之间的联系性，因为还要点乘value，所以应当是一个权重）
- mask的大小？（batch_size, seq_len, seq_len），应该和attention score一样，因为要码谁就跟谁一样

## 6. 🌟🌟🌟K-V Cache



## 7. 🌟🌟🌟常见的正则化方法

- Dropout
- 

## 8. Bert🌟🌟🌟🌟

输入后的对15%的三个处理

bert有三个编码，分别是



## 9. 位置编码总结

### 9.1 绝对位置编码

> 在Attention之前进行位置编码
>
> 对qkv都做位置编码

- 三角（固定）位置编码

- 可学习位置编码

  > 把编码矩阵当作可学习参数进行训练（大小与embedding后的输入一致）
  >
  > Bert模型就是采用的这种编码

### 9.2 相对位置编码

**不使用每个 token 的绝对位置，而是表示 token 之间的相对位置。这样做的好处是，模型不需要对每个位置使用单独的编码，而是通过计算相对距离来捕捉位置信息**。

> 在Attention之后进行位置编码
>
> 只对q和k做位置编码，对value不做，value是结果或者说是token本身的特征信息
>
> 根据数学原理推导的（依赖之前的位置编码公式来推导，**也就是由绝对位置编码启发而来**）
>
> 采用分桶思想（T5的思想）

### 9.3 旋转位置编码——RoPE（大模型常用）

> 也是相对位置编码的一种

### ![image-20250322144241104](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250322144241104.png)

> 通过构建数学模型
>
> 根据想要得到的效果进行反推得到





# 二. 课堂记录

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
