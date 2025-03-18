# 一. DL_Base_Notes

## 1. 🌟🌟🌟🌟🌟Normalization

Batch Norm，Layer Norm，Instance Norm，Group Norm
$$
y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
$$
[Batch Norm的技术博客](https://blog.csdn.net/LoseInVain/article/details/86476010)

**思考：在训练和推理时有何不同？？？**

## 2. 🌟🌟🌟Activation

### 2.1 Non-linear Activations 的两种类型

一种是逐元素操作（Element wise 或者Point wise），eg:ReLU,Sigmoid,Tanh,等，另一种是操作对象（元素）之间具有相关性，eg.Softmax

### 2.2

## 3. 🌟Loss Function

## 4. 🌟🌟🌟🌟Optimizer

> 动量后面的Admw那些据估计忘了

## 5. 🌟🌟🌟🌟🌟Transformer

> 整理来源于作者：https://www.cnblogs.com/rossiXYZ/p/18706134，已获得许可

### 5.0 引入，为啥出现？

Transformer本身还是seq2seq结构的一个模型架构，但像RNN这样的网络他有很多问题点，每一次的预测输出是建立在上一次的输出基础上的，也算是早起自回归模型的问题点：

> - 串行运行，很难以并行化的方式开展训练、提升效率。
> - 只有$h_{t-1}$时刻的信息，容易丢失信息
> - "一步错，步步错"
> - 梯度消失问题
> - 输入输出序列等长限制（n2n）

由此，研究者们就在此基础上设计出了一种"优化版的Encoder 2 Decoder"的架构，其中的过渡就是Context Vector (用C表示)，**输入句子每个时间步的信息**都包含在了这个上下文中。简单理解可以认为Encoder进行特征提取得到输入信息对应的Context Vector，然后Decoder进行解码：

> - 在每个时刻，解码器都是自回归的，即上一个时刻的输出（产生的token $y_{t−1}$）作为下当前时刻$t$的输入之一，生成当前时刻的token $y_t$。
> - 解码器最初的输入是中间语义上下文向量C，解码器依据C计算出第一个输出词和新的隐状态，即解码器的每个预测都受到先前输出词和隐状态的微妙影响。
> - 解码器接着用新的隐状态和第一个输出词作为联合输入来计算第二个输出词，以此类推，直到解码器产生一个 EOS（End Of Service/序列结束）标记或者达到预定序列长度的边界。

从宏观角度看，序列建模的核心就是研究如何把长序列的上下文压缩到一个较小的状态中（好好领悟这句话）。

咋压缩呢？早期的有马尔可夫假设，也就是近因效应，如果考虑前面n个单词，这就得到了N-gram模型，即当前单词的概率取决于前n个单词。

### 5.1 Attention机制

这个在transformer之前就有了，它其实有一定的实际意义，有三种主流禅诗：

> - 注意力机制的本质是上下文决定一切。
> - 注意力机制是一种资源分配方案。
> - 注意力机制是信息交换，或者说是是“全局信息查询”。

> 其实，论文“Recurrent Models of Visual Attention”中有一段话就深刻的印证了资源分配这个角度。具体如下：人类感知的一个重要特性是，人们不会一次处理整个场景。相反，人类有选择地将注意力集中在视觉空间的某些部分上，以在需要的时间和地点获取信息，并随着时间的推移将不同注视点的信息结合起来，建立场景的内部表示，指导未来的眼球运动和决策。将计算资源集中在场景的各个部分可以节省“带宽”，因为需要处理的“像素”更少。但它也大大降低了任务的复杂性，因为感兴趣的对象可以放置在注视的中心，而注视区域外的视觉环境的无关特征（“杂乱”）自然会被忽略。

所以，核心就是分配不同的权重，那又引出问题：

- 在哪里做注意力计算？
- 如何做注意力计算？

### 5.2 Q K V的引入

注意力模型的内部流程如下图所示，该模型的目标是生成V中向量的加权平均值，具体计算流程如下。

- 标号1是输入（两个输入），从输入生成的特征向量F会进一步生成键矩阵K和值矩阵V。

- 标号2使用矩阵K和查询向量q作为输入，通过相似度计算函数来计算注意力得分向量e。q表示对信息的请求，$e_l$表示矩阵K的第l列对于q的重要性。

- 标号3通过对齐层（比如softmax函数）进一步处理注意力分数，进而得到注意力权重a。

- 标号4利用注意力权重a和矩阵V进行计算，得到上下文向量c。

  ![img](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/1850883-20250209102028813-842747599.jpg)

  上图注意力模型中，有两个输入：q（正在处理的序列）和F（被关注的序列），F又分别转换为K和V，这三个变量综合起来使用就可以满足我们的需求。

  从词典的角度来看也许可以促进理解。query是你要找的内容，key是字典的索引（字典里面有什么样的信息），value是对应的信息。

  **注意力机制的计算总体可以分为两步：**

  1. 在所有输入信息上计算注意力分布。编码器不只是传递最后一个隐藏状态，而是传入所有的隐藏状态到解码器。
  2. 根据注意力分布来计算输入信息的加权平均。需要注意，这是一种数据依赖的加权平均，是一种灵活、高效的全局池化操作。

### 5.3 Transformer架构

#### 5.3.1 整体结构

从网络结构来分析，Transformer 包括了四个主体模块。

> - 输入模块，对应下图的绿色圈。
> - 编码器（Encoder），对应下图的蓝色圈。
> - 解码器（Decoder），对应下图的红色圈。编码器和解码器都有自己的输入和输出，编码器的输出会作为解码器输入的一部分（位于解码器的中间的橙色圈）。
> - 输出模块，对应下图的紫色圈。

确切的说，蓝色圈是编码器层（Encoder layer），红色圈是解码器层（Decoder layer）。图中的 N×代表把若干具有相同结构的层堆叠起来，这种将同一结构重复多次的分层机制就是栈。为了避免混淆，我们后续把单个层称为编码器层或解码器层，把堆叠的结果称为编码器或解码器。在Transformer论文中，Transformer使用了6层堆叠来进行学习。

![img](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/1850883-20250209144200970-309684502.jpg)

#### 5.3.2 Attention结构

在Transformer中有三种注意力结构：全局自注意力，掩码自注意力和交叉注意力，具体如下图所示。

![img](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/1850883-20250209144227590-945674250.jpg)

![img](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250209144236400-880372719.jpg)

论文解读三个Attention：

> Transformer 模型以三种不同的方式使用多头注意力机制：
>
> - 在“编码器 - 解码器注意力”层中，quary来自前一个解码器层，而记忆键和值来自编码器的输出。这使得解码器中的每个位置都能关注输入序列中的所有位置。这模仿了诸如[38, 2, 9]等序列到序列模型中典型的编码器 - 解码器注意力机制。
> -  编码器包含自注意力层。在自注意力层中，所有的键、值和查询都来自同一个地方，在这种情况下，来自编码器前一层的输出。编码器中的每个位置都可以关注到编码器前一层中的所有位置。
> -  同样，解码器中的自注意力层允许解码器中的每个位置关注到解码器中该位置之前的所有位置。我们需要防止解码器中的左向信息流以保持自回归属性。我们在缩放点积注意力内部通过掩码（设置为 -inf）来实现这一点，掩码掉 softmax 输入中对应于非法连接的所有值。

![img](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/1850883-20250209144246216-890349384.jpg)

##### 全局自注意力层

全局自注意力层（Global self attention layer）位于编码器中，它负责处理整个输入序列。在全局自注意力机制中，序列中的每个元素都可以直接访问序列中的其它元素，从而与序列中的其他元素建立动态的关联，这样可以使模型更好地捕捉序列中的重要信息。自注意力的意思就是关注于序列内部关系的注意力机制，那么是如何实现让模型关注序列内部之间的关系呢？自注意力将query、key、value设置成相同的东西，都是输入的序列，就是让注意力机制在序列的本身中寻找关系，注意到不同部分之间的相关性。

对于全局自注意力来说，Q、K、V有如下可能：

- Q、K、V都是输入序列。
- Q、K、V都来自编码器中前一层的输出。编码器中的每个位置都可以关注编码器前一层输出的所有位置。

再细化来说，Q是序列中当前位置的词向量，K和V是序列中的所有位置的词向量。

##### 掩码自注意力

掩码自注意力层或者说因果自注意力层（Causal attention layer）可以在解码阶段捕获当前词与已经解码的词之间的关联。它是对解码器的输入序列执行类似全局自注意力层的工作，但是又有不同之处。

Transformer是自回归模型，它逐个生成文本，然后将当前输出文本附加到之前输入上变成新的输入，后续的输出依赖于前面的输出词，具备因果关系。这种串行操作会极大影响训练模型的时间。**为了并行提速，人们引入了掩码**，这样在计算注意力时，通过掩码可以确保后面的词不会参与前面词的计算。

对于掩码自注意力来说，Q、K、V有如下可能：

- Q、K、V都是解码器的输入序列。
- Q、K、V都来自解码器中前一层的输出。解码器中的每个位置都可以关注解码器前一层的所有位置。

再细化来说，Q是序列中当前位置的词向量，K和V是序列中的所有位置的词向量。

##### 交叉注意力层

交叉注意力层（Cross attention layer）其实就是传统的注意力机制。交叉注意力层位于解码器中，但是其连接了编码器和解码器，这样可以刻画输入序列和输出序列之间的全局依赖关系，完成输入和输出序列之间的对齐。因此它需要将目标序列作为Q，将上下文序列作为K和V。

对于交叉注意力来说，Q、K、V来自如下：

- Q来自前一个解码器层，是因果注意力层的输出向量。
- K和V来自编码器输出的注意力向量。

这使得解码器中的每个位置都能关注输入序列中的所有位置。另外，编码器并非只传递最后一步的隐状态，而是把所有时刻（对应每个位置）产生的所有隐状态都传给解码器，这就解决了中间语义编码上下文的长度是固定的问题。

![img](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/1850883-20250218212308574-176454824-20250318112643142.jpg)

#### 5.3.3 执行流程

我们再来结合模型结构图来简述推理阶段的计算流程，具体如下图所示。

![img](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/1850883-20250209144319485-2071213191.jpg)

假设我们进行机器翻译工作，把中文”我吃了一个苹果“翻译成英文”I ate an apple“，在假设模型只有一层，执行步骤如下：

1. 处理输入。用户输入自然语言句子”我吃了一个苹果“；tokenizer先把序列转换成token序列；然后Input Embedding层对每个token进行embedding编码，再加入Positional Encoding（位置编码），最终形成带有位置信息的embedding编码矩阵。编码矩阵用 Xn∗dXn∗d 表示， n 是句子中单词个数，d 是表示向量的维度（论文中 d=512）。注：原论文图上的输入是token，本篇为了更好的说明，把输入设置为自然语言句子。
2. 编码器进行编码。编码矩阵首先进入MHA（Multi-Head Attention，多头注意力）模块，在这里每个token会依据一定权重把自己的信息和其它token的信息进行交换融合；融合结果会进入FFN（Feed Forward Network）模块做进一步处理，最终得到整个句子的数学表示，句子中每个字都会带上其它字的信息。整个句子的数学表示就是Encoder的输出。
3. 通过输入翻译开始符来启动解码器。
4. 解码器进行解码。解码器首先进入Masked Multi-Head Attention模块，在这里解码器的输入序列会进行内部信息交换；然后在Multi-Head Attention模块中，解码器把自己的输入序列和编码器的输出进行融合转换，最终输出一个概率分布，表示词表中每个单词作为下一个输出单词的概率；最终依据某种策略输出一个最可能的单词。这里会预测出第一个单词”I“。
5. 把预测出的第一个单词”I“和一起作为解码器的输入，进行再次解码。
6. 解码器预测出第二个单词”ate“。

针对本例，解码器的每一步输入和输出具体如下表所示。

![img](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/1850883-20250209144327875-1762767210.jpg)



**论文"Attention is not all you need"指出如果没有skip connection（residual connection-残差链接）和MLP，自注意力网络的输出会朝着一个rank-1的矩阵收缩。即，skip connection和MLP可以很好地阻止自注意力网络的这种”秩坍塌（秩坍塌）退化“。这揭示了skip connection，MLP对self-attention的不可或缺的作用**

### 5.1 为啥Attention的时候要除以$\sqrt{d_k}$？

$$
Attention(Q, K, V ) = softmax(\frac{Q·K^T}{\sqrt{d_k}})·V
$$

当 dk*d**k* 的值比较小的时候，两种点积机制(additive 和 Dot-Product)的性能相差相近，当 dk*d**k* 比较大时，additive attention 比不带scale 的点积attention性能好。 我们怀疑，对于很大的 dk*d**k* 值，点积大幅度增长，将softmax函数推向具有极小梯度的区域。 为了抵消这种影响，我们缩小点积 1dk√*d**k*1 倍。

### 5.2 为啥拆多头？为啥效果好了？

### 5.3 Cross Multi-Head Attention？

首先，Self- Attention与传统的Attention机制非常的不同：传统的Attention是基于source端和target端的隐变量（hidden state）计算Attention的，得到的结果是源端（source端）的每个词与目标端（target端）每个词之间的依赖关系。
    其次，Self-Attention首先分别在source端和target端进行自身的attention，仅与source input或者target input自身相关的Self -Attention，以捕捉source端或target端自身的词与词之间的依赖关系；然后再把source端的得到的self -Attention加入到target端得到的Attention中，称作为**Cross-Attention**，以捕捉source端和target端词与词之间的依赖关系。

###  5.4 Mask Multi-Head Attention

​    与Encoder的Multi-Head Attention计算原理一样，只是多加了一个mask码。mask 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果。Transformer 模型里面涉及两种 mask，分别是 padding mask 和 sequence mask。

### 5.5 Masking实现机理

具体的做法是，把**这些位置**的值**加上一个非常大的负数(负无穷)**，这样的话，经过 softmax，这些位置的概率就会接近0！

### 5.6 MQA和GQA

MQA多头共用K，V

GQA将头分组，组内共用KV

## 6. 🌟🌟🌟K-V Cache

## 7. 🌟🌟🌟常见的正则化方法

# 二. 课堂记录

## 🌟0301-0302

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
