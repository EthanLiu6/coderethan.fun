探秘Transformer系列之（5）--- 训练&推理
============================

0x00 概述
-------

Transformer训练的目的是通过对输入源序列和模型输出序列的学习，来拟合真正的目标序列。推理的目的则是仅通过输入序列来产生目标序列，作为输入传递给解码器的只有输入序列，而没有目标序列。

本篇依然以文本翻译为例进行学习。

0x01 训练
-------

LLM是自回归模型，其只能以串行方式进行预测。而为了提高训练效率，理想的训练方式应该是并行计算：一次性输入整个序列，一次性并行解码把各个位置上的预测全部输出。Transformer通过Teacher Forcing结合掩码来满足这个需求。我们接下来就看看训练中的各个要点具体如何实施。

### 1.1 输入

训练数据由两部分组成：

*   源序列，比如”我喜欢吃苹果“。
*   目标序列，比如”I love apple“。

源序列会输入给编码器。目标序列会输入给解码器。同时，目标序列也被转换为真值标签传递给优化器。我们期望解码器的输出尽可能接近真实标签，因此要优化器要最小化交叉熵。因为是并行操作，所以我们会把源序列拆分之后组装成矩阵，一次性给到Transformer。这样通过矩阵运算实现并行操作，一次即可给出所有序列的预测。但是效果等同于一个一个词输入到编码器进行串行解码。

### 1.2 Dropout

dropout（丢弃）率，即随机丢弃的神经元比例，是一个训练时的超参数，需要根据具体任务进行调整。因为Dropout引入了随机性，因此在测试（或推理）阶段，通常会禁用Dropout，确保所有的神经元都参与到计算中，以获得最稳定的模型输出。

#### 原理

Dropout（正则化）是一种广泛用于机器学习和深度学习的通过给参数增加约束项来限制参数取值范围的方法。它的目的就是阻碍模型过度学习（过拟合），从而提升算法的泛化能力。正则化不仅可以防止模型过拟合，还可以在一定程度上缓解梯度爆炸问题。因为通过给参数增加约束项，可以限制参数在更新过程中的取值范围，从而避免梯度因参数值过大而爆炸。

Dropout概念是Hilton在论文“Improving neural networks by preventing co-adaptation of feature detectors”中提出。如下图所示，实施dropout之后，原始网络相当于变成一个更瘦更稀疏的网络。dropout通过随机丢弃的神经元来削弱节点彼此之间的依赖，这样可以有效的缓解过拟合的发生，在一定程度上达到正则化的效果，有助于模型更快地收敛并提高性能，进而解决深度学习神经网络在用小数据集训练时常见的两大问题：过拟合和训练时间长。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222093643996-185614801.jpg)

如果从集成学习角度来看，每做一次Dropout，相当于从原始的网络中采样得到一个子网络。Dropout对于每个batch的step所优化的参数都不同，每次迭代都相当于训练一个不同的子网络，这些子网络都共享原始网络的部分参数。而且它会不断在这个基础上进行叠加训练。那么，最终的网络可以近似看作集成了若干个不同网络的组合模型。即，Dropout的子网络的平均，提供了一种廉价的近似的Bagging集成。

另外，Dropout 实际上也可以被看作是一种稀疏性表现。论文“On the Effectiveness of Parameter-Efficient Fine-Tuning”就指出稀疏性在模型训练中的两个主要优势：增强模型的鲁棒性和降低泛化误差。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222093655194-966476640.jpg)

Dropout 可以在一定程度上达到这种稀疏性理论分析效果。

#### 位置

Dropout layer 在 Transformer 结构中随处可见，如下图所示，具体分为四种：

*   输入时的dropout（对应图上序号1）。
*   注意力机制中对注意力权重会施加dropout（对应图上序号2）。
*   FFN中两个全连接层之间会施加dropout（对应图上序号3）。
*   "Add & Norm"之间也有dropout（对应图上序号4）。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222093711571-934559338.jpg)

具体对应如下代码片段。

*   在 token embedding，positional encoding 求和之后，有 Dropout。

```python
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x) # 这里用到Dropout
```

*   在注意力中，QKTQKTQK^T 经过 scale、掩码、softmax 得到权重之后，要经过 Dropout 才会与 V 相乘。此时随机“丢弃”一些权重的目的是防止模型过分依赖某些特定的输入。用数学公式展示如下。

Z\=Attention(Q,K,V)\=Dropout(softmax(QKT√dk))VZ\=Attention(Q,K,V)\=Dropout(softmax(QKTdk))VZ = Attention(Q,K,V) = Dropout(softmax(\\frac{QK^T}{\\sqrt d\_k}))V

具体代码如下。

```python
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn) # 这里用到Dropout
    return torch.matmul(p_attn, value), p_attn
```

*   在 FFN 中两个全连接层中间也有 Dropout。

```python
class PositionwiseFeedForward(nn.Module):
    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu())) # 这里用到Dropout
```

*   在每个注意力层和 FFN 层的输出（残差连接之前）都有 Dropout。

```python
class SublayerConnection(nn.Module):
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # 在Norm&Add之中用到Dropout
```

#### 源码

大家可以通过下面PyTorch的源码来了解到Dropout的内部机制。

```cpp
template<bool feature_dropout, bool alpha_dropout, bool inplace, typename T>
Ctype<inplace> _dropout_impl(T& input, double p, bool train) {

  if (p == 0 || !train || input.numel() == 0) {
    return input;
  }

  if (p == 1) {
    return multiply<inplace>(input, at::zeros({}, input.options()));
  }

  at::Tensor b; // used for alpha_dropout only
  auto noise = feature_dropout ? make_feature_noise(input) : at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  noise.bernoulli_(1 - p);
  if (alpha_dropout) {
    constexpr double alpha = 1.7580993408473766;
    double a = 1. / std::sqrt((alpha * alpha * p + 1) * (1 - p));
    b = noise.add(-1).mul_(alpha * a).add_(alpha * a * p);
    noise.mul_(a);
  } else {
    noise.div_(1 - p);
  }

  if (!alpha_dropout) {
    return multiply<inplace>(input, noise);
  } else {
    return multiply<inplace>(input, noise).add_(b);
  }
}

ALIAS_SPECIALIZATION(_dropout,               false, false)
ALIAS_SPECIALIZATION(_feature_dropout,       true,  false)
ALIAS_SPECIALIZATION(_alpha_dropout,         false, true )
ALIAS_SPECIALIZATION(_feature_alpha_dropout, true,  true )
    
Tensor make_feature_noise(const Tensor& input) {
  auto input_sizes = input.sizes();
  std::vector<int64_t> sizes;
  sizes.reserve(input.dim());
  sizes.push_back(input_sizes[0]);
  sizes.push_back(input_sizes[1]);
  for (const auto i : c10::irange(2, input.dim())) {
    (void)i; 
    sizes.push_back(1);
  }
  return input.new_empty(sizes);
}
```

#### 发展

在小模型中可能dropout的效果比较显著，因为小模型针对的是特定领域且少量数据的情况，容易过拟合。而在大模型时代是否需要使用dropout？答案不一。

认为大模型不需要dropout的主要原因有如下几点：

*   因为大模型都是深层结构，以及在训练过程中会使用损失低精度量化计算。使用dropout操作固然可以增加模型的泛化性，但其引入噪声会导致模型训练的不稳定性。
*   使用dropout会导致计算资源的增加和效率的降低，首先要生成一个mask（需要额外显存），然后计算结果也需要存下来（需要额外显存）。反向传播也需要执行额外的逻辑操作等，因此总体效率上肯定是更低的。
*   现在的大模型都是decoder-only的结构，模型中使用了大量的如MQA、多头、pre-norm，residual等技术，且使用到了大量的多领域的数据进行预训练，在某种程度上也增加了泛化性，去掉一个dropout影响不大。

然而某些大模型中也的确依然使用dropout，其作用点依然如下：

*   对自注意力的输出表示进行操作。
*   对MLP输出表示进行操作。

当然其设置会依据大模型的特点进行调整，比如：

*   对于输入层的神经元，其保留率通常设为更接近1的数，使得输入变化不会太大。这是因为对输入层神经元进行丢弃时，相当于给数据增加噪声，以此来提高网络的鲁棒性。
*   对于中间隐藏层的神经元，一般来讲， 设置0.5时效果最好，这对大部分的网络和任务都比较有效。 当 = 0.5时，在训练时有一半的神经元被丢弃，只剩余一半的神经元是可以激活的，随机生成的网络结构最具多样性。
*   输出层一般不加dropout。

而且近期也有把dropout进一步应用的工作，比如论文“LoRA Dropout as a Sparsity Regularizer for Overfitting Control”对 LoRA 矩阵 𝐴 和 𝐵 的输入和输出维度进行随机 Dropout来达到更好的效果。之所以不对 𝑟 的维度进行Dropout，是因为这样会导致矩阵秩的降低，相当于在结构上使用更少的 𝑟，从而削弱模型的表达能力。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222093735979-1240955064.jpg)

### 1.3 损失函数

损失函数通过评估模型预测值与真实值之间的差异来直观地了解模型的预测性能，从而为优化算法提供明确的目标和方向，然后通过最小化损失来逐步优化模型参数。对于自回归语言模型而言，关键之处是看模型能否正确预测到下一个单词，因此优化目标是最小化交叉熵（cross entropy）。这里的交叉熵就是信息熵。或者说，预训练阶段量学习的目标就是最小化各领域的信息熵。

在Transformer架构中，解码器输出后面接了一个模块Generator。该模块的作用是把解码器输出的隐向量从word embedding维度映射到词表长度，得到logits。logits对应着该token取不同字的概率，接下来模型会依据这些概率，按照一定的采样规则来采样下一个token。模型的效果好坏就是看看模型是否可以把下一个token分类到真值对应的token。因此，每次预测新token都是一个分类任务，Generator就是一个分类头。训练会依据分类结果来计算损失。

#### 交叉熵

哈佛代码中使用交叉熵损失函数来比较模型的预测的概率分布（logits）和真实分布（targets）之间的差异。然后对损失计算梯度，用反向传播算法来略微调整所有模型的权重，以便接下来生成更接近结果的输出。具体代码如下。

```python
self.criterion = nn.KLDivLoss(reduction="sum")
```

我们用下图来进行分析。假设词表包含6个单词，我们希望得到与预期的目标序列 "I love you"相符的概率分布。图中上方是目标概率分布。第一个输出词的概率分布中，“I”的概率应该是1，而词表中其它词的概率都应该是0。类似的，第二个和第三个输出词的概率分布中，“love”和"you"的概率都应该是1，词表中其它词的概率都应该是0。图下方则是模型对应预测输出的概率分布。损失函数就是要计算两者之间的差异。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222093753039-801034104.jpg)

计算损失函数的代码如下，传入的参数criterion是损失函数。该类除了包含损失计算外，还包含模型generator部分的前向传播逻辑。下面代码有个正则化的细节，这是为了平滑。假设有两个batch，第一个batch有6个字，则loss是这6个预测结果计算损失的和。第二个batch有60个字，则loss是这60个预测结果计算损失的和。显然第二个损失大，这不符合逻辑。所以我们用除以有效token数目来进行平均。

```python
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        
        self.generator = generator # Generator类对象，依据解码器的输出预测下一个token
        self.criterion = criterion # LabelSmoothing类对象，对标签进行平滑和计算损失

    def __call__(self, x, y, norm):
        """
        x: 解码器的输出
        y: batch.tgt_y，要被预测的所有token，例如src为`<bos>我吃了一个苹果<eos>`，
           则tgt_y是"I ate an apple<eos>"
        norm: batch.ntokens, tgt_y中的有效token数  
        """
        x = self.generator(x) # 生成预测输出
        # 首先使用KLDivLoss进行了损失计算，随后又除以batch.ntokens对损失进行正则化。
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm # 对损失进行正则化
        )
        return sloss.data * norm, sloss
```

#### Label Smoothing

Transformer论文中也使用了Label Smoothing（Label Smoothing Regularization）作为正则化技术来防止过拟合。这么做的原因是因为：现实生活中得到的训练数据是存在噪声的，训练得到的模型也趋向于出现多样性的数据，所以需要在真值中添加噪声，对模型进行约束。下面是论文中的摘录。

> Label Smoothing During training, we employed label smoothing of value ϵls = 0.1 \[36\]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

Label Smoothing主要针对的是softmax层，其思路是：在真值（gound-truth）中加入噪声，即不要把真值完全标记成非0即1，而是用一种概率的方式标记，或者说是对标签做平滑处理，把最高值去掉一些，去掉的这些概率均分给其它人。调整之后，虽然所有类别的概率和仍然是归一的，但是这样可以让模型不那么自信，从而减少过拟合。

Label Smoothing起到的作用实际上是抑制了feature norm，损失函数值曲面上不再存在平缓区域，处处都有较大的梯度指向各个类中心，所以特征会更加聚拢。Label Smoothing的原理如下图所示。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222093803201-441715704.jpg)

我们用实例来进行演示。比如我们的标签是2，词典大小为6。原先的真值向量是：\[0，0，1，0，0，0\]，我们现在取平滑因子ϵ = 0.2，则平滑之后的标签是：\[0.2/5, 0.2/5, 1-0.2, 0.2/5, 0.2/5, 0.2/5\] = \[0.04, 0.04, 1, 0.04, 0.04, 0.04\]。这样可以即使模型预测对了，也不要太自信，而是给模型一点惩罚，防止其过度相信预测结果。

Label Smoothing的代码如下。该类除了负责平滑标签外，还负责计算损失。另外，因为词典包括填充符，而预测时候不应该预测到这个词，所以所以公式中的 K-1也在算法中要变成 K − 2。常见的做法是把 Pytorch CrossEntropy 中 ignore\_index设置为 idx，比如loss\_fn = torch.nn.CrossEntropyLoss(ignore\_index=PAD\_IDX)。

```python
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    # 该类除了平滑标签外，还会计算损失

    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        size: 目标语言词典大小。
        padding_idx: <pad>在词典中对应的序号
        smoothing: 平滑因子，0表示不做平滑处理
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum") # 最终使用的损失函数
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None # 平滑后的标签

    def forward(self, x, target):
        """
        x: generator输出的概率分布。形状为(batch_size, voc_size)
        target: 目标真值标签，内容是token index。形状为(batch_size)
        """        
        # 确保generator的输出维度和词典大小一致，否则后面计算loss的时候就会出错
        assert x.size(1) == self.size
        # 创建一个与x有相同形状的张量
        true_dist = x.data.clone()
        # 将true_dist全部填充为 self.smoothing / (self.size - 2)
        """
        假设 smoothing=0.2，词表大小为6，batch size为2
        则true_dist全部填充为 0.2 / (6-2)= 0.05，此时true_dist为：
        [[0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
         [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]]
        """
        true_dist.fill_(self.smoothing / (self.size - 2)) # K - 2 = 6 - 2
        """
        target.data.unsqueeze(1)会给target.data增加一维，假设target.data是[2,3]，则target.data.unsqueeze(1)的结果是[[2],[3]]
        将true_dist的第一个1维度上与target.data.unsqueeze(1)对应的值变为self.confidence。
        假设此例中target.data.unsqueeze(1) 为[[2], [3]]，即2个数据的标签分别为2，3，就是把true_dist上设置为self.confidence，则true_dist执行过scatter后变为:
        [[0.05, 0.05, 0.8, 0.05, 0.05, 0.05],
         [0.05, 0.05, 0.05, 0.8, 0.05, 0.05]]
        """         
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) # 1代表作用到第一个维度上
        # 将<pad>所在的index填充为0
        true_dist[:, self.padding_idx] = 0
        # 找出target中为<pad>的标签。例如target为['i', 'love', 'you', '<pad>', '<pad>']，mask则为[[3], [4]]，表示第3个和第4个为空格。       
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            # 将"<pad>"所在的label设置为0
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        # 保存平滑标签后的label    
        self.true_dist = true_dist
        """
        使用平滑后的标签计算损失
        由于对`<pad>`部分进行了mask，所以这部分不会参与损失计算
        """        
        return self.criterion(x, true_dist.clone().detach())
```

下图给出了上面代码中的部分数据流程示例。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222093853851-1828725711.jpg)

因为训练是并行执行，难以展示，因此下图进行了简化，只展示前三步中单个输出的损失计算。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222093904878-1018194659.jpg)

具体调用损失函数的精简代码如下，因为前面提到了LabelSmoothing类分装了损失函数，所以这里的criterion就是LabelSmoothing类的实例。

```python
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0) 
model = make_model(V, V, N=2)
batch_size = 80
for epoch in range(20):
    model.train()
    run_epoch(
        data_gen(V, batch_size, 20),
        model,
        SimpleLossCompute(model.generator, criterion),
        optimizer,
        lr_scheduler,
        mode="train",
    )

# run_epoch()函数中会调用损失函数
def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        # 计算损失
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)   

```

下面是另一个平滑的例子，从该例子可以看到当模型非常自信的时候就会给予其一个微小的惩罚，越自信，损失反而越大。

```python
def loss(x, crit):
    # x是从0到100的一个不断增大的数。 d=x+3，比x大一点。
    d = x + 3
    """
    模拟模型的输出。
    一开始x为1，输出为：[[0.0000, 0.2500, 0.2500, 0.2500, 0.2500]]，此时模型还不太会预测
    当x到100时，输出为：[[0.0000, 0.9706, 0.0098, 0.0098, 0.0098]]，此时模型很自信的说结果就是 1
    """
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    # 计算模型损失。由于使用的是KLDivLoss，所以要对predict进行log操作
    return crit(predict.log(), torch.LongTensor([1])).data

def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            # x从1开始不断增大，模拟模型的表现越来越好
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )

show_example(penalization_visualization)

```

### 1.4 学习率

学习率决定了模型参数更新的步长。如果学习率设置得过高，那么模型参数在更新时可能会因为步长过大而跳出最优解的范围。同时，过高的学习率会使模型在更新参数时过于激进，从而加剧梯度的波动，导致梯度爆炸。如果学习率过低，模型收敛速度可能会变慢，训练时间变长。因此，学习率的选择需要根据具体任务和模型结构进行调整。在实际应用中，可以使用自适应学习率算法来根据参数梯度的统计信息来调整学习率。例如，Adam、Adagrad、RMSprop等优化算法都可以根据梯度的历史信息来动态调整学习率，从而提高训练的稳定性和效率。

#### Warmup

Warmup（热身）方案也属于动态调整学习率的一种。具体是指在训练开始阶段，将学习率从 0 缓增到指定大小，而不是一开始就从一个指定大小开始训练。如果不进行Warmup，则模型可能在训练开始就快速学习。因为梯度消失的原因，模型对越靠后的层越敏感，也就是越靠后的层学习得越快。然而，靠后的层是以考前层的输出为基础进行学习，如果前面层没有学习好，靠后层的学习就会建立在错误基础上，最终导致模型崩盘。

#### Noam

Transformer论文使用了一种特殊的自适应学习率调整策略，称为“Noam”学习率预热策略。它包括warmup（热身）和decay（衰减）两个部分，总体趋势是学习率先增加再减少。“Noam”学习率预热策略具体如下图所示，是一个以warmup\_steps为分界点的分段函数。其中dmodeldmodeld\_{model}是模型维度，step\_num是当前训练步数，armup\_steps是预热部署。

*   warmup阶段：从0到warmup\_steps是热启动阶段，此时先让学习率线性增长到某个最大的值。大型网络在训练初期尚不稳定，较大学习率会增加收敛难度。warmup阶段用较小的学习率可以有助于模型在训练初期快速收敛。而且大型网络往往使用超大的批量大小（batch size），为了实现超大批量大小，需要保证“k 个 minibatch , size = n , lr = η” 和 “1 个 minibatch , size = kn , lr = kη”的梯度近似相等。但是在模型变化剧烈时，这个等式会被打破。warmup 可以有效缓解这个问题。
*   decay阶段：冷却阶段，此时让学习率按指数的方式衰减。这样可以在训练后期通过减小学习率来让模型稳定训练。常用方法有指数衰减（exponential）、分段常数衰减（piecewise-constant）、反时限衰减（inverse-time）等等。Transformer 采用了负幂形式，衰减速度先快后慢。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222093919676-1131543537.jpg)

Noam机制主要是受人类的学习机制启发：每当我们学习一个新的的领域的时候，刚开始需要摸索入门，不断尝试，此时训练速度很慢；随着吸收基础知识增多，我们学习速度会渐渐加快；当掌握了大量的比较杂的知识之后，我们一般会遇到一个瓶颈期，需要知识整合和感悟，速度又会变慢下来。总结一下，人类学习能力是一个螺旋式的渐进过程，是慢与快的交叉过程。Noam机制就是这个进程的具体体现。

下图给出了具体推导过程。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222093932022-303106393.jpg)

哈佛代码中rate()函数就是对下面公式的实现。

lrate\=d−0.5model⋅min(step\_num−0.5,step\_num⋅warmup\_steps−1.5)lrate\=dmodel−0.5⋅min(step\_num−0.5,step\_num⋅warmup\_steps−1.5)lrate = d\_{\\text{model}}^{-0.5} \\cdot \\min({step\\\_num}^{-0.5}, {step\\\_num} \\cdot {warmup\\\_steps}^{-1.5})

具体代码如下。

```python
def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0: # 如果未提供步数，则设为1
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


```

具体使用方式如下。

```python
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
)
lr_scheduler = LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda step: rate(
        step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
    ),
)

```

在实际应用中，也可以用不同的学习率调整每一层，或者把若干层分为一组，对于不同组应用不同的学习率。这是因为 Transformer 模型中不同层通常捕获不同类型的信息，底层通常编码通用和基础的信息，顶层通常编码更接近预训练任务的信息，因此可以对顶层应用较高学习率而对底层应用较低学习率。

### 1.5 初始化

权重初始化是神经网络训练的重要步骤之一。如果权重初始化过大，那么在反向传播过程中，梯度的计算会受到很大的影响，容易导致梯度爆炸。例如，如果权重由标准正态分布初始化，其期望数量级为1，那么在多层传播后，梯度值可能会变得非常大。

使用合适的权重初始化策略可以有效控制梯度的大小，减少梯度爆炸的可能性。常见的权重初始化方法包括Xavier初始化（也称为Glorot初始化）和He初始化。这些方法根据网络的层数和激活函数的特点来设置权重的初始值，使得在反向传播过程中梯度的变化更加平稳。

例如，Xavier初始化方法根据输入和输出神经元的数量来调整权重的初始值，使得前向传播和反向传播中的激活值和梯度值保持相近的方差。He初始化方法则特别适用于ReLU激活函数，因为它考虑了ReLU激活函数在零点的不连续性，从而更加准确地设置了权重的初始值。

vanilla Transformer使用的就是Xavier初始化。

### 1.6 Teacher Forcing

本质上来讲，Teacher Forcing 是一种引导和加速模型学习过程的方法，在训练的每一步都为其提供正确的输作为指导，而不是让训练根据之前的输出来生成下一步。

#### 问题

前文提到过，自回归推理有两个问题：

*   容易累积错误，导致训练效果不佳。在训练时，我们可以使用与推理时相同的方法，即用自回归模式进行。然而这样整个模式就是串行化过程，如果编码器在某一轮预测错了，那么这个错误的输出就会作为下一轮解码器的输入，这样基于错误输入继续解码就是在错误道路上越走越远，这将导致模型向全局最优收敛的速度减慢。
*   只能以串行方式进行，这就意味着很难以并行化的方式开展训练以提升效率。这种现象和人说话的逻辑是相似的，人也许可以在脑中构思整个句子，但是表述一定是一个词一个词说出来的，而且后面的词一定会被前面的词所影响，这就是说话的逻辑。

我们用下面表格来看看上面两个问题。

*   首先，对于所提供的输入，模型必须经过5个时间步才能完成推理，因为Decoder每一次只会预测一个单词。但是，按照上述流程进行训练会过于缓慢，我们应采用并行（矩阵计算）的方式去训练。
    
*   其次，推理步骤中会出现错误，而且容易在错误道路上越走越远。
    

| 时间步 | 解码器输入1 | 解码器输入2 | 解码器输出 | 真值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| 1 |  | "我吃了一个苹果"编码后的隐向量 | I | I | 预测正确 |
| 2 | I | "我吃了一个苹果"编码后的隐向量 | like | ate | 预测错误 |
| 3 | I like | "我吃了一个苹果"编码后的隐向量 | play | an | 预测错误 |
| 4 | I like play | "我吃了一个苹果"编码后的隐向量 | football | apple | 预测错误 |
| 5 | I like play football | "我吃了一个苹果"编码后的隐向量 |  |  | 预测正确，但是没啥用处 |

#### 概念

为了提升训练效率，我们需要用并行手段来保证在一次训练中输出一个序列中所有的单词的预测结果。为了实现这种理想训练方式，研究人员提出了Teacher Forcing（教师强制训练），这种技术可以通过在训练时向解码器输入整个目标序列来一次性并行解码全部输出。

具体来说，Teacher Forcing就是每次推理给解码器输入时，不使用前次推理的输出作为下一次推理的增加输入，而是使用训练标签的真值（ground truth）作为下一次推理的增加输入。Teacher Forcing机制保证了 Transformer 在训练阶段可以并行地输出所有的词，而不需要循环，这大大加快了训练速度。这种模式具体如下图所示，图中简化了输入，实际上解码器的输入是一个拼接，而非单纯输入某个标签。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222093947380-201646227.jpg)

向解码器提供目标序列实际上是给了模型一个正确指导，即使上一个词预测错误，在下一时间步，它也可以用正确的第一个词（即真值）来预测第二个词，这就避免了错误的持续累加，可以保证对每次推理的监督训练都是从正确的输入出发，因而可以期待正确的结果。其名称中的“Teacher”指的就是真值，自回归模式是“靠自己”进行训练，Teacher Forcing就是有老师带着做训练，即使我们计算出错误的答案，老师也会为我们提供问题的正确答案。我们可以知道是在哪个阶段出现问题，从而很容易地分析自己的错误，更好更快地学习，即“靠标准答案”指导来进行训练。

注：与 Teacher forcing 模式相对的是 free-running 模型。free-running是直接用上一个状态的输出，来作为下一个状态的输入。

#### 示例

我们假设要把“我吃了一个苹果”翻译成“I ate an apple”。“我吃了一个苹果”是编码器的输入，“I ate an apple”是真值标签。我们看看模型是如何利用Teacher Forcing模型在训练中纠正错误，防止错误的累积，从而提高训练效果。

首先，真值标签是目标序列，会作为解码器的输入。为了做更好的训练，我们要把输入的所有token向右移一个位置（Shifted Right），然后在最左边放上一个表示开始的token（）。与之对照，自回归模型本时刻的输入是上一时刻自己输出的值，该值是上一时刻预测出来的，不一定正确；而Teacher Forcing本时刻的输入是上一时刻的真值标签，这肯定是正确的，可以确保解码器本次预测是基于正确基础上进行。

其次，我们再看看解码过程中的历次推理。可以看到，如果第二步预测之后，模型接受了“like”，会导致模型在后续训练中偏离正轨，导致学习速度变慢，模型也变得不稳定。在Teacher Forcing模式中，因为发现了错误，模型会丢弃这个输出，把“ate“作为下一次的输入。或者说，在训练时，不管解码器本次输出是什么，它下次的输入都是本次输出对应的真值。这样模型将更正训练过程中的统计属性，增加了后续单词成功预测的几率，从而更快地学会生成正确的序列。

| 时间步 | 解码器输入1 | 解码器输入2 | 解码器输出 | 真值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| 1 |  | "我吃了一个苹果"编码后的隐向量 | I | I | 预测正确 |
| 2 | I | "我吃了一个苹果"编码后的隐向量 | like | ate | 预测错误，用真值纠正 |
| 3 | I ate | "我吃了一个苹果"编码后的隐向量 | an | an | 预测正确 |
| 4 | I ate an | "我吃了一个苹果"编码后的隐向量 | orange | apple | 预测错误，用真值纠正 |
| 5 | I ate an apple | "我吃了一个苹果"编码后的隐向量 |  |  | 预测正确 |

具体对应下图所示。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222094002661-492886186.jpg)

图片思路来源 ：[解剖Transformer 第二部分：你会用注意力机制组装出一个Transformer吗？](https://zhuanlan.zhihu.com/p/552573482) [大方](https://www.zhihu.com/people/anthony_shi)

#### 原理

实质上，Teacher Foring是在训练过程中去掉了每次推理的序贯关系，使得原先自回归推理先后依赖被解除，解码器的输入就是真值标签，因此具备了并行推理的可能。我们可以将整个句子”I ate an apple“复制5次构成一个矩阵，使得矩阵每一行代表一个时间步的输入，然后把矩阵作为一个批量一次性输入给解码器，这样就可以利用GPU的并行能力，一次并行做5次推理来得到所有时间步的结果。然后对每个对输出序列的每个元素都计算损失即可，这就是Transformer训练时可以并行计算的原因。

在执行的过程中，我们在初始输出中添加了起始符`<bos>`，相当于将输出整体右移一位（Shifted Right）。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222094017577-1955721626.jpg)

对应到具体数据构建，训练代码会先把目标句子扩展为" I ate an apple"，然后向右移动一位构建标签"I ate an apple"。再构建一个批量如下：

```python
<bos>I ate an apple
<bos>I ate an apple
<bos>I ate an apple
<bos>I ate an apple
<bos>I ate an apple

```

最后把这个批量传给解码器。

#### 掩码

虽然上述的并行可以一次性计算所有时间步对应的输出，但是却存在一个问题，即注意力在预测某个词时可以提前关注到其后面的单词，从而模型学会作弊。比如上图的每个推理步的输入都是”I ate an apple“这整个句子。所以在预测第一个输出”I“时，模型实际上可以关注到目标序列中”“后面的单词，模型直接输出”I“就可以满足需求。这样的偷窥行为会让模型学会偷懒而不是学习规律。也就是说，仅仅使用Shifted Right，并不能实现teacher forcing，这是因为如果注意力模块是没有mask的self-attention，就会造成数据泄漏的问题。

因此人们引入了掩码机制来隐藏未来信息。具体做法时在计算注意力时加入一个掩码（mask）该掩码是一个跟输入矩阵一样形状的矩阵，其作用就是遮掉输入矩阵的一部分，让模型只能看到目标序列的一部分（前缀）：在输出第i个元素的时候，不能看目标序列的第i个元素及其后面的部分，只能用到第i个元素之前的信息，从而切断它从未来获得信息的通路，不能泄露天机（把对应的注意力强制置零），这样才能在训练时候模拟实际推理的效果。或者说，通过掩码可以单独调节每一个源元素与每一个目标元素之间的注意力强度。

在训练时，假如Decoder当前的输入为" I ate an apple"，对于单词an 来讲，只需要让其关注自身及I 和ate 即可，后面的apple作为我们将要预测的单词，此时还未出现，因此不用去关注。下图是四个单词各自应该关注的情况。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222094028394-266098203.jpg)

下图是加入了掩码之后的Teacher Foring示例。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222094041178-501859248.jpg)

#### 实现

Teacher Forcing的实现相对简单，就是传入了目标序列，用真实目标序列和掩码作为输入来指导解码器的生成过程。

```python
for i, batch in enumerate(data_iter):
    out = model.forward(
        batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
    )
    loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

```

而损失函数是把所有的out放在一起，然后看损失。

```python
class SimpleLossCompute:
    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss

```

#### 优劣

Teacher Forcing的优势是因为模型是在“正确答案”指引下进行预测，训练的稳定性得到大大增强，收敛速度也得以大幅提升。而且我们可以一次性的输入全部目标序列，然后以并行的方式一次性的输出完整的目标序列，训练效率大幅提升。

但是Teacher Forcing也存在一定的问题。因为训练可以“靠老师”，推理还得“靠自己”，这样推理时遇到的错误输出对于下次推理来说就是在训练数据分布之外（out of distribution）的异常输入，所以会导致用Teacher Forcing模式训练出来的模型在训练环节和预测环节存在行为差异。这种因为训练和推理之间数据分布存在差异，导致模型在部署中表现变差的现象叫做exposure bias（曝光误差）。另外，因为模型生成的结果都必须和参考句一一对应。这种约束在训练过程中减少模型发散，加快收敛速度。但是一方面也扼杀了翻译多样性的可能。

因此研究人员也针对exposure bias做了一些改进工作。比如其中一个变种是Curriculum Learning，它的思路是：既然自回归模式的全靠自身预测结果和Teacher Forcing模式的全靠真值均不可取，那么就不如折中方案，进行有计划的学习。在训练过程的每一步会以一定的概率随机选择是用模型输出还是用真值。上述选择概率是随着训练的推进不断调整的：训练过程会从Teacher Forcing开始，慢慢降低在训练阶段输入真值的频率。即一开始学生是小白，只能老师带着学，后续随着学生的进步，老师慢慢放手让学生自主学。

#### 小结

训练流程是之所以是一步操作，这是因为ground-truth已知，而结合Teacher Forcing和掩码，能够让第i个时间步的上下文向量只是由前i个时间步的向量计算而来，这样就保证了第i个单词的预测只使用了前i个时间步的信息，即提高了计算效率，又符合语言模型的内在规律。

### 1.7 并行

我们接下来看看训练时候的并行机制。总的来说，Transformer的并行化主要体现在训练阶段，特别是在自注意力和FFN中。

在推理阶段，因为ground-truth未知，我们只知道前i-1个时间步预测的单词，显然只能使用自回归迭代操作来预测所有的单词，所以需要多步来预测序列中所有的单词，难以并行。尽管Decoder端在推理阶段的并行化存在挑战，但通过一些先进的技术和模型变种，这个问题也可以得到一定缓解。

#### 逻辑维度

我们首先从seq2seq模型维度来看看。 encoder-decoder 架构是自回归的：通过上一步产生的token和这一步的输入来预测这一步的输出。我们看看Transformer对此做了哪些改进从而完成并行。

##### 编码器

编码器天然支持并行。整个架构从原来的序列模型变成了一个全连接图模型，每个词之间是可以直接关联的，这就很方便的进行矩阵计算，从而享受并行计算或者 GPU 加速带来的运行效率的提升。

下图是使用“北国的特产”为例来看Transformer计算时候的信息流。注意力机制的感知域是整个句子，Transformer在计算任意一个词的特征时，会用到所有词的信息，即”北“的特征L(北)是由所有词共同计算得到的。L(北)是所有单词的加权和，这样就没有距离的概念，不会有长依赖的问题，即序列中任意两个单词之间距离都是一个固定的常数。另外，输入序列中每个位置的单词都各自单独的路径流入编码器，所有单词可以同时流入编码器中，不需要排队进入，这样就可以进行并行处理。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222094053709-1658058776.jpg)

##### 解码器

在自回归模式下，解码器需要两种隐向量：

*   编码器生成的编码隐向量。
*   解码器在解码过程中产生的隐向量，即上一状态的输出。

对于第一种隐向量，编码器通过并行操作可以一次性计算出来，传递给交叉注意力。对于第二种隐向量，在Teacher Forcing模型下，每次推理的序贯关系被打破，原先自回归推理先后依赖被解除，不再需要解码过程中的隐向量。所以配合掩码，我们可以把全量输入和真值标签一次性直接投入到解码器中来完成并行训练。

#### 模型维度

模块即指编码器，也指解码器。因为在训练中，编码器解码器都可以并行。从模型来看，以下维度可以并行。

*   Q、K、V生成可以并行化。使用WQWQW^Q，WKWKW^K，WVWVW^V这三个权重矩阵的计算过程可以并行化。
*   自注意力机制的并行化。在自注意力层中，模型计算输入序列中所有位置的单词之间的注意力分数，并且这些计算是相互独立的。因此，它们可以在不同的处理单元上并行执行。
*   多头注意力机制的并行化。多头注意力机制中，不同的注意力头可以在不同的处理单元上并行计算。
*   FFN的并行化。FFN对输入序列的每个位置执行相同的操作，并且这些操作是独立的。因此，它们也可以在不同的处理单元上并行执行。

##### 自注意力

编码器可以并行的关键是在自注意力机制中，计算ZiZiZ\_i要依赖全部元素x1,...,xnx1,...,xnx\_1,...,x\_n，而非依赖Zi−1Zi−1Z\_{i-1}。参见下图，以”吃了“这个token为例，自注意力机制利用输入元素两两之间的相关性作为权重，然后加权求和把每一个输入元素xixix\_i映射到语义向量ziziz\_i。ziziz\_i是考虑了全局依赖之后的产物，不需要严格时序依次迭代，能够并行，因此我们可以用矩阵运算一下子把所有的ziziz\_i计算出来。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222094105338-1438300873.jpg)

##### FFN

输入序列中每个位置的单词都按照各自单独的路径流入编码器，即各个单词同时流入编码器中，不是排队进入。  
在自注意力self-attention层中，这些路径两两之间是相互依赖的，而FFN则没有这些依赖性，所以这些路径在流经FFN时可以并行计算。

对于输入序列中的每个位置 x 会使用相同的变换矩阵来计算，且每个子层使用的不用的参数。在计算完Multi-Head Attention后，FFN层的输入矩阵为 X∈Rdinput×dmodelX∈Rdinput×dmodelX∈R^{d\_{input} \\times d\_{model}}，可以看作是由每个输入位置（ dinputdinputd\_{input} 行）的attention结果（ dmodeldmodeld\_{model} 列）堆叠而成。这些行进行相同的线性变换后，维度改变，重新堆叠成FFN层的输出。行与行之间无交错，完全是“separately and identically”，按位置进行变换。论文的3.3小节Position-wise Feed-Forward Networks中，对“Position-wise”做了注解，如下图所示。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222094115267-312393953.jpg)

#### 张量维度

输入到 Transformer 的 Tokens 有 `batch_size`、`sequence_length`、`embedding_dim` 三个维度，而 Attention 计算的 multi head 机制把 `embedding_dim` 维度再拆分为 `head_num` 个 `head_dim`，因此从计算量角度来看一共有五个维度：

*   batch\_size
*   sequence\_length
*   token
*   head\_num
*   head\_dim

其中 batch\_size、head\_num和token这三个维度本身就支持并行。而人们最近也在序列维度上进行了并行尝试，即序列并行。序列并行首先由论文"Sequence Parallelism: Long Sequence Training from System Perspective"提出，目的是要解决序列长度过长导致内存使用量过大的问题，我们知道LLM推理主要有两个阶段：prefill和decode。前者瓶颈在于计算，而后者在于带宽。在prefill中已经有将sequence length拆开计算再汇总的做法，序列并行则是将这个过程并行完成，具体是把输入序列切分为多个块，每个块放到不同GPU上进行计算，以减少长序列输入对显存大小的需求。为了合并计算结果，论文也提出了环自注意力（RSA）机制。

另外，也有一种说法叫做上下文并行 Context Parallelism，其最先出现在NVIDIA Megatron-Core中，上下文并行主要是针对self-attention（Linear，LayerNorm）进行优化，它将原本的输入按照sequence length维度拆开，分到不同的device上，分别计算，然后通过all-gather和reduce-scatter通信操作来整合其他device上算出的结果。

### 1.7 代码

#### 训练方式

train\_model()函数会依据配置选择是进行分布式训练还是单机训练。

```python
def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    from the_annotated_transformer import train_worker

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )

def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model( # 分布式训练
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    else:
        train_worker( # 使用0号GPU进行单机训练
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )


```

#### 单机训练代码

单机训练代码如下，它遍历一个 epoch 的数据，然后调用 forward()函数，接着用 loss\_compute() 函数计算梯度，更新参数并且返回 loss。这里的 loss\_compute() 函数的输入是：模型的预测 out，真实的标签序列 batch.trg\_y 和 batch 中词的个数。

```python
def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src, # 源语言词典
    vocab_tgt, # 目标语言词典
    spacy_de, # 源语言分词器
    spacy_en, # 目标语言分词器
    config,
    is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)
 
    pad_idx = vocab_tgt["<pad>"] # 得到目标语言词典中"<pad>"所对应的索引
    d_model = 512 # 词嵌入大小
    model = make_model(len(vocab_src), len(vocab_tgt), N=6) # 构建一个6层模型
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    # 构建损失函数
    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    # 构建数据加载器
    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    # 构建优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    # 构建学习率策略，依据配置来设定warmup参数
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)

```

#### 总体代码

总体代码如下所示，里面包含了训练和使用训练好的模型进行推理。

```python
def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))

```

0x02 推理
-------

当我们谈论大型语言模型（LLM）的推断过程时，我们指的是使用已经训练好的模型来对输入文本进行处理，从而生成相应的输出。因为训练和推理的基本流程类似，所以我们在此处主要看两者的差异点和推理的独有特性。

### 2.1 输入输出

首先，无论是训练和推理，编码器模块的输入（要被翻译的句子）和执行过程都相同。

其次，对于解码器来说，训练和预测的执行过程存在不同。

*   目标输入不同。虽然都是把新词加到之前的输入上，拼接成解码器的新输入，但是新词的来源不同。
    *   推理阶段是用上一次的输出拼接成下一次输入。即tgt是从开始，然后每次加入上一次的输出。
    *   训练阶段时时用真值拼接成下一次输入。即tgt是从开始，然后每次加入下一个真值。而且是一次性把输入序列全部传给解码器。
*   输出不同。
    *   推理：每次输出一个新token。Decoder的并行化仅在训练阶段，在推理阶段，因为我们没有正确的目标语句，t时刻的输入必然依赖t-1时刻的输出，这时跟之前的seq2seq就没什么区别了。
    *   训练：transformer会一次输出多个概率分布。在训练时，得到输出概率分布后就可以计算loss了，并不需要将概率分布再转成对应的token。

### 2.2 流程

在推理时，因为在预测场景下不存在答案文本，只能从位置开始预测，因此预测场景的解码器必定是串行的循环输出。在后续步骤的预测中，会将前一个时间步的输出序列和历史预测单词一齐送到下一个时间步的解码器，直到遇到句末标记。 与Seq2Seq模型的区别在于，在每个时间步，我们重新输入迄今为止生成的整个输出序列，而不仅仅是最后一个单词。

推理过程的逻辑流程如下

| 时间步 | 解码器输入1 | 解码器输入2 | 解码器输出 |
| --- | --- | --- | --- |
| 1 |  | "我吃了一个苹果"编码后的隐向量 | I |
| 2 | I | "我吃了一个苹果"编码后的隐向量 | ate |
| 3 | I ate | "我吃了一个苹果"编码后的隐向量 | an |
| 4 | I ate an | "我吃了一个苹果"编码后的隐向量 | apple |
| 5 | I ate an apple | "我吃了一个苹果"编码后的隐向量 |  |

对应的逻辑图如下。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222094131971-366479802.jpg)

训练过程就是简单地在上述推理过程的基础之上加上对每次推理预测的新元素的监督即可，具体见下图。

注意：下图只是为了展示流程，实际上是一并输入，并行预测。

![](https://img2024.cnblogs.com/blog/1850883/202502/1850883-20250222094141063-809544615.jpg)

最后，我们总结训练和推理在流程上的区别如下表。

| 步骤 |  | 训练 | 推理 |
| --- | --- | --- | --- |
|  | 输入 | 源语言序列 + 目标语言序列（真值） | 源语言序列 + 目标语言序列（预测的输出） |
| 1 | 编码器处理 | 产生整个源语言序列的编码表示 | 产生整个源语言序列的编码表示 |
| 2 | 解码器处理输入 | 目标序列首先加一个句首标记，被转换成嵌入后送入解码器。 | 在第一个时间步使用仅包含句子开头标记的空序列，而非目标序列。后续时间步会输入迄今为止生成的整个输出序列。序列被转换成嵌入后送入解码器。 |
| 3 | 解码器解码 | 解码器将目标嵌入与编码器的编码表示一起处理，生成目标序列的解码表示 | 解码器将目标嵌入与编码器的编码表示一起处理，生成目标序列的解码表示 |
| 4 | 解码器处理输出 | 输出层将目标序列的编码表转换为单词概率和最终输出序列 | 输出层将目标序列的编码表转换为单词概率和最终输出序列 |
| 5 | 计算损失 | 损失函数将此输出序列与训练数据中的目标序列进行比较，计算损失 | 无 |
| 7 | 迭代 | 无迭代，一次性处理完毕 | 迭代运行2~4，逐步输出token |

### 2.3 代码

下面是哈佛源码中的推理测试代码。

```python
# ## Inference:
#
# > Here we make a forward step to generate a prediction of the
# model. We try to use our transformer to memorize the input. As you
# will see the output is randomly generated due to the fact that the
# model is not trained yet. In the next tutorial we will build the
# training function and try to train our model to memorize the numbers
# from 1 to 10.
def inference_test():
    # 构建，源词典和目标词典大小都为11，
    # EncoderLayer和DecoderLayer的数量为2
    test_model = make_model(11, 11, 2)
    test_model.eval()
    # 输入形状为(1, 10)，即一个句子，该句子10个单词。
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # 定义源序列掩码，即所有的词都是有效的，没有填充词
    src_mask = torch.ones(1, 1, 10)

    # 将输入送给编码器，获取输出，记作memory
    memory = test_model.encode(src, src_mask)
    # 初始化ys为[[0]]，用于保存预测结果，其中0表示'<bos>'
    ys = torch.zeros(1, 1).type_as(src)

    # 循环调用解码器来预测下一个token。例如：假设我们要将“I love you”翻译成
	# “我爱你”，则第一次的`ys`为(<bos>)，然后输出为“I”。然后第二次`ys`为(<bos>, I)
	# 输出为"love"，依次类推，直到decoder输出“<eos>”或达到句子长度。
    for i in range(9): 
        # 将编码器的输出memory和之前解码器的所有输出作为参数，让解码器来预测下一个token
        out = test_model.decode(
            # ys就是Decoder之前的所有输出
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # 将Decoder的输出送给generator进行预测。这里只取最后一个词的输出进行预测。
        # 因为传入tgt的词数是变化的，第一次是(<bos>)，第二次是(<bos>, I)
        # 所以输出out的维度也是变化的，变化的就是(batch_size, 词数，词向量)中词数这个维度
        prob = test_model.generator(out[:, -1])
        # 取出数值最大的那个token，它的index在词典中对应的词就是预测结果
        _, next_word = torch.max(prob, dim=1)
        # 取出预测结果
        next_word = next_word.data[0]
        # 将这一次的预测结果和之前的拼到一块，作为之后Decoder的输入
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)

def run_tests():
    for _ in range(10):
        inference_test()

show_example(run_tests)

```

0xFF 参考
-------

[A Contrastive Framework for Neural Text Generation (Su et al., 2022)](https://arxiv.org/abs/2202.06417)  
[A Survey on Efficient Inference for Large Language Models](https://arxiv.org/abs/2404.14294)  
[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)  
[Breaking the Sequential Dependency of LLM Inference Using Lookahead Decoding (Fu et al. 2023)](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)  
[ChatGPT是第一个真正意义的人工通用智能](https://mp.weixin.qq.com/s?__biz=MzI2MjU4MDYwOA==&mid=2247484251&idx=1&sn=a3ed8006077c2d21e1c7829c11f021c2&scene=21#wechat_redirect)  
[Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2022)](https://arxiv.org/abs/2211.17192)  
[https://arxiv.org/abs/1801.06146](https://arxiv.org/abs/1801.06146)  
[https://arxiv.org/abs/1803.05407](https://arxiv.org/abs/1803.05407)  
[https://arxiv.org/abs/2006.05987](https://arxiv.org/abs/2006.05987)  
[https://github.com/1311440131/deep\_blue\_writings/tree/main/2021\_9\_18\_%E5%BE%AE%E8%B0%83Transformer%E7%9A%84%E9%AB%98%E7%BA%A7%E6%8A%80%E6%B3%95](https://github.com/1311440131/deep_blue_writings/tree/main/2021_9_18_%E5%BE%AE%E8%B0%83Transformer%E7%9A%84%E9%AB%98%E7%BA%A7%E6%8A%80%E6%B3%95)  
[https://medium.com/@plienhar/llm-inference-series-1-introduction-9c78e56ef49d](https://medium.com/%40plienhar/llm-inference-series-1-introduction-9c78e56ef49d)  
[https://medium.com/@plienhar/llm-inference-series-2-the-two-phase-process-behind-llms-responses-1ff1ff021cd5](https://medium.com/%40plienhar/llm-inference-series-2-the-two-phase-process-behind-llms-responses-1ff1ff021cd5)  
[https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/](https://pytorch.org/blog/ch-1.6-now-includes-stochastic-weight-averaging/)  
[https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging](https://pytorch.org/docs/stable/.html%23stochastic-weight-averaging)  
[LaViT：这也行，微软提出直接用上一层的注意力权重生成当前层的注意力权重 | CVPR 2024](https://mp.weixin.qq.com/s?__biz=MzkyMDE2OTA3Mw==&mid=2247531933&idx=2&sn=27fa53abb35fbec186cd3801983779f8&chksm=c060731f8427a4c6d32de787447d8604fecf0739f6dcacc599153285a7c4683cf9dfb590d239&mpshare=1&scene=1&srcid=0821UThfJG2X8cWUw2eUOuwb&sharer_shareinfo=41b3507fb5f2282a9516d6c002e13797&sharer_shareinfo_first=41b3507fb5f2282a9516d6c002e13797#rd) VincentLee  
[LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/pdf/2402.16363)  
[LLM的几种并行机制](https://zhuanlan.zhihu.com/p/688687624) [认输你就真了](https://www.zhihu.com/people/mrdoghead)  
[LoRA Dropout as a Sparsity Regularizer for Overfitting Control](https://arxiv.org/pdf/2404.09610)  
[nn.KLDivLoss\_咕噜咕噜day的博客-CSDN博客\_kldivloss pytorch](https://blog.csdn.net/qq_36533552/article/details/104034759)  
[On the Effectiveness of Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2211.15583)  
[Pytorch：交叉熵损失(CrossEntropyLoss)以及标签平滑(LabelSmoothing)的实现\_我是大黄同学呀的博客-CSDN博客\_标签平滑交叉熵](https://blog.csdn.net/qq_36560894/article/details/118424356)  
[The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) [Jay Alammar](https://jalammar.github.io/)  
[Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/abs/2312.15234)  
[《Rethinking the Inception Architecture for Computer Vision》](https://arxiv.org/abs/1512.00567)  
[万字逐行解析与实现Transformer，并进行德译英实战（一）](https://blog.csdn.net/zhaohongfei_358/article/details/126085246)  
[万字逐行解析与实现Transformer，并进行德译英实战（三）](https://blog.csdn.net/zhaohongfei_358/article/details/126085598)  
[万字逐行解析与实现Transformer，并进行德译英实战（二）](https://blog.csdn.net/zhaohongfei_358/article/details/126085557) [iioSnail](https://blog.csdn.net/zhaohongfei_358)  
[大模型参数微调：Sparsity和Dropout](https://zhuanlan.zhihu.com/p/2893564859) [Chongjie](https://www.zhihu.com/people/89-47-66-58-20)  
[大模型时代是否还需Dropout，一次关于GLM4-9B-Chat的分析](https://zhuanlan.zhihu.com/p/701877920) [LeonYi](https://www.zhihu.com/people/lei-yi-43-4)  
[实现 pytorch 中 torch.nn.CrossEntropyLoss\_Agwave的博客-CSDN博客](https://blog.csdn.net/qq_41805511/article/details/99438838)  
[解剖Transformer 第二部分：你会用注意力机制组装出一个Transformer吗？](https://zhuanlan.zhihu.com/p/552573482) [大方](https://www.zhihu.com/people/anthony_shi)  
Yoshua Bengio, Rejean Ducharme, Pascal Vincent, and Christian Jauvin. A neural probabilistic language model. Journal of Machine Learning Research (JMLR), 3:1137–1155, 2003. \[PDF\]

本文转自 <https://www.cnblogs.com/rossiXYZ/p/18730583>，如有侵权，请联系删除。