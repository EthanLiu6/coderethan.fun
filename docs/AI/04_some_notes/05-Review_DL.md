# 回顾和复习整理DL



## 00. python基础（未补）

> 抽空补充

- 迭代器
- 装饰器
- callable
- 



## 01. 激活函数🌟🌟🌟

### 1. 我的记录：

- 为啥需要Activation？（深度学习往往只说非线性Attention）

    > 都是Linear（或者Covn）的话网络还是线性的，多层和一层没啥太大区别。

- 有哪些类型？（工程上主要类别）

    > 一种是point wise的操作（或者叫做element wise操作），也就是逐元素操作；另一种是具有相关性操作
    
    > > Note: 公式、图像、导数图像、优缺点……
    >
    > A类：数据间独立
    >
    > 1. S型：
    >
    >     - sigmoid（0.5的二分类问题）
    >
    >     - Tanh（NLP的递归系列常用）
    >
    > 2. ==ReLU==：AlexNet里面提出。
    >
    >     - ReLU
    >
    >     - 缓解激活值指数性增长=>ReLU6
    >     - 缓解神经元坏死现象=>Leakey ReLU=>PReLU/RReLU
    >     - x=0处平滑过渡=>ELU/SELU
    >     - ==GeLU==（工程上有点简化，现在大模型都会用他的变种）
    >
    > 3. Swish：
    >
    >     - Switch（对sigmoid进行修改）
    >     - Hard Swish（对ReLU6做修改，计算更简单）。“凑图像”
    >
    > 4. mish：平滑性更高（看一阶和二阶导图像）
    >
    > B类：数据间有相互作用
    >
    > 1. ==softmax==；想想sigmoid。这个是多分类，单调非负归一
    >
    >     类似操作还有矩阵乘法等
    
    $$
    sigmoid = \frac{1}{1 + e^{-x}} \\
    
    \frac{d(sigmoid)}{dx} = sigmoid(x) · (1 - sigmoid(x))
    $$
    
    ![image-20250408113044493](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250408113044493.png)
    
    
    $$
    Tanh = \frac{e^x - e^{-x}}{e^x + e^{-x}} \\
    \frac{dTanh}{dx} = 1 - Tanh^2
    $$
    
    ![image-20250408114522622](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250408114522622.png)
    
    ![image-20250408150852517](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250408150852517.png)
    $$
    ReLU = max(0, x)\\\\
    \frac{d(ReLU)}{dx} = \left\{\begin{matrix}
                          1,\ \ x > 0 \\
                          \ \ 0,\ \ x <= 0
    					 \end{matrix}\right.
    $$
    
    
    ![image-20250408115631449](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250408115631449.png)
    $$
    ReLU6 = \left\{\begin{matrix}
              x,\ \ 0 <= x <= 6 \\
              \ \ 6,\ \ x < 0 \ or \ x >6
             \end{matrix}\right.
    $$

$$
Leakey \ ReLU = \left\{\begin{matrix}
                  x,\ \ x >= 0 \\
                  ax,\ \ x < 0
                 \end{matrix}\right. \ a为常数
\\
PReLU = \left\{\begin{matrix}
                  x,\ \ x >= 0 \\
                  ax,\ \ x < 0
                 \end{matrix}\right. \ a为训练参数
\\
RReLU =\left\{\begin{matrix}
                  x,\ \ x >= 0 \\
                  ax,\ \ x < 0
                 \end{matrix}\right. \ a为随机数
$$

$$
ELU = \left\{\begin{matrix}
                  &x,&x >= 0 \\
                  &a(e^x-1),&x < 0
                 \end{matrix}\right. \ a为超参数
\\\\
SELU =\lambda ·\left\{\begin{matrix}
                  &x,&x >= 0 \\
                  &a(e^x-1),&x < 0
                 \end{matrix}\right. \ \lambda和a为超参数
$$

![image-20250408152548266](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250408152548266.png)

> 超参数a=1时


$$
GELU(x)=xP(X≤x)\\
\ \ \ \ \ \ \ \ =xΦ(x)
$$

$$
xΦ(x)≈xσ(1.702x)
\\
xΦ(x)≈\frac{1}{2} ×[1+tanh(\sqrt{\frac{π}{2}}(x+0.044715x^3))]
$$

![image-20250408175314814](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250408175314814.png)![image-20250408175719134](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250408175719134.png)

![](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/op-activation-figure9.jpg)



$$
Swish = x·Sigmoid(\beta x)
$$

$$
Hard\ Swish = x·\frac{ReLU6(x + 3)}{6}
$$

![](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/op-activation-figure13.jpg)

$$
\text{Softmax}(x_{i}) =  \frac{e^{x_i}}{\sum_je^{x_j}} \\
\frac{\partial y_i}{\partial x_j} =

\begin{cases}

y_i (1 - y_i), & \text{if } i = j \\

- y_i y_j,     & \text{if } i \ne j

\end{cases}
\\
\\
J = 
\begin{bmatrix}
y_1(1 - y_1) & -y_1 y_2 & \cdots & -y_1 y_n \\
-y_2 y_1 & y_2(1 - y_2) & \cdots & -y_2 y_n \\
\vdots & \vdots & \ddots & \vdots \\
-y_n y_1 & -y_n y_2 & \cdots & y_n(1 - y_n)
\end{bmatrix}
$$

源码里面有所优化：
$$
\text{Softmax}(x_{i}) =  \frac{e^{x_i - max(x)}}{\sum_je^{x_j - max(x)}}
$$


💡 实际意义：

- Softmax 的导数不是独立的，每个输出值的梯度都跟其它的有关。

###  2. 简单总结（未优化）：

|    激活函数    |                          数学表达式                          |             优点             |             缺点             |         适用场景         |
| :------------: | :----------------------------------------------------------: | :--------------------------: | :--------------------------: | :----------------------: |
|  **Sigmoid**   |                    $\frac{1}{1 + e^{-x}}$                    |  输出在 (0,1)，适合概率输出  | 梯度消失、非零均值、计算较慢 |       二分类输出层       |
|  **Softmax**   |               $\frac{e^{x_i}}{\sum_j e^{x_j}}$               | 多分类概率分布，输出总和为 1 |   对极端值敏感，计算成本高   |       多分类输出层       |
|    **Tanh**    |             $\frac{e^x - e^{-x}}{e^x + e^{-x}}$              |    输出在 (-1,1)，零均值     |         梯度消失问题         |    RNN、数据对称场景     |
|    **ReLU**    |                         $\max(0, x)$                         |    计算高效，缓解梯度消失    |  神经元死亡（负值输出为 0）  |   CNN、默认隐藏层激活    |
| **Leaky ReLU** | $\begin{cases} x & \text{if } x \geq 0 \\ \alpha x & \text{if } x < 0 \end{cases}$ |      缓解神经元死亡问题      |   需手动调参（如 α=0.01）    |    深层网络替代 ReLU     |
|   **PReLU**    |                   Leaky ReLU，但$α$可学习                    |      自适应斜率，更灵活      |          增加参数量          |    复杂任务、深层网络    |
|    **ELU**     | $\begin{cases} x & \text{if } x \geq 0 \\ \alpha(e^x - 1) & \text{if } x < 0 \end{cases}$ |    平滑负值，缓解梯度消失    |    计算复杂（含指数运算）    |    需要处理负值的场景    |
|    **GELU**    |             $x \cdot \Phi(x)$  Φ 为标准正态 CDF              |    平滑柔和，适合深度网络    |           计算复杂           | Transformer、BERT 等模型 |
|   **Swish**    |             $x \cdot \sigma(\beta x)$（$β$可调）             |  非单调，实验性能优于 ReLU   |          计算较复杂          | EfficientNet 等先进网络  |
|    **Mish**    |                $x \cdot \tanh(\ln(1 + e^x))$                 |  平滑、无上界，缓解梯度消失  |          计算成本高          |      计算机视觉任务      |
|    **Step**    | $\begin{cases} 1 & \text{if } x \geq 0 \\ 0 & \text{if } x < 0 \end{cases}$ |           二元输出           |   不可微，无法用于梯度下降   |  早期感知机（现已少用）  |

### 3. 我的思考：

- 对于类似softmax这种同一条数据的分母相同的，每次是否需要重新计算分母？（底层是否会做cache？）
- PyTorch源码里面的`Softmax`激活提及了`NLLLoss`，这是什么？与交叉熵又是什么关系？
- softmax的导数推导。
- softmax输入的shape和梯度的shape不一样大吗？参数更新的时候又是怎样子的？

---





## 02. 常用Torch算子

torch.nn

- [Containers](https://pytorch.org/docs/stable/nn.html#containers)
- [Convolution Layers](https://pytorch.org/docs/stable/nn.html#convolution-layers)
- [Pooling layers](https://pytorch.org/docs/stable/nn.html#pooling-layers)
- [Padding Layers](https://pytorch.org/docs/stable/nn.html#padding-layers)
- [Non-linear Activations (weighted sum, nonlinearity)](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
- [Non-linear Activations (other)](https://pytorch.org/docs/stable/nn.html#non-linear-activations-other)
- [Normalization Layers](https://pytorch.org/docs/stable/nn.html#normalization-layers)
- [Recurrent Layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers)
- [Transformer Layers](https://pytorch.org/docs/stable/nn.html#transformer-layers)
- [Linear Layers](https://pytorch.org/docs/stable/nn.html#linear-layers)
- [Dropout Layers](https://pytorch.org/docs/stable/nn.html#dropout-layers)
- [Sparse Layers](https://pytorch.org/docs/stable/nn.html#sparse-layers)
- [Distance Functions](https://pytorch.org/docs/stable/nn.html#distance-functions)
- [Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [Vision Layers](https://pytorch.org/docs/stable/nn.html#vision-layers)
- [Shuffle Layers](https://pytorch.org/docs/stable/nn.html#shuffle-layers)
- [DataParallel Layers (multi-GPU, distributed)](https://pytorch.org/docs/stable/nn.html#module-torch.nn.parallel)
- [Utilities](https://pytorch.org/docs/stable/nn.html#module-torch.nn.utils)
- [Quantized Functions](https://pytorch.org/docs/stable/nn.html#quantized-functions)
- [Lazy Modules Initialization](https://pytorch.org/docs/stable/nn.html#lazy-modules-initialization)
    - [Aliases](https://pytorch.org/docs/stable/nn.html#aliases)

### 1. Convolution（未补）

### 2. 线性变换层

- Linear/Gemm

    > Note: Linear的weight是转置存放的

- Matmul

    > type与Linear的不同
    >
    > 要满足广播机制

### 3. Normlization（🌟🌟🌟🌟）

- 类型
    $$
    Batch Norm，Layer Norm，Instance Norm，Group Norm，RMS Norm
    $$
    

![figure4](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/op-figure4.jpg)



- 公式
    $$
    y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    $$



- 功能

    > - 去量纲，把数据调整到更强烈的数据分布
    > - 减少梯度消失和梯度爆炸
    > - 主要是有一个计算期望和方差的过程
    > - 做Norm的粒度不同，应用场景不同
    >
    > - 其他资料：[Batch Norm的技术博客](https://blog.csdn.net/LoseInVain/article/details/86476010)

- 特征和不同

    >- 粒度不同（维度不同），对应应用领域不同
    >
    >**Batch Norm是逐channel（每个batch的同一个channel）进行标准化**，也就是垮batch的。图片恰好需要这种方式。
    >
    >LN是逐batch进行标准化的。NLP中往往是一个一个的seq进行训练的，而且长度不同，更适合这种。**这让我想起了Attention的soft max操作是对一个行向量进行归一化的**
    >
    >LayerNorm有助于稳定训练过程并提高收敛性。它的工作原理是对输入的各个特征进行归一化，确保激活的均值和方差一致。**普遍认为这种归一化有助于缓解与内部协变量偏移相关的问题，使模型能够更有效地学习并降低对初始权重的敏感性。**从架构图上看，LayerNorm在每个Transformer 块中应用两次，一次在自注意力机制之后，一次在FFN层之后，但是在实际工作中不一定如此。
    >
    >文本长度不确定，而在LN层可以。
    >
    >应用场景确定LN

- BN期望和方差计算策略

    > `采用移动指数平均`，会有历史信息在，有点类似RNN了
    >
    > $E_n = \alpha E + (1- \alpha)E_{n-1}$
    >
    > Var 同理

- 在训练和推理时有何不同？

    > pytorch的模型有两种模式，在module模块里面有个`training`属性，也有对应的API，里面明确指出了这个
    >
    > 在BatchNorm采用训练时计算的结果（E和Var），应用到测试或者推理的时候
    >
    > 在Dropout后续会说，训练会drop掉，但推理不会，会改成（1-rate）

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

    

- **RMS Norm**(大模型使用)🌟🌟🌟

    > 来源于LN，简化了LN
    >
    > 减均值相当于平移，这里直接去掉平移，只保留缩放
    >
    > 把乘法直接放进来了，

对LN做简化，对于NLP，对缩放敏感，对平移不敏感，所以分子不减$E_x$，减少了很大计算量

![image-20250326224954810](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250326224954810.png)



**补充部分：**

- **DyT**(Transformers without normlization)

    [Transformers without normlization](https://yiyibooks.cn/arxiv/2503.10622v1/index.html)
    $$
    Norm:\ \gamma * \frac{(_x - E_{_x})}{\sqrt (Var_{_x})} + \beta
    $$

    $$
    DyT:\ \gamma * Tanh(\alpha * x) + \beta
    $$

    ![image-20250410151122973](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250410151122973.png)

- **Pre-Norm（大模型使用）和Post-Norm**

    > 结合Transformer那块知识，一个是在残差前，一个在残差后

    $$
    Output_{post}=LayerNorm(x+SubLayer(x))
    \\
    Output_{pre}=x+SubLayer(LayerNorm(x))
    $$

    

$$
公式1:PostNorm(x)=x+LayerNorm(FeedForward(x+LayerNorm(Attention(x))))
\\\\
公式2:PreNorm(x)=x+FeedForward(LayerNorm(x))+Attention(LayerNorm(x))
$$



| 特性     |                Post-Norm                 |             Pre-Norm             |
| -------- | :--------------------------------------: | :------------------------------: |
| 公式     |                  公式1                   |              公式2               |
| 位置     |                  残差后                  |              残差前              |
| 出现时间 | 原始 Transformer（Vaswani et al., 2017） |     之后发展（如 GPT-2 等）      |
| 优点     |        收敛后表现略好（某些任务）        | 更稳定，训练深层模型不易梯度消失 |
| 缺点     |       深层模型中容易梯度消失/爆炸        |  可能最终性能略低，但更容易训练  |
| 应用情况 |          BERT、初版Transformer           |    GPT系列、T5、LLama等大模型    |



|         模型         | 归一化类型 |
| :------------------: | :--------: |
|     **DeepSeek**     |  Pre-Norm  |
|    **GPT-2/3/4**     |  Pre-Norm  |
|       **BERT**       | Post-Norm  |
|        **T5**        |  Pre-Norm  |
|      **LLaMA**       |  Pre-Norm  |
|  **Transformer XL**  |  Pre-Norm  |
| **原始 Transformer** | Post-Norm  |

- DeepNorm

    DeepNorm 是微软在 2022 年提出的改进方法（论文 *"[DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)"*），**基于 Post-Norm 但大幅提升了深层训练的稳定性**，可支持超深层（如 1000 层）Transformer 的训练。

    ![image-20250409164334392](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250409164334392.png)

    ![image-20250409162034019](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250409162034019.png)

    原始残差结构:
    $$
    x_{l+1} = LayerNorm(x_l + F(x_l))
    $$
    DeepNorm:
    $$
    x_{l+1} = \text{LN}(\alpha \cdot x_l + G_l(x_l, \theta_l))
    $$
    ![image-20250409164837814](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250409164837814.png)

**思考：**DeepNorm中的$\beta$是哪里的参数？



> Note：整体回顾一下算子部分，引入后面的算子，不然大脑还停留在norm部分



### 4. Polling(CV)

- 作用

> 增大感受野
>
> 减少特征图尺寸，保留重要信息，降低计算量
>
> 降采样，减少噪音
>
> 位置变化鲁棒性增强

- 其他

> 对应有个out position矩阵

- 类型

> Max、Avg等

### 5. activations（看01部分）

> 这里面内容较多，直接看01章节

[01. 激活函数🌟🌟🌟](#01. 激活函数🌟🌟🌟)

### 6. 其他

### 7. 特别的一些Operator

$$
reshape、view、permute、transpose
$$

> 可能需要补充一下PyTorch的Tensor知识
>
> 比如：matedata和storage；data、storage、data_ptr、stride、contiguous；state、state_dict等

- reshape

    > 原始数据内存排布不变，只变shape

- view

    > 类似reshape，原始数据不变

- permute

    > permute会对数据底层重排，支持多个轴进行交换

- transpose

    > 类似permute，会对数据重排，支持两轴交换

**Tips:**

> 1. view()：当tensor连续时tensor.view()不改变存储区的真实数据，只改变元数据（Metadata）中的信息, 调用view方法张量必须连续的。 
>
> 2.  reshape()：当tensor连续时和view()相同，不连续时等价于contiguous().view() 
>
> 3. permute()：通过改变张量的步长（stride）重新排列张量的维度，但会导致张量在内存中的存储变得不连续 
>
> 4. contiguous()：开辟新的存储区，确保张量在内存中是连续存储，在permute()操作后需要接contiguous()才能接view()
>
> 5. stride()：在指定维度（dim）上，存储区中的数据元素，从一个元素跳到下一个元素所必须的步长 
>
> 6. pytorch的存储方式：metadata+storage
>
>     metadata保存：size,dimension,stride等元信息
>
>     storage保存：以一维数组保存对应的张量数据
>
>     <img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250409174658632.png" alt="image-20250409174658632" style="zoom:50%;" />


$$
squeeze、unsqueeze
$$

- 是对维度的压缩和扩充

    > 增加一个维度，减少一个维度

$$
concat、stack、expand、flatten
$$

- concat

    > 与split相反操作，可以指定某轴

- stack

    > 支持新增一个轴进行拼接

- expand

    > 支持广播机制

- flatten

    > 拉平成一维

$$
pointwize类型
$$

……
$$
split、slice
$$

$$
reduce类型
$$

### 8. Embedding（可能需要单独一节）

> 结合Tokenizer

- 对分词器分到的结果进行Embedding

- 有一个Embedding表，直接根据index查到
- 计算原理：传入weight大小，把Embedding table里面的提取

### 9. Dropout（🌟🌟🌟）

- 功能

- 原理

    > 丢弃 => 置零
    >
    > 随机性，不然会神经元坏死

- 训练和推理有啥不同

    > 推理直接去掉，把连接weight乘以（1-p）

### 10 . 我的思考：

- 不同的Norm的参数量

    > 可学习参数和均值方差

- 不同Norm操作维度，Conv操作维度，Polling操作维度

- 其他哪些算子的底层是copy还是in-place

- 如何实现训练和推理不同的情况？（相当于加锁或者if else）

---





## 03. BP神经网络&BaseLine

