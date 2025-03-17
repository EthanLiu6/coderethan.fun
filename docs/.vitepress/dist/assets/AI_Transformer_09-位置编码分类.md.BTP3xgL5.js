import{_ as a,c as i,o as t,a2 as r,j as l}from"./chunks/framework.DA-Pb-tg.js";const c=JSON.parse('{"title":"探秘Transformer系列之（9）--- 位置编码分类","description":"","frontmatter":{},"headers":[],"relativePath":"AI/Transformer/09-位置编码分类.md","filePath":"AI/Transformer/09-位置编码分类.md","lastUpdated":null}'),o={name:"AI/Transformer/09-位置编码分类.md"};function n(p,e,s,h,f,d){return t(),i("div",null,e[0]||(e[0]=[r('<h1 id="探秘transformer系列之-9-位置编码分类" tabindex="-1">探秘Transformer系列之（9）--- 位置编码分类 <a class="header-anchor" href="#探秘transformer系列之-9-位置编码分类" aria-label="Permalink to &quot;探秘Transformer系列之（9）--- 位置编码分类&quot;">​</a></h1><p>目录</p><ul><li><a href="#探秘transformer系列之9----位置编码分类">探秘Transformer系列之（9）--- 位置编码分类</a><ul><li><a href="#0x00-概述">0x00 概述</a></li><li><a href="#0x01-区别">0x01 区别</a><ul><li><a href="#11-从直观角度来看">1.1 从直观角度来看</a></li><li><a href="#12-从模型处理角度来看">1.2 从模型处理角度来看</a></li><li><a href="#13-优劣">1.3 优劣</a></li></ul></li><li><a href="#0x02-绝对位置编码">0x02 绝对位置编码</a><ul><li><a href="#21-基础方案">2.1 基础方案</a></li><li><a href="#22-训练式">2.2 训练式</a></li><li><a href="#23-三角函数式">2.3 三角函数式</a></li><li><a href="#24-其它">2.4 其它</a></li></ul></li><li><a href="#0x03-相对位置编码">0x03 相对位置编码</a><ul><li><a href="#31-意义">3.1 意义</a><ul><li><a href="#大脑中的参考系">大脑中的参考系</a></li><li><a href="#语义影响">语义影响</a></li><li><a href="#长度外推">长度外推</a></li></ul></li><li><a href="#32-绝对位置编码的位置">3.2 绝对位置编码的位置</a></li><li><a href="#33-绝对位置编码的公式">3.3 绝对位置编码的公式</a></li><li><a href="#34-经典式">3.4 经典式</a></li><li><a href="#35-xlnet">3.5 XLNET</a></li><li><a href="#36-tener">3.6 TENER</a></li><li><a href="#37-t5">3.7 T5</a></li><li><a href="#38-deberta式">3.8 DeBERTa式</a></li><li><a href="#39-tupe">3.9 TUPE</a></li><li><a href="#310-alibi">3.10 ALiBi</a></li><li><a href="#311-偏置编码上下文模式">3.11 偏置编码&amp;上下文模式</a></li><li><a href="#312-小结">3.12 小结</a></li></ul></li><li><a href="#0xff-参考">0xFF 参考</a></li></ul></li></ul><h2 id="_0x00-概述" tabindex="-1">0x00 概述 <a class="header-anchor" href="#_0x00-概述" aria-label="Permalink to &quot;0x00 概述&quot;">​</a></h2><p>由于 Transformer 自身具有置换不变性（Permutation Invariance），无法直接捕获每个词在序列中的位置信息，因此使用位置编码将序列中元素顺序信息融入Transformer成为一种常见做法。根据位置编码表示的是序列中元素的绝对位置信息还是相对位置信息，业界将位置编码主要分为绝对位置编码（Absolute Position Encoding，APE）和相对位置编码（Relative Position Encoding，RPE）。绝对位置编码的核心思想是在每个输入序列的元素上添加一个位置向量，以表示该元素在序列中的具体位置。相对位置编码则侧重于考虑元素之间的距离信息。这里说主要是因为还有一些难以划分的位置编码。当然也有其它的区分方式，比如把RoPE单独列为旋转编码。</p><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211103254-1017748743.jpg" alt=""></p><h2 id="_0x01-区别" tabindex="-1">0x01 区别 <a class="header-anchor" href="#_0x01-区别" aria-label="Permalink to &quot;0x01 区别&quot;">​</a></h2><p>上一篇我们知道为了克服自注意力矩阵带来的影响，有的研究人员提出了相对编码。从而引出了对位置编码的分类。我们本节从各个角度出发，来看看绝对位置编码和相对位置编码的区别。</p><h3 id="_1-1-从直观角度来看" tabindex="-1">1.1 从直观角度来看 <a class="header-anchor" href="#_1-1-从直观角度来看" aria-label="Permalink to &quot;1.1 从直观角度来看&quot;">​</a></h3><p>以句子“从槐树叶底，朝东细数着一丝一丝漏下来的日光“为例，对于如何获取序列顺序？我们大体有两个选择方案：</p><ul><li>绝对位置信息。比如：“从”是第一个token，“底”是第五个token。</li><li>相对位置信息。比如：“光”距离”日&quot;差一个位置，但是距离“漏”差四个位置。</li></ul><p>这两个方案就分别对应了绝对位置编码和相对位置编码。下图给出了从直观角度出发来看，原始无位置编码，绝对位置编码和相对位置编码的区别。</p><ul><li>未引入位置编码。在人类的语言中，单词的位置与顺序定义了语法，也影响着语义。无法捕获的单词顺序会导致我们很难理解一句话的含义。</li><li>绝对位置编码。绝对位置编码的作用方式是告知Transformer架构每个元素在输入序列的位置，类似于为输入序列的每个元素打一个&quot;位置标签&quot;来标明其绝对位置。</li><li>相对位置编码。相对位置编码作用于自注意力机制，告知Transformer架构两两元素之间的距离。</li></ul><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211121023-860000873.jpg" alt=""></p><p>由于自然语言一般更依赖于相对位置，所以相对位置编码通常也有着优秀的表现。</p><h3 id="_1-2-从模型处理角度来看" tabindex="-1">1.2 从模型处理角度来看 <a class="header-anchor" href="#_1-2-从模型处理角度来看" aria-label="Permalink to &quot;1.2 从模型处理角度来看&quot;">​</a></h3><p>从模型处理角度来看，这两种方案有如下分别：</p><ul><li>绝对位置信息是在输入层做文章，在输入阶段就将位置信息融入到token的输入表征中，具体细节如下： <ul><li>在Transformer中的位置。APE只在第一层之前出现。</li><li>建模方式。根据绝对位置k来定义位置编码，使用公式函数或者可学习向量得到每个token的位置编码。</li><li>相对距离。每个位置的位置编码是固定的向量，且每个位置相互独立，不考虑其与其他位置的关系，因此在和注意力机制结合时，无法计算相对距离。</li><li>模型输入。对于模型来说，每个token对应的输入是token自身编码和其位置编码的融合。</li><li>操作对象。位置编码的操作对象是自注意力变换中的特征序列Q, K（Transformer论文是针对输入嵌入序列 X），即将token的绝对位置信息添加到对应的qt,ksqt,ksq_t,k_s中。</li></ul></li><li>相对位置信息主要是在模型网络层做文章，通过微调注意力结构，使得模型有能力分辨不同位置的Token。 <ul><li>在Transformer中的位置。RPE通常在每一层都重复出现，而不是像APE那样只在第一层之前出现。</li><li>建模方式。相对位置编码并没有对每个输入的位置信息做完整建模。而是对相对位置i-j进行建模。即在计算自注意力分布时考虑两个token间的相对位置信息，即下标之差。让模型通过数据自己学习位置信息来分辨不同位置的Token。</li><li>相对距离。绝对位置编码考虑的是各个独立token的位置信息；相对位置编码考虑的则是进行Attention计算时的query、key之间的相对位置信息，或者说是基于两个token之间的相对距离来表示位置关系。</li><li>模型输入。一般来说，相对位置编码并不是将位置编码直接加到词嵌入上，模型的输入依然是词嵌入。也有的方案使用距离编码矩阵（Distance Encoding Matrix）来计算偏移向量，然后与位置嵌入向量相加，再输入模型。</li><li>操作对象。位置编码的操作对象是自注意力变换中的自注意力矩阵 A（早期方案也有涉及特征序列 V 的操作），即将两token的相对位置信息添加到对应的At,sAt,sA_{t,s}上。</li></ul></li></ul><p>这些差异如下图所示。其中 p(j-i) 是编码 j - i 相对位置信息的术语。RPEs倾向于直接修改注意力机制来融合相对位置信息，这种修改独立于值向量，使它们不与位置信息纠缠。</p><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211126388-1846762296.jpg" alt=""></p><h3 id="_1-3-优劣" tabindex="-1">1.3 优劣 <a class="header-anchor" href="#_1-3-优劣" aria-label="Permalink to &quot;1.3 优劣&quot;">​</a></h3><p>绝对位置编码的优点是：实现简单。缺点是：</p><ul><li>难以泛化。</li><li>相对定位的缺乏可能会阻碍模型理解语言结构的细微差别的能力。</li></ul><p>相对位置编码的优点是：</p><ul><li>可以将位置信息渗透进特征向量 qtqtq_t 和 ksksk_s 的每一个维度，在刻画长距离语义关系时，不仅可以有效过滤无关信息，也为传递有价值的语义关联提供了渠道。</li><li>能够更好地处理序列的局部结构，因为它关注的是元素之间的相对位置。</li><li>建立了语义信息和位置信息之间沟通的桥梁，不再让位置信息构成绝对的抑制。</li></ul><p>缺点是：</p><ul><li>计算效率低下。由于自注意力层中的额外计算步骤（比如获得每个时间步的相对位置编码，位置矩阵被添加到查询键矩阵中）使得计算更复杂，可能增加训练和推理的时间。</li><li>KV Cache 使用的复杂性：由于每个附加token都会改变每个其他token的嵌入，这使得 Transformer 中KV Cache的有效使用变得复杂。使用 KV Cache 的一项要求是已经生成的单词的位置编码， 在生成新单词时不改变（绝对位置编码）。因此相对位置编码不适合推理，因为每个标记的嵌入会随着每个新时间步的变化而变化。</li><li>整体偏置易随相对位置大小波动，需要更多的维度、额外的校正才能有所缓解，并且针对自身想额外抑制的语义关系，无法做到彻底的惩罚。</li></ul><p>另外，目前这些位置编码会让模型过分在意局部信息，过分相信邻近生成的内容。如何在位置编码层面实现“三思而后行”，即让模型在注意邻近信息的同时，也能考虑到较远位置的信息，从而对当前输出进行一定的纠正，也是一个不可忽视的问题。</p><p>虽然说位置编码主要是绝对位置编码和相对位置编码两大类，但每一类其实又能衍生出各种各样的变种，研究人员在这方面发挥了极大的聪明才智。本文就让我们来欣赏一下研究人员为了更好地表达位置信息所构建出来的“八仙过海，各显神通”般的编码方案。</p><h2 id="_0x02-绝对位置编码" tabindex="-1">0x02 绝对位置编码 <a class="header-anchor" href="#_0x02-绝对位置编码" aria-label="Permalink to &quot;0x02 绝对位置编码&quot;">​</a></h2><p>形式上来看，绝对位置编码是相对简单的一种方案，经典的绝对位置编码有三种：</p><ul><li>可学习且无约束的。代表作是论文&quot;Convolutional Sequence to Sequence Learning&quot;，该工作使用可训练的嵌入形式作为位置编码。</li><li>可训练的位置编码，代码作是论文“BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”。</li><li>三角函数位置编码。代表作是论文&quot;Attention Is All You Need&quot;。文章中使用正余弦函数生成的位置编码。</li></ul><p>近年来，关于绝对位置编码的工作大多数是以不同的方法生成绝对位置编码为主，比如在三角函数编码的基础之上额外学习一些其他参数。</p><h3 id="_2-1-基础方案" tabindex="-1">2.1 基础方案 <a class="header-anchor" href="#_2-1-基础方案" aria-label="Permalink to &quot;2.1 基础方案&quot;">​</a></h3><p>基础方案是论文&quot;Convolutional Sequence to Sequence Learning&quot;提出来的。该方案将每个单词的位置k映射为一个唯一的位置向量pkpkp_k，然后在每个词的嵌入xkxkx_k上加位置编码pkpkp_k之后输入模型。形式上如下：x=(x1+p1,...,xk+pk)x=(x1+p1,...,xk+pk)x=(x_1+p_1,...,x_k+p_k)。其中，x表示模型的输入，xkxkx_k表示第k个位置的词嵌入，pkpkp_k表示第k个位置的绝对位置编码，且只依赖于位置编号k。</p><h3 id="_2-2-训练式" tabindex="-1">2.2 训练式 <a class="header-anchor" href="#_2-2-训练式" aria-label="Permalink to &quot;2.2 训练式&quot;">​</a></h3><p>BERT/GPT使用的是可学习的位置编码（learned absolute positional embedding），通过在模型的嵌入层中引入可学习的参数来学习位置信息的表示。具体做法就是直接将位置编码当作可训练参数，初始化一个形状为[max_length, hidden_size]的矩阵作为位置向量，让它随着训练过程更新。即为每个输入下标训练一个嵌入向量来刻画绝对位置特征。后续这个矩阵就像词表一样使用。</p><p>可学习方案的优点是可以根据任务的需要进行调整，可以更准确地区分不同位置的词语，并捕捉到位置信息对任务的影响，进而学习到最适合特定任务的位置编码。缺点是扩展性不强，外推性差。只能表征有限长度内的位置，无法对任意位置进行建模，不能很好地泛化到训练时未见过的更长序列，也不具有远程衰减性。而且由于位置信息是通过位置编码隐式提供的，模型需要从数据中学习如何最好地利用这些信息，这可能需要更多的模型参数和训练数据。</p><h3 id="_2-3-三角函数式" tabindex="-1">2.3 三角函数式 <a class="header-anchor" href="#_2-3-三角函数式" aria-label="Permalink to &quot;2.3 三角函数式&quot;">​</a></h3><p>vanilla Transformer通过固定的数学公式（使用正弦和余弦函数）来生成位置向量，从而捕捉到不同位置之间的相对关系。这里不再对细节进行赘述。因为其思路之一是希望通过绝对编码方式来实现相对编码，因此也有人将其归为混合位置编码。</p><h3 id="_2-4-其它" tabindex="-1">2.4 其它 <a class="header-anchor" href="#_2-4-其它" aria-label="Permalink to &quot;2.4 其它&quot;">​</a></h3><p>此外还有一些其它方法，比如：</p><ul><li><p>Encoding Word Order in Complex Embeddings 提出一种复值词向量函数生成绝对位置编码，巧妙地将复值函数的振幅和相位与词义和位置相联系。该复值词向量函数以位置为变量来计算每个词在不同位置的词向量。由于该函数对于位置变量而言是连续的，因此该方法不光建模了绝对位置，也建模了词之间的相对位置。</p></li><li><p>SHAPE: Shifted Absolute Position Embedding for Transformers 提出了一种绝对位置编码的鲁棒性训练方法。SHAPE的基本思想是在训练过程中对绝对位置编码随机整体平移一段距离来实现泛化能力。</p></li><li><p>Rethinking Positional Encoding in Language Pretraining在注意力上添加两个标记位置嵌入之间的点积logit。</p></li><li><p>也有研究人员在考虑使用xk⊗pk（逐位相乘）对词嵌入和位置编码进行融合。因为token embedding和PE相加其实是一种特征交叉，从这个角度来看的话，其实相乘也是一种特征交叉的方式。</p></li></ul><h2 id="_0x03-相对位置编码" tabindex="-1">0x03 相对位置编码 <a class="header-anchor" href="#_0x03-相对位置编码" aria-label="Permalink to &quot;0x03 相对位置编码&quot;">​</a></h2><h3 id="_3-1-意义" tabindex="-1">3.1 意义 <a class="header-anchor" href="#_3-1-意义" aria-label="Permalink to &quot;3.1 意义&quot;">​</a></h3><p>我们从几个方面来看看相对位置的意义。</p><h4 id="大脑中的参考系" tabindex="-1">大脑中的参考系 <a class="header-anchor" href="#大脑中的参考系" aria-label="Permalink to &quot;大脑中的参考系&quot;">​</a></h4><p>美国国家工程院院士杰夫·霍金斯(Jeff Hawkins)在其论文和著作《千脑理论》提出来一些观点很值得我们思考：</p><ul><li>参考系与新皮质。 <ul><li>新皮质的关键是参考系。</li><li>参照系在新皮质中无处不在。</li></ul></li><li>参考系与存储。 <ul><li>参考系是一种信息在大脑中的存储结构，大脑是使用参考系来管理所有知识。</li><li>知识存储在与参考系相关联的位置。我们所知的每一个事实都与参考系中的一个位置相对应。</li></ul></li><li>参考系与建模。 <ul><li>大脑通过感官输入与参考系中的位置联系起来，建立世界模型。</li><li>参考系不仅仅为实物建模，而是为我们所知道的一切建模。除了具象的物体之外，参考系还能衍生到一些抽象的概念，例如哲学，民主都是基于新皮质中不同的参照系进行定义的。</li></ul></li><li>参考系与思考。 <ul><li>序列识别问题。新皮质必须知道接下来的移动是什么，才能做出来对于序列的下一个输入的预测。</li><li>思考是一种特殊形式的移动。假设我们所知的一切都存储在参考系中，那么为了回忆存储的知识，我们需要在参考系中激活适当的位置。当神经元激活一个又一个位置的时候，思考就产生了。</li></ul></li></ul><p>如上所述，参考系是人脑中的重要部分，这对于位置编码具有极其重要的指导意义。或者更确切的说，这是相对位置编码的重要理论支撑之一。</p><h4 id="语义影响" tabindex="-1">语义影响 <a class="header-anchor" href="#语义影响" aria-label="Permalink to &quot;语义影响&quot;">​</a></h4><p>在很多任务中，序列中的元素之间的相对位置关系对于理解序列的语义和结构非常重要。或者说，绝对位置编码对句子语义的影响不大，更为重要的是相对位置编码。比如下面句子中，相对语序比绝对语序对语义的影响更加关键。</p><ul><li>读书好、读好书、好读书。</li><li>四川人不怕辣、贵州人辣不怕、湖南人怕不辣。</li><li>有个不明生物在吃鸡， 有只鸡在吃不明生物。</li></ul><h4 id="长度外推" tabindex="-1">长度外推 <a class="header-anchor" href="#长度外推" aria-label="Permalink to &quot;长度外推&quot;">​</a></h4><p>直观地说，长度外推与长度和位置有很强的相关性。Transformer作者提出了正弦位置嵌入，并声称它可以外推到训练之外的更长的序列。这一说法背后的想法，即只需改变位置表示方法就可以实现长度外推，已得到广泛支持和证明。因此，开发更好的位置编码方法已经成为增强Transformer长度外推的主要途径。</p><p>由于 APE 在长度外推上的表现难以令人满意，而 RPE 天然地由于其位移不变性具备更好的外推能力。并且人们普遍认为上下文中单词的相对顺序更重要。因此，近年来，RPE 已成为编码位置信息的主要方法。</p><p>早期的 RPE 来自于对正弦位置编码的简单修改，并常常结合裁剪或分箱策略来避免出现分布外的位置嵌入，这些策略被认为有利于外推。此外，由于 RPE 解耦了位置和位置表示之间的一对一对应关系，因此将偏差项直接添加到注意力公式中成为将位置信息集成到 Transformer 中的一种可行甚至更好的方法。这种方法要简单得多，并且自然地解开了值（value）向量和位置信息的纠缠。</p><h3 id="_3-2-绝对位置编码的位置" tabindex="-1">3.2 绝对位置编码的位置 <a class="header-anchor" href="#_3-2-绝对位置编码的位置" aria-label="Permalink to &quot;3.2 绝对位置编码的位置&quot;">​</a></h3><p>如何在Transformer中加上相对位置信息？出发点有两个，但是殊途同归。</p><ul><li>因为每个单词的位置编码是相对于其他单词的位置差异而得到的，所以显然就不好像APE那样直接加到输入上了，需要从输入之后的模块入手，这就是注意力模块。</li><li>前文分析过，原始transformer中的相对位置表达能力是在计算注意力阶段被破坏的。因此，研究人员自然想到，用过在注意力计算时候再把相对位置信息加上。</li></ul><p>因此，研究人员通过修改自注意力计算的过程，把相对位置信息植入到Transformer架构的每一层的注意力机制中。相对位置编码会根据矩阵元素的下标，直接考虑每个元素对应的两个token间的相对位置关系。比如在计算自注意力矩阵时，无论是在query和key的dot product，以及最终注意力权重和value矩阵乘时，都会分别额外添加一个表示位置m和位置n相对位置信息的仅依赖于m-n的bias。这样通过将每个元素的位置嵌入向量与其他位置的偏移向量进行组合，来编码元素之间的相对距离。每个元素的位置嵌入向量会随着其与其他元素的位置关系而变化，从而更好地捕捉序列中的局部结构信息，从而提高序列建模的性能。</p><p>以“You are great“这句子为例，如何获取序列顺序？我们大体有两个选择方案。</p><ul><li>绝对位置信息。比如：“You”是第一个token，“are”是二个token。</li><li>相对位置信息。比如：“great”距离”are&quot;差一个位置，但是距离“great”差两个位置。</li></ul><p>而下图展示了可以添加到注意力矩阵中的绝对和相对位置偏差的示例。左：例句中的注意力矩阵。中间：可学习的绝对位置偏置（bias）。右：相对参考位置的位置偏置。它们表现出直观的权重分配模式，这是绝对位置编码所不具备的。</p><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211141902-871449521.jpg" alt=""></p><h3 id="_3-3-绝对位置编码的公式" tabindex="-1">3.3 绝对位置编码的公式 <a class="header-anchor" href="#_3-3-绝对位置编码的公式" aria-label="Permalink to &quot;3.3 绝对位置编码的公式&quot;">​</a></h3><p>因为相对位置编码大多是在正弦位置编码的基础上修改得到，因此我们先考虑一般的带绝对位置编码的注意力机制。下图上方出了Transformer模型的某一层中自注意力机制的计算流程。最终输出的点乘结果ziziz_i是当前位置i和和序列中所有位置间的关系，是输入序列的线性加权表示结果。</p><p>下图下方的公式 (2) 是query、key之间的向量内积展开式，一共是四项注意力的组合，其中每一项的分别为</p><ul><li>“输入-输入”。 没有考虑位置编码的原始分数，只是基于内容的寻址（content-based addressing）。</li><li>“输入-位置”。相对于当前内容的位置偏差（content-dependent positional bias）。</li><li>“位置-输入”。从内容层面衡量key的重要性，表示全局的内容偏差（global content bias）。</li><li>“位置-位置”。从相对位置层面衡量key的重要性，表示全局的位置偏差（global positional bias）。</li></ul><p>相对位置编码的引入，一般就会从这里出发。有的方案将其中某些项变成可训练的参数，有的甚至把中间两项都去掉了。总之，如何刻画序列不同位置间的相对距离、如何通过相对距离控制自注意力矩阵不同位置的偏置大小，一直是位置编码设计的重中之重。而不同位置编码方案添加偏置的方式则各不相同。</p><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211201673-281871221.jpg" alt=""></p><p>接下来，笔者将带领读者分析一些较为经典的相对位置编码工作。</p><h3 id="_3-4-经典式" tabindex="-1">3.4 经典式 <a class="header-anchor" href="#_3-4-经典式" aria-label="Permalink to &quot;3.4 经典式&quot;">​</a></h3><p>相对位置编码起源于论文<a href="https://papers.cool/arxiv/1803.02155" target="_blank" rel="noreferrer">《Self-Attention with Relative Position Representations》</a>，作者是Transformer的原班人马，他们应该早就知道三角函数编码的问题。</p><p>下图给出了三角函数编码的改造过程，主要思路是以当前位置qtqtq_t为锚点，在计算注意力分数eijeije_{ij}和加权求和ziziz_i时各引入一个可训练的相对位置向量aVijaijVa_{ij}^V和aKijaijKa_{ij}^K，具体技术细节如下：</p><ul><li>形式上与正弦位置编码有联系。</li><li>把相对位置信息i-j加在K和V上面，并且在多头之间共享，其中i-j是有约束条件的。</li><li>相对位置编码的目标是序列元素的边或者距离。所谓相对位置，是将本来依赖于二元坐标(i,j)的向量，改为只依赖于相对距离i−j的向量RKi,j,RVi,jRi,jK,Ri,jVR^K_{i,j},R^V_{i,j}，并且通常来说会进行截断，以适应不同任意的距离，避免分布外位置。 这样只需要有限个位置编码，就可以表达出任意长度的相对位置（因为进行了截断）。或者说，通过在确定的范围内裁剪相对位置，减少了要学习的位置嵌入数量，增强了长度外推。</li></ul><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211212177-648385165.jpg" alt=""></p><p>对于裁剪我们做下说明：这里的相对位置其实就是一个分段函数，在[-k,k]范围内为线性，在两侧做了截断。通过预先设定的最大相对位置k来强化模型对以当前词为中心的左右各k个词的注意力计算。因此，最终的窗口大小为2k+1。对于边缘位置窗口超出2k的单词，则采用了裁剪的机制，即只对有效的临近词进行建模。相对位置权重矩阵aijaija_{ij}如下图所示。</p><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211223032-2050879086.jpg" alt=""></p><p>结合绝对位置编码的公式，本方案的具体推导过程如下：</p><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211229753-1317525956.jpg" alt=""></p><h3 id="_3-5-xlnet" tabindex="-1">3.5 XLNET <a class="header-anchor" href="#_3-5-xlnet" aria-label="Permalink to &quot;3.5 XLNET&quot;">​</a></h3><p>XLNET式位置编码源自Transformer-XL的论文<a href="https://papers.cool/arxiv/1901.02860" target="_blank" rel="noreferrer">Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context</a>。</p><p>Transformer-XL 对绝对位置编码公式做了改动：保留了正余弦频率的取值以及位置信息与语义信息的交互，增加了两个全局可学习的变量u、v，Key的变换矩阵也区分为基于内容的和基于相对位置的两个W。具体细节如下。</p><ul><li>把绝对位置编码替换为相对位置编码。 <ul><li>以qtqtq_t为锚点，将所有的pspsp_s改为rt−srt−sr_{t-s}，表示对key而言由绝对位置编码换成相对于qtqtq_t的相对位置编码。相对位置信息 rt−srt−sr_{t-s} （<em>content-dependent positional bias</em>）是依照Transformer中的通过正余弦频率方式来获取的，该项不需要学习，因此本身也没有被截断。从XLNet论文公式角度看，此处是把绝对位置编码 UjUjU_j换成了相对位置编码Ri−jRi−jR_{i-j}。</li><li>在key上引入相对位置信息，key的变换矩阵WKWKW_K被拆分为基于内容的和基于相对位置的两个W，也就是说输入序列和位置编码不再共享权值。从XLNet论文公式角度看，两个W对应Wk,EWk,EW_{k,E}和Wk,RWk,RW_{k,R}。</li></ul></li><li>调整绝对位置编码公式的第二项。通过矩阵 WRWRW_R 刻画相对距离与上下文语义之间的关系。</li><li>调整绝对位置编码公式的第三项和第四项。 <ul><li>在第三项和第四项中引入引入了两个新的可学习的参数 u∈Rdu∈Rdu∈R^d 和 v∈Rdv∈Rdv∈R^d 来替换Transformer中的query向量。这是因为无论query位置如何，其对不同词的注意偏差都保持一致。因为我们已经把相对位置编码融入了key端，那么query端就不再需要位置编码了。</li><li>如果从XLNet论文公式角度看是替换掉 UTiWTqUiTWqTU_i^TW_q^T。</li></ul></li></ul><p>结合绝对位置编码的公式，本方案的具体推导过程如下：</p><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211241441-1281183323.jpg" alt=""></p><p>应该是从这个工作开始，后续的RPE都只加到K上去，而不加到V上了。</p><h3 id="_3-6-tener" tabindex="-1">3.6 TENER <a class="header-anchor" href="#_3-6-tener" aria-label="Permalink to &quot;3.6 TENER&quot;">​</a></h3><p>从位置编码的角度，TENER作者发现了传统三角式位置编码在实践中不具备单调性，在理论上也缺少前后token间相对方向的感知。因此，TENER作者提出了将相对方向和相对距离都纳入到位置编码当中。TENER的位置编码和Transformer-XL的位置编码类似，形式上，只是去掉了相对位置信息 rt,srt,sr_{t,s} 的变换矩阵。此外，TENER还发现，去除自注意力变换中的校正系数 √dd\\sqrt d ，可以提升其在NER任务上的效果。</p><p>结合绝对位置编码的公式，本方案的具体推导过程如下：</p><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211250879-699898678.jpg" alt=""></p><p>TENER其实揭示了目前位置编码的一些弊病，即：已有的位置编码主要刻画两个token之间的相对距离，缺少对于token间相对方向的刻画，如何实现一个可拆分可解释的方向感知位置编码，是一个很大的挑战。</p><h3 id="_3-7-t5" tabindex="-1">3.7 T5 <a class="header-anchor" href="#_3-7-t5" aria-label="Permalink to &quot;3.7 T5&quot;">​</a></h3><p>同样基于注意力分数计算的分解，<a href="https://arxiv.org/pdf/1910.10683.pdf" target="_blank" rel="noreferrer">T5</a>采用了一种简单的相对位置编码方案，将相对位置直接映射成可学习的标量。从绝对编码公式角度看，T5去除了位置信息和语义信息的交互，直接刻画相对位置信息，使用偏差（浮点数）来表示每个可能的位置偏移。例如，偏差 B1 表示任意两个相距一个位置的标记之间的相对距离，无论它们在句子中的绝对位置如何。 这保证了偏置随着相对位置的单调性。</p><p>简要的说，T5直接将绝对位置公式的后三项换成一个可学习的bias，或者说，它是在（概率未归一化的）Attention矩阵的基础上加一个可训练的偏置项。具体如下：</p><ul><li>删除(b)，(c)项。因为T5作者认为输入信息与位置信息应该是独立（解耦）的，它们就不应该有过多的交互。</li><li>简化(d)项为bijbijb_{ij}。第4项相对位置信息实际上只是一个只依赖于(i,j)的标量，可以将直接映射成可学习的标量，作为参数训练出来。该相对位置偏差矩阵被添加到自注意力层中的查询矩阵和关键矩阵的乘积中。这确保了相同相对距离的标记始终由相同的偏差表示，无论它们在序列中的位置如何。</li><li>去除vj=(xj+pj)WVvj=(xj+pj)WVv_j = (x_j + p_j)W_V中的位置编码。</li></ul><p>结合绝对位置编码的公式，本方案的具体推导过程如下：</p><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211300966-1291996701.jpg" alt=""></p><p>该方法的一个显着优点是其可扩展性。它可以扩展到任意长的序列，这比绝对位置嵌入有明显的优势。</p><h3 id="_3-8-deberta式" tabindex="-1">3.8 DeBERTa式 <a class="header-anchor" href="#_3-8-deberta式" aria-label="Permalink to &quot;3.8 DeBERTa式&quot;">​</a></h3><p>DeBERTa出自<a href="https://arxiv.org/abs/2006.03654" target="_blank" rel="noreferrer">《DeBERTa: Decoding-enhanced BERT with Disentangled Attention》</a>。和T5恰恰相反，DeBERTa去掉了分解后的第四项，在第二、三项中将绝对位置编码改为相对位置编码。其思路如下：</p><ul><li>将二三项中的位置编码换成相对位置编码。首先通过 δ(t,s)δ(t,s)\\delta(t,s) 将 t−s 直接截断在区间 (−k,k] 内，接着通过参数矩阵 P∈R2k×dP∈R2k×dP∈R^{2k×d} 刻画将相对位置映射为特征向量；即采用相对位置编码，并解耦了内容和位置的注意力。</li><li>将第4项扔掉。因为已经使用了相对位置编码，position2position不会带来额外的信息。</li></ul><p>另外，DeBERTa提供了使用相对位置和绝对位置编码的一个新视角，它指出NLP的大多数任务可能都只需要相对位置信息，但确实有些场景下绝对位置信息更有帮助，于是它将整个模型分为两部分来理解。它一共有13层，前11层只是用相对位置编码，这部分称为Encoder，后面2层加入绝对位置信息，这部分它称之为Decoder。</p><p>结合绝对位置编码的公式，本方案的具体推导过程如下：</p><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211311425-107076367.jpg" alt=""></p><h3 id="_3-9-tupe" tabindex="-1">3.9 TUPE <a class="header-anchor" href="#_3-9-tupe" aria-label="Permalink to &quot;3.9 TUPE&quot;">​</a></h3><p>TUPE出自论文&quot;RETHINKING POSITIONAL ENCODING IN LANGUAGE PRE-TRAINING&quot;。</p><p>注：TUPE有APE和RPE两个版本，本文归类时是按照其出发点来归为APE。</p><p>TUPE其实可以看作是T5和DeBERTa的位置编码的结合。TUPE位置编码去掉绝对位置编码的公式的第二三项，保留第四项。相较于T5压缩后学习一个标量刻画相对位置，TUPE将语义信息和位置信息同等看待、分开刻画：以 WQWQW_Q, WKWKW_K 刻画语义关系，并以 UQUQU_Q,UKUKU_K 来刻画位置关系（<em>directly model the relationships between a pair of words or positions by using different projection matrices</em>）</p><p>针对绝对位置编码的公式的四项，论文认为存在两个问题：</p><ul><li>位置嵌入和词嵌入不应该耦合。 <ul><li>在绝对位置编码中，应用于位置嵌入和词嵌入的加法运算带来了两种异构信息资源之间的混合相关性。 它可能会在注意力中带来不必要的随机性，并进一步限制模型的表达能力。</li><li>论文作者对这四项做了可视化，发现中间两项看起来很均匀，说明position和token之间没有太大的关联。因此，TUPE移除了二三项，即移除了单词-位置、位置-单词的对应关系。</li><li>token和position之间用了相同的矩阵做QKV变化，但是position和token之间所包含的信息不同，所以共享矩阵不合理。因此，TUPE解耦了token和position的投影矩阵，通过不同的参数化分别计算词的上下文相关性和位置相关性，然后将它们相加。</li><li>引用T5模型中的偏置项。</li></ul></li><li>其次，考虑到符号 [CLS] 在下游任务中的特殊作用（整个句子的表示），论文质疑将符号 [CLS] 的位置与其他单词一样对待是否是一个合理的设计。 因此，TUPE 将 [CLS] 符号与其他位置分开（untie），从而使其更容易从所有位置捕获信息。</li></ul><p>TUPE架构如下图所示。</p><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211322979-744439333.jpg" alt=""></p><p>解耦的逻辑如下图所示。</p><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211330784-1742857672.jpg" alt=""></p><p>结合绝对位置编码的公式，本方案的具体推导过程如下：</p><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211340845-695504168.jpg" alt=""></p><h3 id="_3-10-alibi" tabindex="-1">3.10 ALiBi <a class="header-anchor" href="#_3-10-alibi" aria-label="Permalink to &quot;3.10 ALiBi&quot;">​</a></h3><p>ALiBi编码出自论文“rain Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation”。ALiBi (Attention with Linear Biases) 其实和T5类似，直接给（未概率归一化的）注意力分数加上了一个线性的bias，通过线性偏置项，让自注意力分布更加关注相对距离较小，即邻近位置的语义信息。区别在于：T5 的偏置是可训练参数，而 ALiBi 的偏置是预设好、不可训练的。</p><p><strong>动机</strong></p><p>ALiBi的动机是：靠近的词比远离的词更重要。</p><p><strong>实施</strong></p><p>ALiBi编码不是给词向量加入位置嵌入向量，而是用一个和query, key之间的距离成比例的一个“惩罚项”来偏置query-key的注意力得分。这个偏置根据 query 和 key 的相对距离来惩罚 attention score，相对距离越大，惩罚项越大。相当于两个 token 的距离越远，相互贡献就越小。比如，针对原始attention层中第i个Token的Query向量和第j个Token的Key向量进行相乘操作，ALiBi通过加入位置i、j的绝对值差将位置i和位置j的相对位置信息嵌入到attention计算过程中。该绝对值差是常量，可以事先计算出来，并且每个头（head）的值都有所不同。</p><p>具体公式是qikTj→qikTj−λ|i−j|qikjT→qikjT−λ|i−j|q_ik_j^T \\rightarrow q_ik_j^T - \\lambda|i-j|，其中 λλ\\lambda 是超参数，对于每一个head采用不同数值设置，论文经过实验发现对超参数 λλ\\lambda 以 121,122,123,...,128,121,122,123,...,128,\\frac{1}{2^1},\\frac{1}{2^2},\\frac{1}{2^3},...,\\frac{1}{2^8},区间进行设置效果最佳，即如果有 n 个head，则区间 λλ\\lambda 起始从2−8/n2−8/n 2^{−8/n} 开始到终点 2−82−82^{−8}。</p><p>实施过程如下图所示。</p><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211350732-1501833648.jpg" alt=""></p><p><strong>特色</strong></p><p>ALIBI是一个很朴素（当然也很有效）的光滑局部注意力技巧，但如果将它理解为“位置编码”，又并非十分妥当。</p><p>ALIBI通过线性偏置项，让自注意力分布更加关注相对距离较小，即邻近位置的语义信息，相当于增强局域性的作用。虽然其是线性偏置项，但是经过softmax变化之后，真正乘以自注意力分布上的却是指数的衰减。其次，线性的偏置意味着当相对距离很大时，偏置项趋近于负无穷。因此，ALiBi的偏置项更像是在计算注意力分数时通过一个带坡度的滑动窗口或者掩码来直接实现注意力计算过程中的远程衰减，即只能获取到训练长度内的信息。而只要相对距离足够大，ALiBi都会对其给予严格的惩罚。随着序列长度的增加，ALiBi 往往从全局注意力过渡到几乎局部的注意力，这就是为什么 ALiBi 在训练长度内的表现比大多数基线差，但在超出训练长度后表现更好的原因。</p><p>与此同时，还要注意的是，以ALiBi为代表的绝对偏置编码，无法将对 At,sAt,sA_{t,s} 的操作拆分至 qtqtq_t,ksksk_s ；而从参数角度，由于偏置 b(t-s) 是直接加在 qtqtq_t,ksksk_s 内积上的，因此对于参数矩阵 WqWqW_q, WkWkW_k ，每个特征维度间缺少区分性，相比其他两种偏置形式更难优化。</p><p>结合绝对位置编码的公式，本方案的具体推导过程如下：</p><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211400725-1484025187.jpg" alt=""></p><h3 id="_3-11-偏置编码-上下文模式" tabindex="-1">3.11 偏置编码&amp;上下文模式 <a class="header-anchor" href="#_3-11-偏置编码-上下文模式" aria-label="Permalink to &quot;3.11 偏置编码&amp;上下文模式&quot;">​</a></h3><p>分析完上述相对位置编码之后，我们看看一种分类方式。无论是绝对位置编码，还是相对位置编码，如何刻画序列不同位置间的相对距离、如何通过相对距离控制自注意力矩阵不同位置的偏置大小，一直是位置编码设计的重中之重。而不同位置编码方案添加偏置的方式则各不相同。但是基本上都可以整理成如下形式。</p>',135),l("p",{"t,s":""},"St,s=qTtks=xTtWTQWKxs+bt,sSt,s=qtTks=xtTWQTWKxs+bt,sS_{t,s} = q^T_tk_s = x_t^TW_Q^TW_Kx_s + b_",-1),r('<p>其中，bijbijb_{ij}被称为位置偏置，依据其不同形式，可以把相对位置编码分为以下两种流派。</p><ul><li>bias方案。以T5、TUPE、ALiBi为代表的位置编码，其bijbijb_{ij}是一个与qjqjq_j，kjkjk_j无关的标量。直接把偏置加qjqjq_j，kjkjk_j的内积上，直接对自注意力矩阵操作。这种方式直接将偏置加在自注意力矩阵 At,sAt,sA_{t,s} 上，计算简单理解容易，但惩罚过于绝对。</li><li>上下文方案。包括比如Transfomrer-XL，DeBERTa，其中bij=f(xi,xj,rij)bij=f(xi,xj,rij)b_{ij}=f(x_i,x_j,r_{ij})。这种方案将偏置渗透进特征向量 qtqtq_t,ksksk_s 的每个维度上，能很好区分特征维度，并以过滤取代惩罚，具有更强表达能力。但其其整体偏置易随相对位置大小波动，需要更多的维度、额外的校正才能有所缓解。</li></ul><p><img src="https://img2024.cnblogs.com/blog/1850883/202503/1850883-20250302211410626-1924445983.jpg" alt=""></p><h3 id="_3-12-小结" tabindex="-1">3.12 小结 <a class="header-anchor" href="#_3-12-小结" aria-label="Permalink to &quot;3.12 小结&quot;">​</a></h3><p>一般来说，绝对位置编码具有实现简单、计算速度快等优点，而相对位置编码则直接地体现了相对位置信号，更符合直觉也更符合文本的特性，实际效果往往也更好。如何做到集二者之所长呢？很自然的想法是，通过绝对位置编码来实现相对位置编码！也就是混合位置编码。因此，APE和RPE两者最终统一于旋转位置编码（Rotary Position Embedding，RoPE），以绝对位置编码的形式，实现了相对位置编码的效果。</p><p>注：此处分类模糊，有人把RoPE看作是混合位置编码，也有人把其归结为相对位置编码，也有人把其单独列为旋转位置编码。</p><h2 id="_0xff-参考" tabindex="-1">0xFF 参考 <a class="header-anchor" href="#_0xff-参考" aria-label="Permalink to &quot;0xFF 参考&quot;">​</a></h2><p><a href="https://arxiv.org/pdf/2312.17044.pdf" target="_blank" rel="noreferrer">Length Extrapolation of Transformers: A Survey from the Perspective of Position Encoding</a></p><p><a href="https://mp.weixin.qq.com/s/zyUDKc8VgeDvK4AQTCmfGg" target="_blank" rel="noreferrer">LLaMA长度外推高性价比trick：线性插值法及相关改进源码阅读及相关记录</a></p><p><a href="https://arxiv.org/pdf/2108.12409.pdf" target="_blank" rel="noreferrer">Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation</a></p><p><a href="https://arxiv.org/pdf/2309.12307.pdf" target="_blank" rel="noreferrer">2309.12307.pdf (arxiv.org)</a></p><p><a href="https://arxiv.org/pdf/2108.12409.pdf" target="_blank" rel="noreferrer">ALiBi</a></p><p><a href="https://kexue.fm/archives/9577" target="_blank" rel="noreferrer">Bias项的神奇作用：RoPE + Bias = 更好的长度外推性</a></p><p><a href="https://arxiv.org/pdf/2006.03654v6.pdf" target="_blank" rel="noreferrer">DeBERTa: Decoding-enhanced BERT with Disentangled Attention</a>\\</p><p><a href="https://arxiv.org/pdf/1910.10683v3.pdf" target="_blank" rel="noreferrer">Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</a></p><p><a href="https://arxiv.org/pdf/2306.15595.pdf" target="_blank" rel="noreferrer">EXTENDING CONTEXT WINDOW OF LARGE LANGUAGE MODELS VIA POSITION INTERPOLATION</a></p><p><a href="https://github.com/dvlab-research/LongLoRA" target="_blank" rel="noreferrer">GitHub - dvlab-research/LongLoRA: Code and documents of LongLoRA and LongAlpaca</a></p><p><a href="https://blog.csdn.net/weixin_44826203/article/details/129255185" target="_blank" rel="noreferrer">https://blog.csdn.net/weixin_44826203/article/details/129255185</a></p><p><a href="https://medium.com/%40ddxzzx/why-and-how-to-achieve-longer-context-windows-for-llms-5f76f8656ea9" target="_blank" rel="noreferrer">https://medium.com/@ddxzzx/why-and-how-to-achieve-longer-context-windows-for-llms-5f76f8656ea9</a></p><p><a href="https://twitter.com/GregKamradt/status/1727018183608193393" target="_blank" rel="noreferrer">https://twitter.com/GregKamradt/status/1727018183608193393</a></p><p><a href="https://mp.weixin.qq.com/s/IC5-FGLVHzHHYqH6x-aNng" target="_blank" rel="noreferrer">Kimi Chat 公布“大海捞针”长文本压测结果，也搞清楚了这项测试的精髓 (qq.com)</a></p><p><a href="https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/%3Frdt%3D44479" target="_blank" rel="noreferrer">NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation. : r/LocalLLaMA (reddit.com)</a></p><p><a href="https://arxiv.org/pdf/2006.15595.pdf" target="_blank" rel="noreferrer">RETHINKING POSITIONAL ENCODING IN LANGUAGE PRE-TRAINING</a></p><p><a href="https://arxiv.org/pdf/2104.09864.pdf" target="_blank" rel="noreferrer">RoPE</a></p><p><a href="https://zhuanlan.zhihu.com/p/660073229" target="_blank" rel="noreferrer">RoPE外推的缩放法则 —— 尝试外推RoPE至1M上下文</a> <a href="https://www.zhihu.com/people/liuxiaoran-34" target="_blank" rel="noreferrer">河畔草lxr</a></p><p><a href="https://arxiv.org/pdf/1910.10683v3.pdf" target="_blank" rel="noreferrer">T5</a></p><p><a href="https://arxiv.org/pdf/1911.04474.pdf" target="_blank" rel="noreferrer">TENER: Adapting Transformer Encoder for Named Entity Recognition</a></p><p><a href="https://arxiv.org/pdf/1901.02860v3.pdf" target="_blank" rel="noreferrer">Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context</a></p><p><a href="https://kexue.fm/archives/9859" target="_blank" rel="noreferrer">Transformer升级之路：15、Key归一化助力长度外推</a> 苏剑林</p><p><a href="https://spaces.ac.cn/archives/8231" target="_blank" rel="noreferrer">Transformer升级之路：1、Sinusoidal位置编码追根溯源 - 科学空间|Scientific Spaces</a></p><p><a href="https://kexue.fm/archives/8265" target="_blank" rel="noreferrer">Transformer升级之路：2、博采众长的旋转式位置编码</a></p><p><a href="https://kexue.fm/archives/9403" target="_blank" rel="noreferrer">Transformer升级之路：6、旋转位置编码的完备性分析</a></p><p><a href="https://arxiv.org/pdf/2006.15595v4.pdf" target="_blank" rel="noreferrer">TUPE</a></p><p><a href="https://arxiv.org/pdf/1906.08237v2.pdf" target="_blank" rel="noreferrer">XLNet</a></p><p><a href="https://arxiv.org/pdf/2212.10554.pdf" target="_blank" rel="noreferrer">xPos</a></p><p><a href="https://arxiv.org/abs/2104.09864" target="_blank" rel="noreferrer">RoFormer: Enhanced Transformer with Rotary Position Embedding</a></p><p><a href="https://arxiv.org/abs/1803.02155" target="_blank" rel="noreferrer">Self-Attention with Relative Position Representations</a></p><p><a href="https://zhuanlan.zhihu.com/p/624740065" target="_blank" rel="noreferrer">分析transformer模型的参数量、计算量、中间激活、KV cache - 知乎 (zhihu.com)</a></p><p><a href="https://mp.weixin.qq.com/s/-1xVXjoM0imXMC7DKqo-Gw" target="_blank" rel="noreferrer">图解RoPE旋转位置编码及其特性</a></p><p><a href="https://mp.weixin.qq.com/s/_SB5saeszza1Dmzs7n8iGQ" target="_blank" rel="noreferrer">大模型分布式训练并行技术（五）-序列并行 (qq.com)</a></p><p><a href="https://kexue.fm/archives/7947" target="_blank" rel="noreferrer">层次分解位置编码，让BERT可以处理超长文本</a></p><p><a href="https://zhuanlan.zhihu.com/p/572600395" target="_blank" rel="noreferrer">干货！On Position Embeddings</a> <a href="https://www.zhihu.com/people/yun-yun-14-18" target="_blank" rel="noreferrer">AI TIME</a></p><p><a href="https://blog.51cto.com/u_15588078/6531187" target="_blank" rel="noreferrer">理解Transformer的位置编码_51CTO博客_transformer的位置编码</a></p><p><a href="https://mp.weixin.qq.com/s/OysnthTQXPG_AqogQtnIcw" target="_blank" rel="noreferrer">羊驼再度进化，“长颈鹿版”LongLLaMA 来啦，上下文长度冲向 100K ，性能不减</a></p><p><a href="https://kexue.fm/archives/8130" target="_blank" rel="noreferrer">让研究人员绞尽脑汁的Transformer位置编码 - 科学空间|Scientific Spaces</a></p><p><a href="https://mp.weixin.qq.com/s/RtI95hu-ZLxGkdGuNIkERQ" target="_blank" rel="noreferrer">详解基于调整RoPE旋转角度的大模型长度外推方法 (qq.com)</a></p><p><a href="https://blog.csdn.net/qq_27590277/article/details/106264402" target="_blank" rel="noreferrer">https://blog.csdn.net/qq_27590277/article/details/106264402</a></p><p>Ke G, He D, Liu T Y. Rethinking positional encoding in language pre-training[J]. arXiv preprint arXiv:2006.15595, 2020.</p><p>本文转自 <a href="https://www.cnblogs.com/rossiXYZ/p/18746838" target="_blank" rel="noreferrer">https://www.cnblogs.com/rossiXYZ/p/18746838</a>，如有侵权，请联系删除。</p>',49)]))}const m=a(o,[["render",n]]);export{c as __pageData,m as default};
