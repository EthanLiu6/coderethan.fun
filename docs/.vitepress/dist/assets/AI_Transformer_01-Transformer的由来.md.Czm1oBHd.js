import{_ as a,c as e,o as t,a2 as n}from"./chunks/framework.DA-Pb-tg.js";const u=JSON.parse('{"title":"01-Transformer的由来","description":"","frontmatter":{},"headers":[],"relativePath":"AI/Transformer/01-Transformer的由来.md","filePath":"AI/Transformer/01-Transformer的由来.md","lastUpdated":1742269800000}'),l={name:"AI/Transformer/01-Transformer的由来.md"};function i(p,s,r,o,m,c){return t(),e("div",null,s[0]||(s[0]=[n('<h1 id="_01-transformer的由来" tabindex="-1">01-Transformer的由来 <a class="header-anchor" href="#_01-transformer的由来" aria-label="Permalink to &quot;01-Transformer的由来&quot;">​</a></h1><blockquote><p>整理来源于作者：<a href="https://www.cnblogs.com/rossiXYZ/p/18706134%EF%BC%8C%E5%B7%B2%E8%8E%B7%E5%BE%97%E8%AE%B8%E5%8F%AF" target="_blank" rel="noreferrer">https://www.cnblogs.com/rossiXYZ/p/18706134，已获得许可</a></p></blockquote><h2 id="_1-引入" tabindex="-1">1. 引入 <a class="header-anchor" href="#_1-引入" aria-label="Permalink to &quot;1. 引入&quot;">​</a></h2><p>Transformer本身还是seq2seq结构的一个模型架构，但像RNN这样的网络他有很多问题点，每一次的预测输出是建立在上一次的输出基础上的，也算是早起自回归模型的问题点：</p><blockquote><ul><li>串行运行，很难以并行化的方式开展训练、提升效率。</li><li>只有<span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>h</mi><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub></mrow><annotation encoding="application/x-tex">h_{t-1}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.69444em;"></span><span class="strut bottom" style="height:0.902771em;vertical-align:-0.208331em;"></span><span class="base textstyle uncramped"><span class="mord"><span class="mord mathit">h</span><span class="vlist"><span style="top:0.15em;margin-right:0.05em;margin-left:0em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle scriptstyle cramped"><span class="mord scriptstyle cramped"><span class="mord mathit">t</span><span class="mbin">−</span><span class="mord mathrm">1</span></span></span></span><span class="baseline-fix"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span>​</span></span></span></span></span></span>时刻的信息，容易丢失信息</li><li>&quot;一步错，步步错&quot;</li><li>梯度消失问题</li><li>输入输出序列等长限制（n2n）</li></ul></blockquote><p>由此，研究者们就在此基础上设计出了一种&quot;优化版的Encoder 2 Decoder&quot;的架构，其中的过渡就是Context Vector (用C表示)，<strong>输入句子每个时间步的信息</strong>都包含在了这个上下文中。简单理解可以认为Encoder进行特征提取得到输入信息对应的Context Vector，然后Decoder进行解码：</p><blockquote><ul><li>在每个时刻，解码器都是自回归的，即上一个时刻的输出（产生的token y_{t−1}）作为下当前时刻<span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>t</mi></mrow><annotation encoding="application/x-tex">t</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.61508em;"></span><span class="strut bottom" style="height:0.61508em;vertical-align:0em;"></span><span class="base textstyle uncramped"><span class="mord mathit">t</span></span></span></span>的输入之一，生成当前时刻的token <span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>y</mi><mi>t</mi></msub></mrow><annotation encoding="application/x-tex">y_t</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.43056em;"></span><span class="strut bottom" style="height:0.625em;vertical-align:-0.19444em;"></span><span class="base textstyle uncramped"><span class="mord"><span class="mord mathit" style="margin-right:0.03588em;">y</span><span class="vlist"><span style="top:0.15em;margin-right:0.05em;margin-left:-0.03588em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle scriptstyle cramped"><span class="mord mathit">t</span></span></span><span class="baseline-fix"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span>​</span></span></span></span></span></span>。</li><li>解码器最初的输入是中间语义上下文向量C，解码器依据C计算出第一个输出词和新的隐状态，即解码器的每个预测都受到先前输出词和隐状态的微妙影响。</li><li>解码器接着用新的隐状态和第一个输出词作为联合输入来计算第二个输出词，以此类推，直到解码器产生一个 EOS（End Of Service/序列结束）标记或者达到预定序列长度的边界。</li></ul></blockquote><p>从宏观角度看，序列建模的核心就是研究如何把长序列的上下文压缩到一个较小的状态中（好好领悟这句话）。</p><p>咋压缩呢？早期的有马尔可夫假设，也就是近因效应，如果考虑前面n个单词，这就得到了N-gram模型，即当前单词的概率取决于前n个单词。</p><h2 id="_2-attention机制" tabindex="-1">2. Attention机制 <a class="header-anchor" href="#_2-attention机制" aria-label="Permalink to &quot;2. Attention机制&quot;">​</a></h2><p>这个在transformer之前就有了，它其实有一定的实际意义，有三种主流禅诗：</p><blockquote><ul><li>注意力机制的本质是上下文决定一切。</li><li>注意力机制是一种资源分配方案。</li><li>注意力机制是信息交换，或者说是是“全局信息查询”。</li></ul></blockquote><blockquote><p>其实，论文“Recurrent Models of Visual Attention”中有一段话就深刻的印证了资源分配这个角度。具体如下：人类感知的一个重要特性是，人们不会一次处理整个场景。相反，人类有选择地将注意力集中在视觉空间的某些部分上，以在需要的时间和地点获取信息，并随着时间的推移将不同注视点的信息结合起来，建立场景的内部表示，指导未来的眼球运动和决策。将计算资源集中在场景的各个部分可以节省“带宽”，因为需要处理的“像素”更少。但它也大大降低了任务的复杂性，因为感兴趣的对象可以放置在注视的中心，而注视区域外的视觉环境的无关特征（“杂乱”）自然会被忽略。</p></blockquote><p>所以，核心就是分配不同的权重，那又引出问题：</p><ul><li>在哪里做注意力计算？</li><li>如何做注意力计算？</li></ul><h2 id="_3-q-k-v的引入" tabindex="-1">3. Q K V的引入 <a class="header-anchor" href="#_3-q-k-v的引入" aria-label="Permalink to &quot;3. Q K V的引入&quot;">​</a></h2><p>注意力模型的内部流程如下图所示，该模型的目标是生成V中向量的加权平均值，具体计算流程如下。</p><ul><li><p>标号1是输入（两个输入），从输入生成的特征向量F会进一步生成键矩阵K和值矩阵V。</p></li><li><p>标号2使用矩阵K和查询向量q作为输入，通过相似度计算函数来计算注意力得分向量e。q表示对信息的请求，<span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>e</mi><mi>l</mi></msub></mrow><annotation encoding="application/x-tex">e_l</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.43056em;"></span><span class="strut bottom" style="height:0.58056em;vertical-align:-0.15em;"></span><span class="base textstyle uncramped"><span class="mord"><span class="mord mathit">e</span><span class="vlist"><span style="top:0.15em;margin-right:0.05em;margin-left:0em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle scriptstyle cramped"><span class="mord mathit" style="margin-right:0.01968em;">l</span></span></span><span class="baseline-fix"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span>​</span></span></span></span></span></span>表示矩阵K的第l列对于q的重要性。</p></li><li><p>标号3通过对齐层（比如softmax函数）进一步处理注意力分数，进而得到注意力权重a。</p></li><li><p>标号4利用注意力权重a和矩阵V进行计算，得到上下文向量c。</p><p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/1850883-20250209102028813-842747599.jpg" alt="img"></p><p>上图注意力模型中，有两个输入：q（正在处理的序列）和F（被关注的序列），F又分别转换为K和V，这三个变量综合起来使用就可以满足我们的需求。</p><p>从词典的角度来看也许可以促进理解。query是你要找的内容，key是字典的索引（字典里面有什么样的信息），value是对应的信息。</p><p>我们用淘宝搜索来类比，可以帮助我们对这些矩阵有更好的理解。假如我们在淘宝上进搜索”李宁鞋“。</p><ul><li>query是你在搜索栏输入的查询内容。</li><li>key是在页面上返回的商品描述、标题，其实就是淘宝商品数据库中与候选商品相关的关键字。</li><li>value是李宁鞋商品本身。因为一旦依据搜索词（query）搜到了匹配的商品描述、标题（key），我们就希望具体看看商品内容。</li></ul><p>通过使用这些 QKV 值，模型可以计算注意力分数，从而确定每个token在生成预测时应从其它token那里获得多少关注。</p><p><strong>注意力机制的计算总体可以分为两步：</strong></p><ol><li>在所有输入信息上计算注意力分布。编码器不只是传递最后一个隐藏状态，而是传入所有的隐藏状态到解码器。</li><li>根据注意力分布来计算输入信息的加权平均。需要注意，这是一种数据依赖的加权平均，是一种灵活、高效的全局池化操作。</li></ol></li></ul>',18)]))}const d=a(l,[["render",i]]);export{u as __pageData,d as default};
