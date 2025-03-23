import{_ as a,c as r,o as l,a2 as t}from"./chunks/framework.DA-Pb-tg.js";const m=JSON.parse('{"title":"1. Tensor parallel","description":"","frontmatter":{},"headers":[],"relativePath":"AI/distribute_training/21_SP-Theory.md","filePath":"AI/distribute_training/21_SP-Theory.md","lastUpdated":1742709547000}'),o={name:"AI/distribute_training/21_SP-Theory.md"};function n(s,e,i,p,c,d){return l(),r("div",null,e[0]||(e[0]=[t('<h1 id="_1-tensor-parallel" tabindex="-1">1. Tensor parallel <a class="header-anchor" href="#_1-tensor-parallel" aria-label="Permalink to &quot;1. Tensor parallel&quot;">​</a></h1><p>如下图所示，张量并行化了 Transformer 层中在训练期间占用大部分时间的部件，因此它在计算上是高效的。 但是，它保留了注意力和 MLP 模块之后的<strong>layernorm以及dropout</strong>，因此它们在张量并行组中被复制。 这些元素不需要大量的计算，但<code>需要大量的激活内存</code>, 因为张量并行对他们无效。<br></p><p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/tensor-parallel.png" alt="images"></p><h1 id="_2-sequece-parallel" tabindex="-1">2. Sequece parallel <a class="header-anchor" href="#_2-sequece-parallel" aria-label="Permalink to &quot;2. Sequece parallel&quot;">​</a></h1><p>我们注意到在 Transformer 层的非张量并行区域中，操作<strong>在序列维度上是独立的</strong>。 这种特性允许我们在序列维度上对这些区域进行划分。 沿着序列维度进行划分减少了激活所需的内存。 这种额外的并行级别在TP前后通讯外引入了新的通信集合，它们将充当<code>序列和张量并行区域之间的转换器</code>。 这些额外的通信引入了开销，并会减慢训练速度。<br></p><p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/sequence-parallel.png" alt="images"></p><h1 id="_3-mlp-tp-and-sp-special" tabindex="-1">3. mlp TP and SP special <a class="header-anchor" href="#_3-mlp-tp-and-sp-special" aria-label="Permalink to &quot;3. mlp TP and SP special&quot;">​</a></h1><p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/mlp-tensor-sequence-parallel.png" alt="images"></p><h1 id="_4-通讯开销" tabindex="-1">4 通讯开销 <a class="header-anchor" href="#_4-通讯开销" aria-label="Permalink to &quot;4 通讯开销&quot;">​</a></h1><p>        张量并行在单个正向和反向传播中需要四个全归约，而张量与序列并行在单个正向和反向传播中需要四个全聚合和四个归约散射。 乍一看，似乎张量与序列并行相比张量并行需要更多的通信。 然而，我们注意到环形全归约包含两个步骤：归约散射后跟全聚合。 因此，张量并行和张量与序列并行使用的通信带宽相同。 因此，序列并行不会引入任何通信开销。<br></p><h1 id="_5-参考连接" tabindex="-1">5 参考连接 <a class="header-anchor" href="#_5-参考连接" aria-label="Permalink to &quot;5 参考连接&quot;">​</a></h1><ul><li><a href="https://arxiv.org/pdf/2205.05198" target="_blank" rel="noreferrer">论文连接-EN</a></li><li><a href="https://yiyibooks.cn/arxiv/2205.05198v1/index.html" target="_blank" rel="noreferrer">论文连接-EN</a></li></ul>',12)]))}const u=a(o,[["render",n]]);export{m as __pageData,u as default};
