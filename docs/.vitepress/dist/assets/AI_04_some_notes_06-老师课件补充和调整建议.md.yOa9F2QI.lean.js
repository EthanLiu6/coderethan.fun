import{_ as e,c as a,o as d,a2 as l}from"./chunks/framework.DA-Pb-tg.js";const y=JSON.parse('{"title":"Lessons Additions and Adjustments","description":"","frontmatter":{},"headers":[],"relativePath":"AI/04_some_notes/06-老师课件补充和调整建议.md","filePath":"AI/04_some_notes/06-老师课件补充和调整建议.md","lastUpdated":null}'),n={name:"AI/04_some_notes/06-老师课件补充和调整建议.md"};function r(s,t,i,o,x,c){return d(),a("div",null,t[0]||(t[0]=[l('<h1 id="lessons-additions-and-adjustments" tabindex="-1">Lessons Additions and Adjustments <a class="header-anchor" href="#lessons-additions-and-adjustments" aria-label="Permalink to &quot;Lessons Additions and Adjustments&quot;">​</a></h1><h2 id="内容补充" tabindex="-1">内容补充： <a class="header-anchor" href="#内容补充" aria-label="Permalink to &quot;内容补充：&quot;">​</a></h2><table tabindex="0"><thead><tr><th style="text-align:center;">序号</th><th style="text-align:left;">补充内容</th><th style="text-align:center;">状态</th></tr></thead><tbody><tr><td style="text-align:center;">001</td><td style="text-align:left;">add：softmax 激活函数的导数（雅可比矩阵）</td><td style="text-align:center;">0</td></tr><tr><td style="text-align:center;">002</td><td style="text-align:left;">add：001 中为啥输入的 shape 和梯度的 shape 不一样大，参数更新的时候又是怎样子的？</td><td style="text-align:center;">0</td></tr><tr><td style="text-align:center;">003</td><td style="text-align:left;">add：Norm 讲解的时候，未加入最新的 DYT（<a href="https://yiyibooks.cn/arxiv/2503.10622v1/index.html" target="_blank" rel="noreferrer">Transformers without normalization</a>）</td><td style="text-align:center;">0</td></tr><tr><td style="text-align:center;">004</td><td style="text-align:left;">add：DeepNorm 补充</td><td style="text-align:center;">0</td></tr><tr><td style="text-align:center;">005</td><td style="text-align:left;">add：PyTorch 等框架模型结构中的参数类型和数据整理（我的笔记）</td><td style="text-align:center;">0</td></tr><tr><td style="text-align:center;">006</td><td style="text-align:left;">思考：工程如何实现训练和推理不同的模块或者算子（那个 training 参数和具体的算子结构）</td><td style="text-align:center;">0</td></tr><tr><td style="text-align:center;">007</td><td style="text-align:left;"></td><td style="text-align:center;">0</td></tr></tbody></table><h2 id="调整建议" tabindex="-1">调整建议: <a class="header-anchor" href="#调整建议" aria-label="Permalink to &quot;调整建议:&quot;">​</a></h2><table tabindex="0"><thead><tr><th style="text-align:center;">序号</th><th style="text-align:left;">调整建议</th><th style="text-align:center;">状态</th></tr></thead><tbody><tr><td style="text-align:center;">001</td><td style="text-align:left;">updata：torch 的 Tensor 中，数据有 metadata 和 storage 之分（之前讲成 rawdata，但官网未使用这种叫法） <a href="https://pytorch.org/docs/stable/storage.html" target="_blank" rel="noreferrer">torch.Srorage</a></td><td style="text-align:center;">0</td></tr><tr><td style="text-align:center;">002</td><td style="text-align:left;">优化：前后知识交叉部分可以切回到原理快速回顾一下（比如：训练模式与Norm和Dropout、torch的数据结构与一些基础算子等）</td><td style="text-align:center;">0</td></tr><tr><td style="text-align:center;"></td><td style="text-align:left;"></td><td style="text-align:center;"></td></tr><tr><td style="text-align:center;"></td><td style="text-align:left;"></td><td style="text-align:center;"></td></tr><tr><td style="text-align:center;"></td><td style="text-align:left;"></td><td style="text-align:center;"></td></tr><tr><td style="text-align:center;"></td><td style="text-align:left;"></td><td style="text-align:center;"></td></tr><tr><td style="text-align:center;"></td><td style="text-align:left;"></td><td style="text-align:center;"></td></tr></tbody></table>',5)]))}const h=e(n,[["render",r]]);export{y as __pageData,h as default};
