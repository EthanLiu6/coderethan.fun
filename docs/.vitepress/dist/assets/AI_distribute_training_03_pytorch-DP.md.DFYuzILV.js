import{_ as s,c as i,o as e,a2 as t}from"./chunks/framework.DA-Pb-tg.js";const c=JSON.parse('{"title":"1. DP（DataParalle）Summary","description":"","frontmatter":{},"headers":[],"relativePath":"AI/distribute_training/03_pytorch-DP.md","filePath":"AI/distribute_training/03_pytorch-DP.md","lastUpdated":1742709547000}'),l={name:"AI/distribute_training/03_pytorch-DP.md"};function n(h,a,r,p,d,o){return e(),i("div",null,a[0]||(a[0]=[t(`<h1 id="_1-dp-dataparalle-summary" tabindex="-1">1. DP（DataParalle）Summary <a class="header-anchor" href="#_1-dp-dataparalle-summary" aria-label="Permalink to &quot;1. DP（DataParalle）Summary&quot;">​</a></h1><h2 id="数据并行的概念" tabindex="-1">数据并行的概念 <a class="header-anchor" href="#数据并行的概念" aria-label="Permalink to &quot;数据并行的概念&quot;">​</a></h2><p>当一张 GPU 可以存储一个模型时，可以采用数据并行得到更准确的梯度或者加速训练：<br> 即每个 GPU 复制一份模型，将一批样本分为多份输入各个模型并行计算。<br> 因为求导以及加和都是线性的，数据并行在数学上也有效。<br></p><h2 id="dp原理及步骤" tabindex="-1">DP原理及步骤 <a class="header-anchor" href="#dp原理及步骤" aria-label="Permalink to &quot;DP原理及步骤&quot;">​</a></h2><ul><li>Parameter Server 架构 --&gt; 单进程 多线程的方式 --&gt; 只能在单机多卡上使用;</li><li>DP 基于单机多卡，所有设备都负责计算和训练网络；</li><li>除此之外， device[0] (并非 GPU 真实标号而是输入参数 device_ids 首位) 还要负责整合梯度，更新参数。</li><li>大体步骤：</li></ul><ol><li>各卡分别计算损失和梯度；</li><li>所有梯度整合到 device[0]；</li><li>device[0] 进行参数更新，其他卡拉取 device[0] 的参数进行更新；</li></ol><p><img src="https://pic3.zhimg.com/80/v2-1cee4e8fd9e2d4dce24b0aa0a47f8c86_1440w.webp" alt="DP 原理图1"><img src="https://pic1.zhimg.com/80/v2-5c5b0d8e3d7d6653a9ebd47bac93090c_1440w.webp" alt="DP 原理图2"></p><h1 id="_2-code-implement" tabindex="-1">2. code implement <a class="header-anchor" href="#_2-code-implement" aria-label="Permalink to &quot;2. code implement&quot;">​</a></h1><h2 id="pytorch-相关源码" tabindex="-1">pytorch 相关源码 <a class="header-anchor" href="#pytorch-相关源码" aria-label="Permalink to &quot;pytorch 相关源码&quot;">​</a></h2><div class="language-python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.nn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">as</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> nn</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> nn.DataParallel(model) </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 只需要将原来单卡的 module 用 DP 改成多卡</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">class</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> DataParallel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">Module</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">):</span></span></code></pre></div><h2 id="train-mode-use-pytorch-dp" tabindex="-1">train mode use pytorch DP <a class="header-anchor" href="#train-mode-use-pytorch-dp" aria-label="Permalink to &quot;train mode use pytorch DP&quot;">​</a></h2><p><strong>运行 dp_hello.py</strong></p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">python</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> dp_hello.py</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">output:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&gt;&gt;&gt; </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">output:</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Let&#39;s use 2 GPUs!</span></span></code></pre></div><p><strong>运行 dp_demo.py</strong></p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">python</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> dp_demo.py</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">result:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&gt;&gt;&gt; </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">data</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> shape:</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">  torch.Size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([64, 1, 28, 28])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&gt;&gt;&gt;  </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">=============x</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> shape:</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">  torch.Size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([32, 1, 28, 28])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&gt;&gt;&gt; </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">=============x</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> shape:</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">  torch.Size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([32, 1, 28, 28])</span></span></code></pre></div><h1 id="_3-dp-的优缺点" tabindex="-1">3. DP 的优缺点 <a class="header-anchor" href="#_3-dp-的优缺点" aria-label="Permalink to &quot;3. DP 的优缺点&quot;">​</a></h1><ul><li>负载不均衡：device[0] 负载大一些；</li><li>通信开销大；</li><li>单进程；</li><li>Global Interpreter Lock (GIL)全局解释器锁，简单来说就是，一个 Python 进程只能利用一个 CPU kernel，<br> 即单核多线程并发时，只能执行一个线程。考虑多核，多核多线程可能出现线程颠簸 (thrashing) 造成资源浪费，<br> 所以 Python 想要利用多核最好是多进程。<br></li></ul><h1 id="_4-references" tabindex="-1">4. [references] <a class="header-anchor" href="#_4-references" aria-label="Permalink to &quot;4. [references]&quot;">​</a></h1><ol><li><a href="https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/data_parallel.py" target="_blank" rel="noreferrer">pytorch 源码</a></li><li><a href="https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html?highlight=data+parallel#torch.nn.DataParallel" target="_blank" rel="noreferrer">torch.nn.DataParallel</a></li><li><a href="https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#create-model-and-dataparallel" target="_blank" rel="noreferrer">代码参考链接</a></li><li><a href="https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/notes/cuda.html%3Fhighlight%3Dbuffer" target="_blank" rel="noreferrer">DP 和 DDP</a></li></ol>`,19)]))}const g=s(l,[["render",n]]);export{c as __pageData,g as default};
