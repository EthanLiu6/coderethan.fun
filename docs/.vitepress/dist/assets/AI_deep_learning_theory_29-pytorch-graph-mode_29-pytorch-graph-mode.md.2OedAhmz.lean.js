import{_ as i,c as a,o as n,a2 as h}from"./chunks/framework.DA-Pb-tg.js";const y=JSON.parse('{"title":"1 pytorch 几种模式概览","description":"","frontmatter":{},"headers":[],"relativePath":"AI/deep_learning_theory/29-pytorch-graph-mode/29-pytorch-graph-mode.md","filePath":"AI/deep_learning_theory/29-pytorch-graph-mode/29-pytorch-graph-mode.md","lastUpdated":null}'),t={name:"AI/deep_learning_theory/29-pytorch-graph-mode/29-pytorch-graph-mode.md"};function p(l,s,k,e,r,E){return n(),a("div",null,s[0]||(s[0]=[h(`<h1 id="_1-pytorch-几种模式概览" tabindex="-1">1 pytorch 几种模式概览 <a class="header-anchor" href="#_1-pytorch-几种模式概览" aria-label="Permalink to &quot;1 pytorch 几种模式概览&quot;">​</a></h1><h2 id="_1-1-pytorch-不仅仅是动态图" tabindex="-1">1.1 pytorch 不仅仅是动态图 <a class="header-anchor" href="#_1-1-pytorch-不仅仅是动态图" aria-label="Permalink to &quot;1.1 pytorch 不仅仅是动态图&quot;">​</a></h2><p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/pytorch-patterns-figure1.jpg" alt="figure1"></p><h2 id="_1-2-理解动态图和静态图" tabindex="-1">1.2 理解动态图和静态图 <a class="header-anchor" href="#_1-2-理解动态图和静态图" aria-label="Permalink to &quot;1.2 理解动态图和静态图&quot;">​</a></h2><p>         静态图（Static Graph）和动态图（Dynamic Graph）是两种不同的计算图表示形式，用于描述深度学习模型的计算过程。<br></p><p>         静态图是在模型定义阶段构建的计算图，其中所有的操作和数据流都在定义时确定。它以一种静态的方式表示模型的结构和计算逻辑，并且在执行过程中不会改变。静态图的一个典型示例是TensorFlow的计算图。在静态图中，模型的计算图在执行前需要显式地进行编译和优化，然后可以重复地执行编译后的图形，以获得更高的执行效率。<br></p><p>         动态图是在模型执行阶段构建的计算图，其中计算图是根据实际输入数据的流动而动态生成的。它以一种动态的方式表示模型的计算过程，并且可以根据不同的输入数据进行灵活的控制流操作。动态图的一个典型示例是PyTorch的计算图。在动态图中，模型的计算图是根据实际输入数据动态构建的，这样可以更灵活地处理变长序列数据、条件分支和循环等情况。<br></p><p>         静态图和动态图各有其优势和应用场景。静态图在模型执行前可以进行静态优化和编译，从而在执行过程中获得更高的性能。它适用于具有固定结构和无需动态控制流的模型。动态图更灵活，能够处理动态控制流和变长输入数据等情况，适用于具有动态特性和复杂逻辑的模型。由于动态图可以根据实际输入数据生成计算图，因此它在模型的调试和开发过程中更具交互性和可读性。<br></p><h2 id="_1-3-静态图的优势" tabindex="-1">1.3 静态图的优势 <a class="header-anchor" href="#_1-3-静态图的优势" aria-label="Permalink to &quot;1.3 静态图的优势&quot;">​</a></h2><ul><li><p>图优化：<br>          静态图允许PyTorch在转换过程中应用一系列的图优化技术。这些优化包括常量折叠、死代码消除、运算融合、内存优化等，可以减少不必要的计算、内存消耗和数据传输，从而提高模型的执行效率。<br></p></li><li><p>编译优化：<br>          静态图需要在转换过程中进行编译，将动态操作转换为优化的静态内核。这种编译过程可以包括静态类型推断、代码生成和优化等步骤，可以提前优化模型的计算图，并生成高效的机器代码。<br></p></li></ul><h1 id="_2-几种模式简介" tabindex="-1">2 几种模式简介 <a class="header-anchor" href="#_2-几种模式简介" aria-label="Permalink to &quot;2 几种模式简介&quot;">​</a></h1><h2 id="_2-1-fx-图" tabindex="-1">2.1 fx 图 <a class="header-anchor" href="#_2-1-fx-图" aria-label="Permalink to &quot;2.1 fx 图&quot;">​</a></h2><p>         在PyTorch中，FX图（FX Graph）是一种中间表示形式，用于表示和转换PyTorch模型。FX图是一种基于符号执行的图表示，类似于抽象语法树（AST），用于表示模型的计算图及其相关操作。FX图本身并不能直接运行。FX图是一种中间表示形式，用于表示和转换PyTorch模型，而非用于实际的模型执行。<br></p><h2 id="_2-2-torch-jit-script" tabindex="-1">2.2 torch.jit.script <a class="header-anchor" href="#_2-2-torch-jit-script" aria-label="Permalink to &quot;2.2 torch.jit.script&quot;">​</a></h2><p>         jit.script通过静态分析和推断来捕获模型的计算图。具体来说，当使用jit.script装饰器标记模型的函数时，PyTorch会执行以下步骤来进行捕获：<br></p><ul><li><p>静态分析：<br>          PyTorch会对模型函数进行静态分析，分析模型的结构、操作和控制流等信息。它会检查函数中的各种操作符和模块，并构建一个计算图的表示形式。<br></p></li><li><p>类型推断：<br>          PyTorch会进行类型推断，根据输入和操作的语义来确定操作的数据类型。这有助于生成正确的计算图，并在后续的转换过程中进行类型检查。<br></p></li><li><p>生成脚本：<br>          根据分析和推断的结果，PyTorch会生成一个独立的可运行脚本，其中包含模型的结构和计算逻辑。这个脚本是模型的静态表示形式，不再依赖于动态图和Python解释器。<br></p></li></ul><h2 id="_2-3-torch-jit-trace" tabindex="-1">2.3 torch.jit.trace <a class="header-anchor" href="#_2-3-torch-jit-trace" aria-label="Permalink to &quot;2.3 torch.jit.trace&quot;">​</a></h2><p>         jit.trace：jit.trace是将PyTorch模型转换为跟踪形式的工具。它通过跟踪模型的输入和输出，捕获模型的执行轨迹，并将其转换为一个可重放的跟踪模型。这个跟踪模型可以在后续的推理过程中重复使用，避免了重复的图构建和优化过程。使用jit.trace时，通常需要提供示例输入来触发模型的执行并捕获跟踪。<br></p><h2 id="_2-4-torch-compile" tabindex="-1">2.4 torch.compile <a class="header-anchor" href="#_2-4-torch-compile" aria-label="Permalink to &quot;2.4 torch.compile&quot;">​</a></h2><p>         torch.compile是最新的方法，可以加速您的PyTorch代码！torch.compile通过即时编译(PyTorch代码编译成优化内核)来使PyTorch代码运行更快，同时需要最小的代码更改。<br></p><h1 id="_3-案例" tabindex="-1">3 案例： <a class="header-anchor" href="#_3-案例" aria-label="Permalink to &quot;3 案例：&quot;">​</a></h1><h2 id="_3-1-模型准备" tabindex="-1">3.1 模型准备 <a class="header-anchor" href="#_3-1-模型准备" aria-label="Permalink to &quot;3.1 模型准备&quot;">​</a></h2><div class="language-python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 定义数据集</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">class</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> MyDataset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">data</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">Dataset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">):</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    def</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> __init__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(self):</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # fake data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        self</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.randn(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        self</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.target </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.randint(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        </span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    def</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> __getitem__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(self, index):</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> self</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.data[index], </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">self</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.target[index]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    </span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    def</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> __len__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(self):</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> len</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">self</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.data)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 定义模型</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">class</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> MyModel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">nn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">Module</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">):</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    def</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> __init__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(self):</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        super</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(MyModel, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">self</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">).</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">__init__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        self</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.fc1 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> nn.Linear(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        self</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.fc2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> nn.Linear(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        </span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    def</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> forward</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(self, x):</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> self</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.fc1(x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> nn.functional.relu(x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> self</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.fc2(x)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 定义训练函数</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">def</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, dataset, optimizer, criterion, epochs):</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> range</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(epochs):</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        running_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        correct </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        </span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data, target </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataset:</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">            # 将数据和目标转换为Tensor类型</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.tensor(data)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            target </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.tensor(target)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">            # 前向传播</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            output </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> model(data)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> criterion(output, target)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">            # 反向传播和优化</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            optimizer.zero_grad()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss.backward()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            optimizer.step()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">            # 统计损失和正确率</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            running_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss.item()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            _, predicted </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.max(output.data, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            total </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> target.size(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            correct </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (predicted </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> target).sum().item()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # 打印训练信息</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        print</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;Epoch [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">{}</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">{}</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">], Loss: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">{</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:.4f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">}</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">, Accuracy: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">{</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:.2f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">}</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">%&#39;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">              .format(epoch</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, epochs, running_loss</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">len</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">100</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">correct</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">total))</span></span></code></pre></div><h2 id="_3-2-jit-script-代码展示" tabindex="-1">3.2 jit.script 代码展示 <a class="header-anchor" href="#_3-2-jit-script-代码展示" aria-label="Permalink to &quot;3.2 jit.script 代码展示&quot;">​</a></h2><div class="language-python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">def</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> script_demo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">():</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # 实例化模型</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MyModel()</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # 将模型转换为script模式</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    scripted_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.jit.script(model)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # 定义优化器和损失函数</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    optimizer </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> optim.SGD(scripted_model.parameters(), </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">lr</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    criterion </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> nn.CrossEntropyLoss()    </span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # 实例化数据集和数据加载器</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dataset </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MyDataset()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dataloader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data.DataLoader(dataset, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">batch_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">True</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # 调用训练函数</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train(scripted_model, dataloader, optimizer, criterion, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">epochs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    scripted_model.save(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;scripted_model.pt&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">def</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> run_script_model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">():</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.jit.load(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;scripted_model.pt&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    output </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> model(torch.rand(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    print</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;output: &quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, output)</span></span></code></pre></div><h2 id="_3-3-jit-traced-代码展示" tabindex="-1">3.3 jit.traced 代码展示 <a class="header-anchor" href="#_3-3-jit-traced-代码展示" aria-label="Permalink to &quot;3.3 jit.traced 代码展示&quot;">​</a></h2><div class="language-python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">def</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> traced_demo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">():</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MyModel()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    traced_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.jit.trace(model, torch.randn(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # 保存模型到文件</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    traced_model.save(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;traced_model.pt&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # 重新加载模型</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loaded_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.jit.load(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;traced_model.pt&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # 重新运行模型</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    input_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.randn(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    output_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loaded_model(input_data)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    print</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;traced model output: &quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, output_data)</span></span></code></pre></div><h1 id="_4-export-to-onnx" tabindex="-1">4 export to onnx <a class="header-anchor" href="#_4-export-to-onnx" aria-label="Permalink to &quot;4 export to onnx&quot;">​</a></h1><div class="language-python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">def</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> onnx_demo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">():</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MyModel()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    torch.onnx.export(model, torch.randn(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;onnx_model.onnx&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       </span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">def</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> onnx_infer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">():</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    input</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.randn(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # 加载模型并运行</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> onnxruntime </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">as</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ort</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ort_session </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ort.InferenceSession(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;onnx_model.onnx&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 加载模型到 session中</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ort_inputs </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {ort_session.get_inputs()[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">].name: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">input</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.numpy()} </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 设置我们input --&gt; numpy 格式的数据</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ort_outputs </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ort_session.run(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">None</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ort_inputs) </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 开始run --&gt; outputs --</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    print</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;onnx run output: &quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ort_outputs[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]) </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 取出结果</span></span></code></pre></div><h1 id="_5-compile-to-graph-dynamo" tabindex="-1">5 compile to graph (dynamo) <a class="header-anchor" href="#_5-compile-to-graph-dynamo" aria-label="Permalink to &quot;5 compile to graph (dynamo)&quot;">​</a></h1><div class="language-python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># Reset since we are using a different mode.</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch._dynamo</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">torch._dynamo.reset()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 方式一：      </span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">def</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> train</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, dataloader):</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.compile(model) </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 是有开销的</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> batch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dataloader:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        run_epoch(model, batch)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    def</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> infer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, input):</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.compile(model)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> model(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">input</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    </span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 方式二：</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">@torch.compile</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">def</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> opt_foo2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y):</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    a </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.sin(x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    b </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> torch.cos(y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> a </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> b</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">print</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(opt_foo2(torch.randn(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), torch.randn(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span></code></pre></div>`,31)]))}const g=i(t,[["render",p]]);export{y as __pageData,g as default};
