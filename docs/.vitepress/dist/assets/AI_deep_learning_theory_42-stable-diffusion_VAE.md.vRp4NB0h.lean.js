import{_ as s,c as t,o as e,a2 as n}from"./chunks/framework.DA-Pb-tg.js";const d=JSON.parse('{"title":"VAE","description":"","frontmatter":{},"headers":[],"relativePath":"AI/deep_learning_theory/42-stable-diffusion/VAE.md","filePath":"AI/deep_learning_theory/42-stable-diffusion/VAE.md","lastUpdated":1739797437000}'),p={name:"AI/deep_learning_theory/42-stable-diffusion/VAE.md"};function m(r,a,l,i,o,c){return e(),t("div",null,a[0]||(a[0]=[n('<h1 id="vae" tabindex="-1">VAE <a class="header-anchor" href="#vae" aria-label="Permalink to &quot;VAE&quot;">​</a></h1><ul><li><a href="https://arxiv.org/pdf/1312.6114.pdf" target="_blank" rel="noreferrer">论文链接</a></li></ul><h1 id="_1-vae-的作用-数据压缩和数据生成" tabindex="-1">1 VAE 的作用 （数据压缩和数据生成） <a class="header-anchor" href="#_1-vae-的作用-数据压缩和数据生成" aria-label="Permalink to &quot;1 VAE 的作用 （数据压缩和数据生成）&quot;">​</a></h1><h2 id="_1-1-数据压缩" tabindex="-1">1.1 数据压缩 <a class="header-anchor" href="#_1-1-数据压缩" aria-label="Permalink to &quot;1.1 数据压缩&quot;">​</a></h2><p>        数据压缩也可以成为数据降维，一般情况下数据的维度都是高维的，比如手写数字（28*28=784维），如果数据维度的输入，机器的处理量将会很大， 而数据经过降维以后，如果保留了原有数据的主要信息，那么我们就可以用降维的数据进行机器学习模型的训练和预测，由于数据量大大缩减，训练和预测的时间效率将大为提高。还有一种好处就是我们可以将数据降维至2D或3D以便于观察分布情况。<br>         平常最常用到的就是PCA（主成分分析法：将原来的三维空间投影到方差最大且线性无关的两个方向或者说将原矩阵进行单位正交基变换以保留最大的信息量）。<br></p><p><img src="https://img2022.cnblogs.com/blog/2679798/202201/2679798-20220119160204780-585362428.png" alt="figure1"></p><h2 id="_1-2-数据生成" tabindex="-1">1.2 数据生成 <a class="header-anchor" href="#_1-2-数据生成" aria-label="Permalink to &quot;1.2 数据生成&quot;">​</a></h2><p>        近年来最火的生成模型莫过于GAN和VAE，这两种模型在实践中体现出极好的性能。<br>         所谓数据的生成，就是经过样本训练后，<strong>人为输入或随机输入数据</strong>，得到一个类似于样本的结果。<br>         比如样本为很多个人脸，生成结果就是一些人脸，但这些人脸是从未出现过的全新的人脸。又或者输入很多的手写数字，得到的结果也是一些手写数字。而给出的数据可能是一个或多个随机数，或者一个分布。然后经过神经网络，将输入的数据进行放大，得到结果。<br></p><h2 id="_1-3-数据压缩与数据生成的关系" tabindex="-1">1.3 数据压缩与数据生成的关系 <a class="header-anchor" href="#_1-3-数据压缩与数据生成的关系" aria-label="Permalink to &quot;1.3 数据压缩与数据生成的关系&quot;">​</a></h2><p>        在数据生成过程中要输入一些数进去，可是这些数字<strong>不能是随随便便的数字</strong>吧，至少得有一定的规律性才能让神经网络进行学习(就像要去破译密码，总得知道那些个密码符号表示的含义是什么才可以吧)。<br>         那如何获得输入数字（或者说密码）的规律呢。这就是数据压缩过程我们所要考虑的问题，我们想要获得数据经过压缩后满足什么规律，在VAE中，我们将这种规律用概率的形式表示。在经过一系列数学研究后：我们最终获得了<strong>数据压缩的分布规律</strong>，这样我们就可以<strong>根据这个规律去抽取样本进行生成</strong>，生成的结果一定是类似于样本的数据。<br></p><p><img src="https://img2022.cnblogs.com/blog/2679798/202201/2679798-20220119160204832-1264950075.png" alt="figure2"></p><h2 id="_1-4-example" tabindex="-1">1.4 example <a class="header-anchor" href="#_1-4-example" aria-label="Permalink to &quot;1.4 example&quot;">​</a></h2><p>        在前面讲解过，将图片进行某种编码，我们将原来 28*28 = 784 维的图片编码为2维的高斯分布(也可以不是2维，只是为了好可视化), 二维平面的中心就是图片的二维高斯分布的 μ(1) 和 μ(2) ，表示椭圆的中心(注意：这里其实不是椭圆，我们只是把最较大概率的部分框出来)。<br></p><p>        假设一共有5个图片(手写数字0-4)，则在隐空间中一共有5个二维正态分布（椭圆），如果生成过程中<strong>在坐标中取的点</strong>接近蓝色区域，则说明，最后的生成结果接近数字0，如果在蓝色和黑色交界处，则结果介于0和1之间。<br></p><p><img src="https://img2022.cnblogs.com/blog/2679798/202201/2679798-20220119163940347-229172341.png" alt="figure3"></p><h2 id="_1-5-可能出现的问题" tabindex="-1">1.5 可能出现的问题 <a class="header-anchor" href="#_1-5-可能出现的问题" aria-label="Permalink to &quot;1.5 可能出现的问题&quot;">​</a></h2><p><strong>问题</strong>：如果每个椭圆离得特别远会发生什么？？？ <br></p><p><strong>答案</strong>：椭圆之间完全没有交集。<br></p><p><strong>结果</strong>：假如随机取数据的时候，<strong>取的数据不在任何椭圆里</strong>，最后的生成的结果将会非常离谱，根本不知道生成模型生成了什么东西，我们称这种现象为过拟合，因此，我们必须要让这些个椭圆<strong>尽可能的推叠在一起</strong>，并且<strong>尽可能占满整个空间的位置</strong>，防止生成不属于任何分类的图片。后面我们会介绍如何将椭圆尽可能堆叠。<br></p><p>        在解决上面问题后，我们就得到了一个较为标准的数据压缩形态，这样我们就可以放心采样进行数据生成。<br></p><h2 id="_1-6-vae-要点总结" tabindex="-1">1.6 VAE 要点总结 <a class="header-anchor" href="#_1-6-vae-要点总结" aria-label="Permalink to &quot;1.6 VAE 要点总结&quot;">​</a></h2><p>        到现在为止，VAE框架已经形成: <br></p><ul><li>隐空间(latent space)有规律可循，长的像的图片离得近; <br></li><li>隐空间随便拿个点解码之后，得到的点<strong>有意义</strong>; <br></li><li>隐空间中对应不同标签的点不会离得很远，但也不会离得太近（因为每个高斯的中心部分因为被采样次数多必须特色鲜明，不能跟别的类别的高斯中心离得太近）（VAE做生成任务的基础）; <br></li><li>隐空间对应相同标签的点离得比较近，但又不会聚成超小的小簇，然而也不会有相聚甚远的情况（VAE做分类任务的基础）; <br></li></ul><h1 id="_2-理论推导vae" tabindex="-1">2 理论推导VAE <a class="header-anchor" href="#_2-理论推导vae" aria-label="Permalink to &quot;2 理论推导VAE&quot;">​</a></h1><p>        怎么去求那么复杂的高斯分布也就是隐空间呢??? 这个问题与变分推断遇到的几乎一样。<br></p><h2 id="_2-1-引入变分" tabindex="-1">2.1 引入变分 <a class="header-anchor" href="#_2-1-引入变分" aria-label="Permalink to &quot;2.1 引入变分&quot;">​</a></h2><p>        在变分推断中，我们想要通过样本x来估计关于z的分布，也就是后验，用概率的语言描述就是：p(z|x)。根据贝叶斯公式：<br></p><p><span class="katex-display"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>p</mi><mo>(</mo><mi>z</mi><mo>∣</mo><mi>x</mi><mo>)</mo><mo>=</mo><mfrac><mrow><mi>p</mi><mo>(</mo><mi>x</mi><mo>∣</mo><mi>z</mi><mo>)</mo><mi>p</mi><mo>(</mo><mi>z</mi><mo>)</mo></mrow><mrow><mi>p</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mfrac></mrow><annotation encoding="application/x-tex">p(z \\mid x)=\\frac{p(x \\mid z) p(z)}{p(x)} </annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:1.427em;"></span><span class="strut bottom" style="height:2.363em;vertical-align:-0.936em;"></span><span class="base displaystyle textstyle uncramped"><span class="mord mathit">p</span><span class="mopen">(</span><span class="mord mathit" style="margin-right:0.04398em;">z</span><span class="mrel">∣</span><span class="mord mathit">x</span><span class="mclose">)</span><span class="mrel">=</span><span class="mord reset-textstyle displaystyle textstyle uncramped"><span class="sizing reset-size5 size5 reset-textstyle textstyle uncramped nulldelimiter"></span><span class="mfrac"><span class="vlist"><span style="top:0.686em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle textstyle cramped"><span class="mord textstyle cramped"><span class="mord mathit">p</span><span class="mopen">(</span><span class="mord mathit">x</span><span class="mclose">)</span></span></span></span><span style="top:-0.2300000000000001em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle textstyle uncramped frac-line"></span></span><span style="top:-0.677em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle textstyle uncramped"><span class="mord textstyle uncramped"><span class="mord mathit">p</span><span class="mopen">(</span><span class="mord mathit">x</span><span class="mrel">∣</span><span class="mord mathit" style="margin-right:0.04398em;">z</span><span class="mclose">)</span><span class="mord mathit">p</span><span class="mopen">(</span><span class="mord mathit" style="margin-right:0.04398em;">z</span><span class="mclose">)</span></span></span></span><span class="baseline-fix"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span>​</span></span></span><span class="sizing reset-size5 size5 reset-textstyle textstyle uncramped nulldelimiter"></span></span></span></span></span></span></p><p>        p(x)不能直接求, 所以直接贝叶斯这个方法报废，于是我们寻找新的方法. 这时我们想到了变分法,用另一个分布 <span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>Q</mi><mo>(</mo><mi>z</mi><mo>∣</mo><mi>x</mi><mo>)</mo></mrow><annotation encoding="application/x-tex">Q(z \\mid x)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.75em;"></span><span class="strut bottom" style="height:1em;vertical-align:-0.25em;"></span><span class="base textstyle uncramped"><span class="mord mathit">Q</span><span class="mopen">(</span><span class="mord mathit" style="margin-right:0.04398em;">z</span><span class="mrel">∣</span><span class="mord mathit">x</span><span class="mclose">)</span></span></span></span> 来估计 <span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>p</mi><mo>(</mo><mi>z</mi><mo>∣</mo><mi>x</mi><mo separator="true">,</mo><mi>θ</mi><mo>)</mo></mrow><annotation encoding="application/x-tex">p(z \\mid x, \\theta)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.75em;"></span><span class="strut bottom" style="height:1em;vertical-align:-0.25em;"></span><span class="base textstyle uncramped"><span class="mord mathit">p</span><span class="mopen">(</span><span class="mord mathit" style="margin-right:0.04398em;">z</span><span class="mrel">∣</span><span class="mord mathit">x</span><span class="mpunct">,</span><span class="mord mathit" style="margin-right:0.02778em;">θ</span><span class="mclose">)</span></span></span></span> , 变分自编码器的变分就来源于此. <br><em>(注释：求泛函极值的方法称为变分法)</em> <br><em>(注释2：对于给定的值x∈[x0, x1]，两个可取函数y(x)和y0(x)，函数y(x)在y0(x)处的变分或函数的变分被定义为它们之差，即y(x) - y0(x)。这个变分表示了函数y(x)相对于y0(x)的变化或偏离程度。）</em> <br></p><p>        用一个函数去近似另一个函数，可以看作从概率密度函数所在的函数空间到实数域R的一个函数f，自变量是Q的密度函数，因变量是Q与真实后验密度函数的“距离”，而这一个f关于概率密度函数的“导数”就叫做 <strong>变分</strong> ，我们每次降低这个距离，让Q接近真实的后验，就是让概率密度函数朝着“导数“的负方向进行函数空间的梯度下降。所以叫做变分推断。<br></p><p>变分推断和变分自编码器的最终目标是相同的，都是将 <span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>Q</mi><mo>(</mo><mi>z</mi><mo>∣</mo><mi>x</mi><mo>)</mo></mrow><annotation encoding="application/x-tex">Q(z \\mid x)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.75em;"></span><span class="strut bottom" style="height:1em;vertical-align:-0.25em;"></span><span class="base textstyle uncramped"><span class="mord mathit">Q</span><span class="mopen">(</span><span class="mord mathit" style="margin-right:0.04398em;">z</span><span class="mrel">∣</span><span class="mord mathit">x</span><span class="mclose">)</span></span></span></span> 尽量去近似 <span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>p</mi><mo>(</mo><mi>z</mi><mo>∣</mo><mi>x</mi><mo separator="true">,</mo><mi>θ</mi><mo>)</mo></mrow><annotation encoding="application/x-tex">p(z \\mid x, \\theta)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.75em;"></span><span class="strut bottom" style="height:1em;vertical-align:-0.25em;"></span><span class="base textstyle uncramped"><span class="mord mathit">p</span><span class="mopen">(</span><span class="mord mathit" style="margin-right:0.04398em;">z</span><span class="mrel">∣</span><span class="mord mathit">x</span><span class="mpunct">,</span><span class="mord mathit" style="margin-right:0.02778em;">θ</span><span class="mclose">)</span></span></span></span> , 我们知道有一种距离可以量化两种分布的差异Kullback-Leibler divergence—KL散度，我们要尽量减小KL散度。<br></p><h2 id="" tabindex="-1"><a class="header-anchor" href="#" aria-label="Permalink to &quot;&quot;">​</a></h2><p>在这种情况下，我们可以让变分近似后验是一个具有对角协方差结构的多元高斯:</p><h1 id="_4-参考文献" tabindex="-1">4 参考文献 <a class="header-anchor" href="#_4-参考文献" aria-label="Permalink to &quot;4 参考文献&quot;">​</a></h1><ul><li><a href="https://www.cnblogs.com/lvzhiyi/p/15822716.html" target="_blank" rel="noreferrer">vae 导读</a></li><li><a href="https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73" target="_blank" rel="noreferrer">vae 导读2</a></li><li><a href="https://zhuanlan.zhihu.com/p/34998569" target="_blank" rel="noreferrer">vae 参考3</a></li></ul>',35)]))}const x=s(p,[["render",m]]);export{d as __pageData,x as default};
