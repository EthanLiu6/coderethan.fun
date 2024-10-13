import{_ as e,c as r,a2 as t,o}from"./chunks/framework.C9NVOr0y.js";const p=JSON.parse('{"title":"【操作系统】进程同步","description":"","frontmatter":{"title":"【操作系统】进程同步","date":"2024-10-12T00:00:00.000Z"},"headers":[],"relativePath":"IT-learning/408知识/操作系统/【操作系统】进程同步.md","filePath":"IT-learning/408知识/操作系统/【操作系统】进程同步.md","lastUpdated":1728792724000}'),i={name:"IT-learning/408知识/操作系统/【操作系统】进程同步.md"};function l(n,a,h,s,d,c){return o(),r("div",null,a[0]||(a[0]=[t('<h1 id="【操作系统】进程同步" tabindex="-1">【操作系统】进程同步 <a class="header-anchor" href="#【操作系统】进程同步" aria-label="Permalink to &quot;【操作系统】进程同步&quot;">​</a></h1><details class="details custom-block"><summary>本篇内容课前定位（下拉展开）</summary><p>首先还是看这张图，对我们当前正在学习的地方做一个定位： <img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241012172750676.png" alt="image-20241012172750676"></p><p>上一篇笔记我们已经讲了进程通信相关的知识，这篇笔记我们讲讲进程协作之间的进程同步。</p><p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241012235407890.png" alt="image-20241012235407890"></p></details><h2 id="_1-基本概念" tabindex="-1">1. 基本概念 <a class="header-anchor" href="#_1-基本概念" aria-label="Permalink to &quot;1. 基本概念&quot;">​</a></h2><h3 id="_1-1为什么要提出" tabindex="-1">1.1为什么要提出？ <a class="header-anchor" href="#_1-1为什么要提出" aria-label="Permalink to &quot;1.1为什么要提出？&quot;">​</a></h3><p>进程同步是为了<strong>协调多个进程或线程对共享资源的访问</strong>，以确保数据一致性、避免冲突，并保证程序按预期执行。在操作系统中，多个进程或线程可能需要同时访问共享资源（例如内存、文件、硬件设备等），如果没有合适的同步机制，可能会出现一些不可预期的问题，如数据竞争、死锁（后面详讲）、资源浪费等。</p><h3 id="_1-2-同步是什么" tabindex="-1">1.2 同步是什么？ <a class="header-anchor" href="#_1-2-同步是什么" aria-label="Permalink to &quot;1.2 同步是什么？&quot;">​</a></h3><blockquote><p><strong>进程</strong>我们前面已经提及过了，这里不多赘述，直接理解同步</p></blockquote><p>同步分为<strong>进程同步</strong>和<strong>资源同步</strong>，进程同步我们下面会详细讲述，这里简单提一下。进程同步指多个进程在特定点会合（join up）或者<a href="https://zh.wikipedia.org/wiki/%E6%8F%A1%E6%89%8B_(%E6%8A%80%E6%9C%AF)" target="_blank" rel="noreferrer">握手</a>使得达成协议或者使得操作序列有序。数据同步指一个数据集的多份拷贝一致以维护<a href="https://zh.wikipedia.org/wiki/%E5%AE%8C%E6%95%B4%E6%80%A7" target="_blank" rel="noreferrer">完整性</a>。常用进程同步原语实现数据同步。[1]</p><h3 id="_1-3-什么又是互斥" tabindex="-1">1.3 什么又是互斥？ <a class="header-anchor" href="#_1-3-什么又是互斥" aria-label="Permalink to &quot;1.3 什么又是互斥？&quot;">​</a></h3><p>两个或两个以上的进程，不能同时进入关于同一组共享变量的临界区域，否则可能发生与时间有关的错误，这种现象被称作进程互斥。 也就是说，一个进程正在访问<a href="https://baike.baidu.com/item/%E4%B8%B4%E7%95%8C%E8%B5%84%E6%BA%90/1880269?fromModule=lemma_inlink" target="_blank" rel="noreferrer">临界资源</a>，另一个要访问该资源的进程必须等待。</p><h3 id="_1-4-临界资源是啥" tabindex="-1">1.4 临界资源是啥？ <a class="header-anchor" href="#_1-4-临界资源是啥" aria-label="Permalink to &quot;1.4 临界资源是啥？&quot;">​</a></h3><h4 id="_1-4-1-系统资源" tabindex="-1">1.4.1 系统资源 <a class="header-anchor" href="#_1-4-1-系统资源" aria-label="Permalink to &quot;1.4.1 系统资源&quot;">​</a></h4><p>前面一直说系统资源，或许有同学会疑问啥是系统资源。在<a href="https://zh.wikipedia.org/wiki/%E8%A8%88%E7%AE%97%E6%A9%9F%E7%A7%91%E5%AD%B8" target="_blank" rel="noreferrer">计算机科学</a>中，<strong>系统资源</strong>（英语：system resource），或是<strong>资源</strong>（英语：resource），意指是一个电脑系统中，限制其运算能力的任何实体或是虚拟的组成元件。任何连结到电脑系统中的装置，都是一个资源，例如键盘、萤幕等。电脑系统内部的任何元件都是资源，如CPU，RAM。电脑系统中的软件虚拟元件，包括档案，网络连线与记忆体区块等，都是一种资源。[2]</p><h4 id="_1-4-2-临界资源-共享资源" tabindex="-1">1.4.2 临界资源（共享资源） <a class="header-anchor" href="#_1-4-2-临界资源-共享资源" aria-label="Permalink to &quot;1.4.2 临界资源（共享资源）&quot;">​</a></h4><p><strong>临界资源</strong>（Critical Resource）是指在<strong>并发环境</strong>中，多个进程或线程<strong>需要共享访问</strong>，但同一时刻<strong>只能被一个进程或线程使用</strong>的资源。也就是一种特殊的共享资源。[3]</p><p>在<a href="https://baike.baidu.com/item/%E5%A4%9A%E9%81%93%E7%A8%8B%E5%BA%8F/8192392?fromModule=lemma_inlink" target="_blank" rel="noreferrer">多道程序</a>环境下，存在着临界资源，它是指<a href="https://baike.baidu.com/item/%E5%A4%9A%E8%BF%9B%E7%A8%8B/9796976?fromModule=lemma_inlink" target="_blank" rel="noreferrer">多进程</a>存在时必须互斥访问的资源。也就是某一时刻不允许多个进程同时访问，只能单个进程的访问。我们把这些程序的片段称作<a href="https://baike.baidu.com/item/%E4%B8%B4%E7%95%8C%E5%8C%BA/8942134?fromModule=lemma_inlink" target="_blank" rel="noreferrer">临界区</a>或临界段，它存在的目的是有效的防止<a href="https://baike.baidu.com/item/%E7%AB%9E%E4%BA%89%E6%9D%A1%E4%BB%B6/10354815?fromModule=lemma_inlink" target="_blank" rel="noreferrer">竞争条件</a>又能保证最大化使用共享数据。而这些并发进程必须有好的解决方案，才能防止出现以下情况：多个进程同时处于临界区，临界区外的进程阻塞其他的进程，有些进程在临界区外无休止的等待。除此以外，这些方案还不能对CPU的速度和数目做出任何的假设。只有满足了这些条件，才是一个好的解决方案。[4]</p><h4 id="_1-4-3-临界区" tabindex="-1">1.4.3 临界区 <a class="header-anchor" href="#_1-4-3-临界区" aria-label="Permalink to &quot;1.4.3 临界区&quot;">​</a></h4><p><strong>临界区</strong>指的是一个访问共用资源（例如：共用设备或是共用存储器）的程序片段，而这些共用资源又无法同时被多个<a href="https://baike.baidu.com/item/%E7%BA%BF%E7%A8%8B/103101?fromModule=lemma_inlink" target="_blank" rel="noreferrer">线程</a>访问的特性。当有线程进入临界区段时，其他线程或是<a href="https://baike.baidu.com/item/%E8%BF%9B%E7%A8%8B/382503?fromModule=lemma_inlink" target="_blank" rel="noreferrer">进程</a>必须等待（例如：bounded waiting 等待法），有一些同步的机制必须在临界区段的进入点与离开点实现，以确保这些共用资源是被互斥获得使用，例如：<a href="https://baike.baidu.com/item/semaphore/1322231?fromModule=lemma_inlink" target="_blank" rel="noreferrer">semaphore</a>。只能被单一线程访问的设备，例如：<a href="https://baike.baidu.com/item/%E6%89%93%E5%8D%B0%E6%9C%BA/215563?fromModule=lemma_inlink" target="_blank" rel="noreferrer">打印机</a>。[5]</p><h2 id="_2-同步如何实现" tabindex="-1">2. 同步如何实现 <a class="header-anchor" href="#_2-同步如何实现" aria-label="Permalink to &quot;2. 同步如何实现&quot;">​</a></h2><div class="tip custom-block"><p class="custom-block-title">TIP</p><p>实现同步也就是实现临界区进程（或线程）之间的互斥访问</p></div><h3 id="_2-1-访问原则" tabindex="-1">2.1 访问原则 <a class="header-anchor" href="#_2-1-访问原则" aria-label="Permalink to &quot;2.1 访问原则&quot;">​</a></h3><p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241013000945237.png" alt="image-20241013000945237"></p><h3 id="_2-2-软件实现-后续补充" tabindex="-1">2.2 软件实现（后续补充） <a class="header-anchor" href="#_2-2-软件实现-后续补充" aria-label="Permalink to &quot;2.2 软件实现（后续补充）&quot;">​</a></h3><h4 id="_2-2-1-单标志法" tabindex="-1">2.2.1 单标志法 <a class="header-anchor" href="#_2-2-1-单标志法" aria-label="Permalink to &quot;2.2.1 单标志法&quot;">​</a></h4><h4 id="_2-2-2-双标志先检查法" tabindex="-1">2.2.2 双标志先检查法 <a class="header-anchor" href="#_2-2-2-双标志先检查法" aria-label="Permalink to &quot;2.2.2 双标志先检查法&quot;">​</a></h4><h4 id="_2-2-3-双标志后检查法" tabindex="-1">2.2.3 双标志后检查法 <a class="header-anchor" href="#_2-2-3-双标志后检查法" aria-label="Permalink to &quot;2.2.3 双标志后检查法&quot;">​</a></h4><h3 id="_2-3-硬件实现-后续补充" tabindex="-1">2.3 硬件实现（后续补充） <a class="header-anchor" href="#_2-3-硬件实现-后续补充" aria-label="Permalink to &quot;2.3 硬件实现（后续补充）&quot;">​</a></h3><h4 id="_2-3-1-中断屏蔽方法" tabindex="-1">2.3.1 中断屏蔽方法 <a class="header-anchor" href="#_2-3-1-中断屏蔽方法" aria-label="Permalink to &quot;2.3.1 中断屏蔽方法&quot;">​</a></h4><h4 id="_2-3-2-test-and-set-ts指令-tsl指令" tabindex="-1">2.3.2 Test-And-Set（TS指令/TSL指令） <a class="header-anchor" href="#_2-3-2-test-and-set-ts指令-tsl指令" aria-label="Permalink to &quot;2.3.2 Test-And-Set（TS指令/TSL指令）&quot;">​</a></h4><h4 id="_2-3-3-swap指令-exchange-xchg指令" tabindex="-1">2.3.3 Swap指令（EXCHANGE，XCHG指令） <a class="header-anchor" href="#_2-3-3-swap指令-exchange-xchg指令" aria-label="Permalink to &quot;2.3.3 Swap指令（EXCHANGE，XCHG指令）&quot;">​</a></h4><h4 id="_2-3-4-信号量机制-重点-下一节详细讲解" tabindex="-1">2.3.4 信号量机制（重点，下一节详细讲解） <a class="header-anchor" href="#_2-3-4-信号量机制-重点-下一节详细讲解" aria-label="Permalink to &quot;2.3.4 信号量机制（重点，下一节详细讲解）&quot;">​</a></h4><h2 id="_3-参考资料" tabindex="-1">3. 参考资料 <a class="header-anchor" href="#_3-参考资料" aria-label="Permalink to &quot;3. 参考资料&quot;">​</a></h2><p>[1]. 维基百科：<a href="https://zh.wikipedia.org/wiki/%E5%90%8C%E6%AD%A5_(%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%A7%91%E5%AD%A6)" target="_blank" rel="noreferrer">同步(计算机科学)</a></p><p>[2]. 维基百科：<a href="https://zh.wikipedia.org/wiki/%E8%B3%87%E6%BA%90_(%E8%A8%88%E7%AE%97%E6%A9%9F%E7%A7%91%E5%AD%B8)" target="_blank" rel="noreferrer">资源(计算机科学)</a></p><p>[3]. ChatGPT：<a href="https://chatgpt.com/" target="_blank" rel="noreferrer">临界资源</a></p><p>[4]. 百度百科：<a href="https://baike.baidu.com/item/%E8%BF%9B%E7%A8%8B%E4%BA%92%E6%96%A5/5096533" target="_blank" rel="noreferrer">进程互斥</a></p><p>[5]. 百度百科：<a href="https://baike.baidu.com/item/%E4%B8%B4%E7%95%8C%E5%8C%BA/8942134" target="_blank" rel="noreferrer">临界区</a></p>',37)]))}const b=e(i,[["render",l]]);export{p as __pageData,b as default};
