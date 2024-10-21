import{_ as t,c as e,a2 as o,o as r}from"./chunks/framework.C9NVOr0y.js";const u=JSON.parse('{"title":"人的本性——贪心！！！","description":"","frontmatter":{},"headers":[],"relativePath":"IT-learning/生活与算法/贪心算法/1.人的本性——贪心！.md","filePath":"IT-learning/生活与算法/贪心算法/1.人的本性——贪心！.md","lastUpdated":null}'),l={name:"IT-learning/生活与算法/贪心算法/1.人的本性——贪心！.md"};function i(n,a,s,h,d,c){return r(),e("div",null,a[0]||(a[0]=[o(`<h1 id="人的本性——贪心" tabindex="-1">人的本性——贪心！！！ <a class="header-anchor" href="#人的本性——贪心" aria-label="Permalink to &quot;人的本性——贪心！！！&quot;">​</a></h1><hr><h2 id="title-人的本性——贪心-data-2024-10-21" tabindex="-1">title: 人的本性——贪心！ data: 2024-10-21 <a class="header-anchor" href="#title-人的本性——贪心-data-2024-10-21" aria-label="Permalink to &quot;title: 人的本性——贪心！
data: 2024-10-21&quot;">​</a></h2><blockquote><p>“人生处处是算法，贪心更是人的一种本性”</p><p>——Ethan.Liu</p></blockquote><div class="tip custom-block"><p class="custom-block-title">贪心算法大纲内容（下拉展示）</p><img src="https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20241021171552525.png" style="zoom:70%;"></div><h2 id="_1-贪心与人生" tabindex="-1">1. 贪心与人生 <a class="header-anchor" href="#_1-贪心与人生" aria-label="Permalink to &quot;1. 贪心与人生&quot;">​</a></h2><p>一说到“贪心”二字，我想大家想到的并不是什么算法，而是一种人的本性，想想日常生活，我们在购物的时候，往往追求“性价比”，这点女生可能比男生更有经验，哈哈哈，我们绘不自觉地优先选择价格最低或折扣最大的商品，在预算内购买到更多的商品，而且还要让这些商品的价值尽可能的大。是不是很符合人的本性，这里讲的”贪心算法“与其本质没有太大区别，所以大家不用太怕，只是现实的量化而已😄。</p><h2 id="_2-贪心算法基本概念" tabindex="-1">2. 贪心算法基本概念 <a class="header-anchor" href="#_2-贪心算法基本概念" aria-label="Permalink to &quot;2. 贪心算法基本概念&quot;">​</a></h2><h3 id="_2-1-概念" tabindex="-1">2.1 概念 <a class="header-anchor" href="#_2-1-概念" aria-label="Permalink to &quot;2.1 概念&quot;">​</a></h3><p>贪心算法是一种常用的<strong>算法设计策略</strong>（并没有具体模板），旨在通过逐步做出<strong>局部最优</strong>选择来寻找<strong>全局最优</strong>解。</p><h3 id="_2-2-通俗讲解" tabindex="-1">2.2 通俗讲解 <a class="header-anchor" href="#_2-2-通俗讲解" aria-label="Permalink to &quot;2.2 通俗讲解&quot;">​</a></h3><p><strong>算法设计策略</strong>：没有具体的模板，根据不同情况，采取对应的策略和方法。</p><p><strong>局部最优</strong>：想象你在一座山上爬山，途中你看到一个小山峰。在你所处的位置，这个小山峰是最高的（也就是局部最优），但如果你继续往上爬，你可能会发现更高的山峰（有可能是当前的局部最优，因为往上爬可能还有更高的）。</p><p><strong>全局最优</strong>：你最终到达了这座山的最高点，这个点就是全局最优解。无论你从哪个方向来看，它都是最高的，没有其他点可以更高。</p><h2 id="_3-贪心算法原理" tabindex="-1">3. 贪心算法原理 <a class="header-anchor" href="#_3-贪心算法原理" aria-label="Permalink to &quot;3. 贪心算法原理&quot;">​</a></h2><h3 id="_3-1-我想说" tabindex="-1">3.1 我想说 <a class="header-anchor" href="#_3-1-我想说" aria-label="Permalink to &quot;3.1 我想说&quot;">​</a></h3><p>其实贪心算法不是一种具体模板化的算法，更像是一种随机应变、想当然的、理所应当的算法，这里这样说不太严谨，但我觉得我们追求严谨并非是件好事（各种证明和推导一定是必要的吗？）。可能大家不太理解我具体在说什么，大家想了解的话可以浏览器或者AI查一查贪心算法的严谨证明。这里会有一些<strong>贪心策略的选择</strong>，后续会提到。所以，大家对下面的这些原理和策略的证明可以暂时跳过，有兴趣的可以自己去研究（特别是数学相关的），随着自己的能力的提升后续尝试自己证明。</p><h3 id="_3-2-贪心策略" tabindex="-1">3.2 贪心策略 <a class="header-anchor" href="#_3-2-贪心策略" aria-label="Permalink to &quot;3.2 贪心策略&quot;">​</a></h3><p>我们简单理解一下这个东西，比如你要规划任务（任务的开始结束时间都不同），你是要选择时间最短的最先安排，还是最花费时间的最先安排，又或者说把结束最早的最先安排。可能举例举的不好，大家找找感觉能理解什么是**“策略”**就行，其实就是不同的实现方式。</p><h3 id="_3-3-基本原理" tabindex="-1">3.3 基本原理 <a class="header-anchor" href="#_3-3-基本原理" aria-label="Permalink to &quot;3.3 基本原理&quot;">​</a></h3><blockquote><p>感兴趣的小伙伴可以自己查阅学习</p></blockquote><ul><li>将大问题划分成若干小问题</li><li>考虑局部“最优”（选择合适的贪心策略）</li><li>证明全局最优 <ul><li>贪心选择性质：局部最优解在全局最优解内</li><li>最优子结构性质：子问题的最优解可以组合成原问题的最优解，即 整体最优包含了子问题最优。</li></ul></li></ul><h2 id="_4-总结" tabindex="-1">4. 总结 <a class="header-anchor" href="#_4-总结" aria-label="Permalink to &quot;4. 总结&quot;">​</a></h2><p>回归到开始，我们只需对“贪心”有一定的初步认识，大概知道是个什么样的思想就行，因为人的本性就是贪心的，我们解决问题的时候会觉得这是本该如此的。至于具体原理和证明，以及如何实现，后面大家慢慢会有更深刻的认识。</p>`,24)]))}const _=t(l,[["render",i]]);export{u as __pageData,_ as default};
