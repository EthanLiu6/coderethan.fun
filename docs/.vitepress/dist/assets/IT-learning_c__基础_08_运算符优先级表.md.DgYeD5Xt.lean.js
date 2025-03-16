import{_ as d,c as e,o,a2 as c}from"./chunks/framework.DA-Pb-tg.js";const h=JSON.parse('{"title":"","description":"","frontmatter":{},"headers":[],"relativePath":"IT-learning/c++基础/08_运算符优先级表.md","filePath":"IT-learning/c++基础/08_运算符优先级表.md","lastUpdated":null}'),r={name:"IT-learning/c++基础/08_运算符优先级表.md"};function a(l,t,n,_,i,s){return o(),e("div",null,t[0]||(t[0]=[c('<h2 id="运算符优先级表" tabindex="-1">运算符优先级表 <a class="header-anchor" href="#运算符优先级表" aria-label="Permalink to &quot;运算符优先级表&quot;">​</a></h2><table tabindex="0"><thead><tr><th>优先级</th><th>运算符</th><th>描述</th><th>结合性</th></tr></thead><tbody><tr><td>1</td><td><code>[]</code></td><td>数组下标</td><td>从左到右</td></tr><tr><td></td><td><code>()</code></td><td>函数调用或圆括号</td><td></td></tr><tr><td></td><td><code>++</code>, <code>--</code></td><td>后缀递增和递减</td><td></td></tr><tr><td></td><td><code>-&gt;</code></td><td>通过指针访问成员</td><td></td></tr><tr><td></td><td><code>.</code></td><td>结构体和联合体成员访问</td><td></td></tr><tr><td>2</td><td><code>++</code>, <code>--</code></td><td>前缀递增和递减</td><td>从右到左</td></tr><tr><td></td><td><code>+</code>, <code>–</code></td><td>一元加，一元减</td><td></td></tr><tr><td></td><td><code>(type)</code></td><td>类型转换运算符</td><td></td></tr><tr><td></td><td><code>!</code>, <code>~</code></td><td>逻辑非和按位取反</td><td></td></tr><tr><td></td><td><code>*</code></td><td>解引用运算符</td><td></td></tr><tr><td></td><td><code>&amp;</code></td><td>取地址运算符</td><td></td></tr><tr><td></td><td><code>sizeof</code></td><td>获取字节大小</td><td></td></tr><tr><td></td><td><code>_Alignof</code></td><td>对齐要求</td><td></td></tr><tr><td>3</td><td><code>*</code>, <code>/</code>, <code>%</code></td><td>乘法，除法和取模</td><td>从左到右</td></tr><tr><td>4</td><td><code>+</code>, <code>–</code></td><td>加法和减法</td><td>从左到右</td></tr><tr><td>5</td><td><code>&lt;&lt;</code>, <code>&gt;&gt;</code></td><td>位左移和位右移</td><td>从左到右</td></tr><tr><td>6</td><td><code>&lt;</code>, <code>&lt;=</code></td><td>关系运算符 小于 和 小于等于</td><td>从左到右</td></tr><tr><td></td><td><code>&gt;</code>, <code>&gt;=</code></td><td>关系运算符 大于 和 大于等于</td><td></td></tr><tr><td>7</td><td><code>==</code>, <code>!=</code></td><td>关系运算符 等于 和 不等于</td><td>从左到右</td></tr><tr><td>8</td><td><code>&amp;</code></td><td>位与</td><td>从左到右</td></tr><tr><td>9</td><td><code>^</code></td><td>位异或 (XOR)</td><td>从左到右</td></tr><tr><td>10</td><td>`</td><td>`</td><td>位或 (包含 OR)</td></tr><tr><td>11</td><td><code>&amp;&amp;</code></td><td>逻辑与</td><td>从左到右</td></tr><tr><td>12</td><td>`</td><td></td><td>`</td></tr><tr><td>13</td><td><code>?:</code></td><td>三元条件运算符</td><td>从右到左</td></tr><tr><td>14</td><td><code>=</code></td><td>赋值</td><td>从右到左</td></tr><tr><td></td><td><code>+=</code>, <code>-=</code></td><td>增强的加法和减法</td><td></td></tr><tr><td></td><td><code>*=</code>, <code>/=</code></td><td>增强的乘法和除法</td><td></td></tr><tr><td></td><td><code>%=</code>, <code>&amp;=</code></td><td>增强的取模和位与</td><td></td></tr><tr><td></td><td><code>^=</code>, `</td><td>=`</td><td>增强的位异或和位或</td></tr><tr><td></td><td><code>&lt;&lt;=</code>, <code>&gt;&gt;=</code></td><td>增强的位左移和位右移</td><td></td></tr><tr><td>15</td><td><code>,</code></td><td>逗号（表达式分隔符）</td><td>从左到右</td></tr></tbody></table>',2)]))}const m=d(r,[["render",a]]);export{h as __pageData,m as default};
