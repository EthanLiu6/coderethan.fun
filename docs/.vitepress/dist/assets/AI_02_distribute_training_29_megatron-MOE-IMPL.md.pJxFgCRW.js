import{_ as t,c as e,o as i,a2 as r}from"./chunks/framework.DA-Pb-tg.js";const c=JSON.parse('{"title":"1 MOE 概述","description":"","frontmatter":{},"headers":[],"relativePath":"AI/02_distribute_training/29_megatron-MOE-IMPL.md","filePath":"AI/02_distribute_training/29_megatron-MOE-IMPL.md","lastUpdated":1743069065000}'),a={name:"AI/02_distribute_training/29_megatron-MOE-IMPL.md"};function n(o,l,p,s,u,d){return i(),e("div",null,l[0]||(l[0]=[r('<h1 id="_1-moe-概述" tabindex="-1">1 MOE 概述 <a class="header-anchor" href="#_1-moe-概述" aria-label="Permalink to &quot;1 MOE 概述&quot;">​</a></h1><p>        MoE（混合专家）是在Megatron-Core框架中实现的一种强大的大型语言模型（LLM）架构，旨在提高大型语言模型的效率和可扩展性。它利用专家并行性，允许多个专家分布在不同的工作节点上，<strong>每个工作节点处理不同的训练样本批次</strong>。这种方法显著提高了计算吞吐量，使模型能够实现高性能指标，例如在H100上使用BF16训练8个70亿参数模型时达到47%的MFU（模型实际使用的浮点运算能力占硬件平台理论最大计算能力的比例）。<br></p><p><strong>MoE的关键特性：</strong> <br></p><ul><li><p>并行性技术：MoE结合了多种并行策略，包括专家并行性、数据并行性、张量并行性、序列并行性、管道并行性和上下文并行性。这种组合使得能够有效处理更大的模型变体。</p></li><li><p>路由和负载均衡：系统采用先进的路由机制，如Top-K路由器，并利用负载均衡算法来优化专家之间的令牌（token）分配。</p></li><li><p>性能优化：诸如GroupedGEMM和FP8训练等技术提高了MoE模型的效率，特别是在涉及多个专家时。</p></li><li><p>Token分发机制：MoE支持无丢弃和令牌丢弃两种策略，以有效管理专家之间的令牌分配。</p></li></ul><h1 id="_2-megatron-core-moe-key-features" tabindex="-1">2 Megatron Core MoE Key Features <a class="header-anchor" href="#_2-megatron-core-moe-key-features" aria-label="Permalink to &quot;2 Megatron Core MoE Key Features&quot;">​</a></h1><ul><li><a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md" target="_blank" rel="noreferrer">Megatron-MoE</a></li></ul><h2 id="_2-1-与其他并行模式结合性" tabindex="-1">2.1 与其他并行模式结合性 <a class="header-anchor" href="#_2-1-与其他并行模式结合性" aria-label="Permalink to &quot;2.1 与其他并行模式结合性&quot;">​</a></h2><p>        Megatron-Core 提供了丰富的并行映射，将专家并行与张量并行、数据并行、序列并行和管道并行相结合。这使得 Mixtral 8X7B bf16 训练在 MCore v0.9 版本下能够达到 468 TFLOPS 的性能。<br></p><p><strong>并行性</strong> <br></p><ul><li><p>专家并行（Expert Parallelism）: <br>         一种针对混合专家（MoE）模型的特定并行方法，其中<strong>专家被划分到不同的工作节点上，每个工作节点处理不同的训练样本批次，<strong>每个工作节点为每个 MoE 层处理</strong>一个或多个专家</strong>。<br></p></li><li><p>3D 并行性：数据并行（Data Parallelism）、张量并行（Tensor Parallelism）、管道并行（Pipeline Parallelism）<br>         注：当在使用 MoE 的同时启用专家并行和张量并行时，<strong>必须启用序列并行</strong>。<br></p></li><li><p>上下文并行（Context Parallelism）<br>         将序列维度进行划分，以支持长上下文训练。<br></p></li><li><p>更丰富的并行映射：专家并行可以与数据并行/张量并行/管道并行/上下文并行相结合，以处理更大的 MoE 变体。<br></p></li><li><p>全面的分布式优化器支持。<br></p></li></ul><h2 id="_2-2-路由器与负载均衡" tabindex="-1">2.2 路由器与负载均衡 <a class="header-anchor" href="#_2-2-路由器与负载均衡" aria-label="Permalink to &quot;2.2 路由器与负载均衡&quot;">​</a></h2><ul><li><p>路由器类型：<br></p></li><li><ul><li>Top-K 多层感知器（MLP）路由器 <br></li></ul></li><li><p>负载均衡算法：<br></p></li><li><ul><li>Sinkhorn（S-BASE）<br></li></ul></li><li><ul><li>辅助损失/负载均衡损失 <br></li></ul></li><li><ul><li>无辅助损失的负载均衡策略 <br></li></ul></li><li><p>性能优化</p></li><li><ul><li>当本地专家数量大于1时使用GroupedGEMM</li></ul></li><li><ul><li>支持的数据类型：bf16</li></ul></li><li><ul><li>针对更大规模混合专家（MoE）模型的性能提升</li></ul></li><li><ul><li>为MoE启用--tp-comm-overlap</li></ul></li><li><ul><li>支持FP8训练</li></ul></li><li><p>Token分发机制</p></li><li><ul><li>无丢弃/无令牌丢弃</li></ul></li><li><ul><li>令牌丢弃，可选择是否填充至容量。</li></ul></li><li><p>易用性</p></li><li><ul><li>Mixtral模型的检查点转换器，详见示例。</li></ul></li><li><ul><li>分布式检查点存储</li></ul></li><li><ul><li>逐层日志记录</li></ul></li><li><ul><li>升级支持</li></ul></li><li><ul><li>细粒度升级</li></ul></li><li><p>即将推出的功能</p></li><li><ul><li>大规模混合专家（MoE）训练的新型并行机制</li></ul></li><li><ul><li>GroupedGEMM支持FP8格式</li></ul></li><li><ul><li>Token permutation/Unpermutation 融合</li></ul></li><li><ul><li>TopK路由器融合</li></ul></li><li><ul><li>MoE层频率</li></ul></li></ul><h1 id="_3-performance-best-practice" tabindex="-1">3 Performance Best Practice <a class="header-anchor" href="#_3-performance-best-practice" aria-label="Permalink to &quot;3 Performance Best Practice&quot;">​</a></h1><h2 id="_3-1-parallel-mapping" tabindex="-1">3.1 Parallel Mapping <a class="header-anchor" href="#_3-1-parallel-mapping" aria-label="Permalink to &quot;3.1 Parallel Mapping&quot;">​</a></h2><p>        为了找到一个良好的并行映射方法，以帮助你实现新模型的高吞吐量，有一些通用规则可以帮到你。以下是每种并行策略在不同方面的特性概述。<br></p><table tabindex="0"><thead><tr><th style="text-align:center;">Parallel Strategy</th><th style="text-align:center;">Peak Activation Memory</th><th style="text-align:center;">Weight Memory</th><th style="text-align:center;">Optimizer states</th><th style="text-align:center;">Communication (Per-Layer)</th></tr></thead><tbody><tr><td style="text-align:center;">TP</td><td style="text-align:center;">1/N (with SP on)</td><td style="text-align:center;">1/N</td><td style="text-align:center;">1/N</td><td style="text-align:center;">High</td></tr><tr><td style="text-align:center;">EP</td><td style="text-align:center;">1</td><td style="text-align:center;">1/N in MoELayer</td><td style="text-align:center;">1/N</td><td style="text-align:center;">Medium</td></tr><tr><td style="text-align:center;">PP</td><td style="text-align:center;">1 (&gt;1 with virtual pipeline)</td><td style="text-align:center;">1/N</td><td style="text-align:center;">1/N</td><td style="text-align:center;">Medium</td></tr><tr><td style="text-align:center;">CP</td><td style="text-align:center;">1/N</td><td style="text-align:center;">1</td><td style="text-align:center;">1/N (with distributed optimizer)</td><td style="text-align:center;">Medium</td></tr><tr><td style="text-align:center;">DP</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td><td style="text-align:center;">1/N (with distributed optimizer)</td><td style="text-align:center;">Low</td></tr></tbody></table><ol><li>尽量减小模型并行化规模。</li></ol><ul><li>对于大型语言模型，通常需要采用模型并行化来防止内存溢出（OOM），但这会带来通信开销并影响性能。<br></li><li>使用分布式优化器时，主权重和优化器状态将在所有数据并行（DP）节点间进行分片，且通信开销较小。因此，在训练过程中如果GPU内存充足，应尽量减小模型并行化规模，并增大数据并行化规模。<br></li></ul><ol start="2"><li>确保专家并行（EP）和张量并行（TP）的通信在NVLink域内进行。</li></ol><ul><li>EP和TP的通信应尽量保持在NVLink域内，因为这两者都是<strong>通信密集型操作</strong>。</li><li>如果模型过大，需要跨多个节点进行扩展，首先考虑在TP和EP之前使用管道并行（PP）。详见第3点。</li></ul><ol start="3"><li>使用管道并行来进一步扩展模型规模。</li></ol><ul><li>当PP规模（PP_size）大于等于2时，启用虚拟管道并行（VPP）来减少PP气泡，通过设置每个虚拟管道阶段的层数（num_layers_per_virtual_pipeline_stage）来实现。</li><li>VPP规模调优：vpp_size的合法值是num_layers/pp_size的所有公约数。例如，若num_layers=24，pp_size=4，则vpp_size可选{1, 2, 3, 6}。vpp_size越大，管道气泡越小，但每个PP阶段之间的点对点（P2P）通信次数越多。经验上，选择一个中间值往往能取得最佳平衡。vpp_size=num_layers / pp_size / num_layers_per_virtual_pipeline_stage。</li></ul><ol start="4"><li>在可能的情况下，专家层优先选择专家并行（EP）而非张量并行（TP）：</li></ol><ul><li>TP比EP节省更多内存，但EP能实现更高的GEMM效率和更低的通信开销。</li><li>如果EP规模增加到与专家数量相同，则可以省略专家计算中的本地token permutation/un-permutation.</li><li>简化混合专家（MoE）层的计算图，便于实现潜在的通信-计算重叠。</li><li>在实际应用中，对于8x7B模型，EP8TP1优于EP4TP2。</li></ul><ol start="5"><li>对于长上下文训练，启用上下文并行（CP）。</li></ol><ul><li>CP的效率很大程度上取决于其通信是否能与计算重叠。</li><li>经验上，当序列长度大于等于8K时，使用CP。</li></ul><h2 id="_3-2-moe-并行折叠" tabindex="-1">3.2 MoE 并行折叠 <a class="header-anchor" href="#_3-2-moe-并行折叠" aria-label="Permalink to &quot;3.2 MoE 并行折叠&quot;">​</a></h2><p>MoE 并行折叠将 MoE（混合专家）相关的并行组与密集（Dense）组分离。</p><p>传统的 MoE 并行组通过使用具有默认顺序（tp-cp-ep-dp-pp）的5维并行组生成器与密集组交织在一起。在 MoE 中，<strong>EP（专家并行）组是注意力（Attention）中 DP（数据并行）的一个子组</strong>。</p><p>通过 MoE 并行折叠，我们为注意力使用了一个具有 tp-cp-dp-pp 顺序的并行组生成器，而为 MoE 使用了另一个具有 tp-ep-dp-pp 顺序的并行组生成器。在 MoE 中，EPxTP 组是注意力中 <strong>DPxCPxTP 的一个子组</strong>。</p><p>通过设置 --expert-tensor-parallel-size，我们可以为 MoE 设置特定的 TP（张量并行）规模。<br></p><h2 id="_3-3-moe-并行折叠的优势" tabindex="-1">3.3 MoE 并行折叠的优势 <a class="header-anchor" href="#_3-3-moe-并行折叠的优势" aria-label="Permalink to &quot;3.3 MoE 并行折叠的优势&quot;">​</a></h2><ol><li>默认情况下，CP（上下文并行）和 EP（专家并行）组被折叠在一起，这样：<br></li></ol><ul><li>它减少了启用 CP 和 EP 所需的最小 GPU 数量。例如，传统方式下（CP=8，EP=8）至少需要 64 个 GPU，而现在只需 8 个 GPU。<br></li><li>CP 和 EP 的通信都可以放在 NVLink 域内进行。<br></li></ul><ol start="2"><li>我们可以为注意力（Attention）部分和 MoE（混合专家）部分设置不同的 TP（张量并行）规模。<br> 对于 MoE，EP 通常比 TP 更高效。但在传统方式下，仅使用 EP 可能会导致大多数模型出现内存溢出（OOM）。<br> 通过 MoE 并行折叠，我们可以为注意力部分启用 TP，并为 MoE 模型设置 TP=1，这通常能获得更好的 MFU（可能是指某种性能或利用率指标）。<br></li></ol>',35)]))}const g=t(a,[["render",n]]);export{c as __pageData,g as default};
