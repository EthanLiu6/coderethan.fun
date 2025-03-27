# 0 torch.optim
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; torch.optim 是一个实现各种优化算法的包。最常用的方法已经得到支持，并且接口足够通用，以便未来可以轻松地集成更复杂的方法。<br>
- [torch.optim guide link](https://pytorch.org/docs/stable/optim.html)

# 1 如何使用torch.optim 
## 1.1 创建一个优化器对象
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;要使用 torch.optim，您需要构建一个优化器对象，该对象将**保存当前状态**并根据计算得到的梯度更新参数。<br>

**思考：当前状态 指的是什么 ？？？** <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;要构建一个优化器（Optimizer），您需要提供一个可迭代的对象，其中包含要优化的参数（所有参数应该是 Parameter 类型）。然后，您可以指定优化器特定的选项，例如学习率（learning rate）、权重衰减（weight decay）等。<br>

- 代码展示 <br>
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
```

## 1.2 逐参数选项(Per-parameter options)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;优化器还支持指定每个参数的选项。为了实现这一点，不要传递一个 Parameter 对象的可迭代对象，而是传递一个 dict 对象的可迭代对象。每个 dict 对象将定义一个单独的参数组，并且应包含一个 **"params" key**，其中包含属于该组的参数列表。**其他 key**应与优化器接受的关键字参数匹配，并将用作该组的优化选项。<br>

- 注意(Note) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;您仍然可以将选项作为关键字参数传递。它们将被用作默认值，在未覆盖它们的组中生效。当您只想改变单个选项，同时保持其他参数组之间一致时，这非常有用。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;例如，当需要指定每个层的学习率时，这非常有用：<br>
```python
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这意味着 model.base 的参数将使用默认学习率 1e-2，model.classifier 的参数将使用学习率 1e-3，并且所有参数都将使用动量 0.9。<br>

## 1.3 进行优化步骤
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;所有的优化器都实现了 step() 方法，用于更新参数。它可以通过两种方式使用：<br>

- 方式1：<br>
**optimizer.step()** <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这是大多数优化器支持的简化版本。可以在计算梯度（例如通过 backward()）之后调用该函数，具体代码为：<br>

```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

- 方式2：<br>
**optimizer.step(closure)** <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一些优化算法，如共轭梯度（Conjugate Gradient）和LBFGS，需要多次重新评估函数，因此您需要传递一个闭包（closure），使它们能够重新计算您的模型。闭包应该清除梯度、计算损失并返回它。<br>

```python
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)
```

# 2 torch.optim base class introduce
- [CLASStorch.optim.Optimizer(params, defaults)](https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer) <br>

## 2.1 torch.optim.Optimizer 的输入参数
- params(可迭代对象) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一个包含 torch.Tensor 或 dict 的可迭代对象。指定应该进行优化的张量。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**注意** <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;参数需要以具有确定性顺序且在运行之间保持一致的集合形式进行指定。不满足这些属性的对象的示例包括集合（sets）和对字典值进行迭代的迭代器（iterators）。<br>

- defaults(字典) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一个包含优化选项默认值的字典（在参数组未指定时使用）。

## 2.2 torch.optim.Optimizer 属性
```python
self.defaults = defaults # 存储优化器的全局超参数
self._optimizer_step_pre_hooks = OrderedDict() # 存储 用于 step()方法 中的前置钩子函数
self._optimizer_step_post_hooks = OrderedDict() # 存储 用于step()方法 中的后置钩子函数
self._optimizer_state_dict_pre_hooks = OrderedDict() # 存储 用于 state_dict()方法中的 前置钩子函数
self._optimizer_state_dict_post_hooks = OrderedDict() # 存储 用于 state_dict()方法 中的 后置钩子函数
self._optimizer_load_state_dict_pre_hooks = OrderedDict() # 存储 用于load_state_dict() 方法中的前置钩子函数
self._optimizer_load_state_dict_post_hooks = OrderedDict() # 存储 用于load_state_dict() 方法中的后置钩子函数

self.state: DefaultDict[torch.Tensor, Any] = defaultdict(dict) # 存储 优化器的状态
self.param_groups: List[Dict[str, Any]] = [] # 存储不同group 的 Parameter参数 和 其它超参数

param_groups = list(params) # 输入参数转为 list
if len(param_groups) == 0: # params 不能为空
    raise ValueError("optimizer got an empty parameter list")
if not isinstance(param_groups[0], dict): # 不是字典类型的话，说明是 params 组成的可迭代对象
    param_groups = [{'params': param_groups}]  # 将 params的可迭代对象转换为 字典形式

for param_group in param_groups:
    self.add_param_group(cast(dict, param_group)) # 将param_groups 中每个group 添加到 self.param_groups 中
```

**思考：上述属性那几个属性最为关键 ？？？** <br>

## 2.3 torch.optim.Optimizer 方法
```python
# 构造函数：主要完成self.defaults self.state self.param_groups 的初始化
def __init__(self, params: params_t, defaults: Dict[str, Any]) -> None:

# 获取 optimize 的状态，序列化时会触发
def __getstate__(self) -> Dict[str, Any]:

# 反序列化时触发，调用 pickle.load(file) 或 pickle.loads(bytes) 时，加载保存的模型状态
def __setstate__(self, state: Dict[str, Any]) -> None:

# 当我们在交互式环境中输出对象或使用 repr() 函数时，会自动调用对象的 __repr__ 方法来获取其字符串表示。
def __repr__(self) -> str:

# 用于pytorch2.0 torch.compile 时的加速，避免一些耗时的检查
def _cuda_graph_capture_health_check(self) -> None:

# torch.profile.profiler的入口点，被 profile_hook_step 方法调用
def _optimizer_step_code(self) -> None:

# 静态方法：返回一个装饰器 wrapper，用于插入 钩子函数 和 _optimizer_step_code 函数， 被 _patch_step_function 调用
@staticmethod
def profile_hook_step(func: Callable[_P, R]) -> Callable[_P, R]:

# 按设备和数据类型对张量的列表进行分组。
# 如果我们正在编译，则跳过此步骤，因为这将在inductor lowering 阶段发生。
@staticmethod
def _group_tensors_by_device_and_dtype(
    tensorlistlist: TensorListList,
    with_indices: bool = False,
) -> Union[
    Dict[Tuple[None, None], Tuple[TensorListList, Indices]],
    Dict[Tuple[torch.device, torch.dtype], Tuple[TensorListList, Indices]],
]:

# 通过wrapper的方式给step方法加补丁(patch)
def _patch_step_function(self) -> None:

# 注册一个优化器 step 前 的前置钩子函数，该钩子将在优化器步骤之前调用。它应该具有以下签名：
# hook(optimizer, args, kwargs) -> None or modified args and kwargs
def register_step_pre_hook(self, hook: OptimizerPreHook) -> RemovableHandle:

# 注册一个优化器 step 操作后的 后置钩子函数，该钩子将在优化器步骤之后调用。它应该具有以下签名：
# hook(optimizer, args, kwargs) -> None or modified args and kwargs
def register_step_post_hook(self, hook: OptimizerPostHook) -> RemovableHandle:

# 注册一个状态字典前置钩子，该钩子将在调用torch.optim.Optimizer.state_dict之前被调用。它应该具有以下签名：
# hook(optimizer) -> None
# optimizer参数是正在使用的优化器实例。在调用self上的state_dict之前，将使用参数self调用该钩子。注册的钩子可用于在进行state_dict调用之前执行预处理操作。
def register_state_dict_pre_hook(
    self, hook: Callable[["Optimizer"], None], prepend: bool = False
) -> RemovableHandle:

# 注册一个状态字典后置钩子，该钩子将在调用:torch.optim.Optimizer.state_dict之后被调用。它应该具有以下签名：
# hook(optimizer, state_dict) -> state_dict or None
# 在self上生成state_dict之后，将使用参数self和state_dict调用该钩子。
# 该钩子可以就地修改state_dict，或者可选择返回一个新的state_dict。注册的钩子可用于在返回state_dict之前对其进行后处理。
def register_state_dict_post_hook(
    self,
    hook: Callable[["Optimizer", StateDict], Optional[StateDict]],
    prepend: bool = False,
) -> RemovableHandle:

# 以class:dict的形式返回优化器的状态。
# 它包含两个条目：
# 1. state：保存当前优化状态的字典。不同的优化器类之间其内容可能不同，但有一些共同的特征。
# 例如，状态是按参数保存的，而参数本身不会被保存。state是一个将参数ID映射到相应参数状态字典的字典。
# 2. param_groups：包含所有参数组的列表，其中每个参数组是一个字典。
# 每个参数组包含特定于优化器的元数据，如学习率和权重衰减，以及参数组中参数的参数ID列表。
# 参数ID可能看起来像索引，但它们只是将状态与参数组关联起来的ID
@torch._disable_dynamo
def state_dict(self) -> StateDict:

# 根据param 规则进行 Tensor value 的转换
@staticmethod
def _process_value_according_to_param_policy(
    param: torch.Tensor,
    value: torch.Tensor,
    param_id: int,
    param_groups: List[Dict[Any, Any]],
    key: Hashable = None,
) -> torch.Tensor:

# 注册一个 load_state_dict 前置钩子，在调用torch.optim.Optimizer.load_state_dict 之前将会被调用。它应该具有以下的签名：
# hook(optimizer, state_dict) -> state_dict or None
# optimizer 参数是正在使用的优化器实例，state_dict 参数是用户传递给 load_state_dict 的 state_dict 的浅拷贝。
# 预钩子可以就地修改 state_dict，或者可选择返回一个新的 state_dict。如果返回了一个 state_dict，它将被用于加载到优化器中。
# 在调用 self 上的 load_state_dict 之前，钩子将以 self 和 state_dict 作为参数被调用。
# 注册的钩子可以用于在进行 load_state_dict 调用之前执行预处理操作。
def register_load_state_dict_pre_hook(
    self,
    hook: Callable[["Optimizer", StateDict], Optional[StateDict]],
    prepend: bool = False,
) -> RemovableHandle:

# 注册一个 load_state_dict 后钩子，在调用torch.optim.Optimizer.load_state_dict 之后将会被调用。它应该具有以下的签名：
# hook(optimizer) -> None
# optimizer 参数是正在使用的优化器实例。
# 在调用 self 上的 load_state_dict 之后，钩子将以 self 作为参数被调用。注册的钩子可以用于在 load_state_dict 加载完 state_dict 后执行后处理操作。
def register_load_state_dict_post_hook(
    self, hook: Callable[["Optimizer"], None], prepend: bool = False
) -> RemovableHandle:

# 加载存储的优化器状态
@torch._disable_dynamo
def load_state_dict(self, state_dict: StateDict) -> None:

# 重置所有被优化的 torch.Tensor 的梯度。
@torch._disable_dynamo
def zero_grad(self, set_to_none: bool = True) -> None:

@overload
def step(self, closure: None = ...) -> None:

@overload
def step(self, closure: Callable[[], float]) -> float:

# 执行一次优化步骤（参数更新）。
def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:

# class:Optimizer 的 param_groups 添加一个参数组
@torch._disable_dynamo
def add_param_group(self, param_group: Dict[str, Any]) -> None:
```

# 3 不同实现与性能优化
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们的许多算法都有各种不同的实现方式，针对性能、可读性和/或通用性进行了优化，因此如果用户没有指定特定的实现方式，我们会尝试默认选择当前设备上通常最快的实现方式。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们有三个主要的实现类别：for 循环、foreach（多张量）和 fused。最直接的实现方式是对参数进行 for 循环，并进行大块的计算。相比于 for 循环，foreach 实现通常更快，它将参数组合成一个多张量，并一次性执行大块的计算，从而节省了许多连续的内核调用。我们的一些优化器甚至具有更快的fused实现方式，将大块的计算融合成一个内核。我们可以将 foreach 实现看作是在水平方向上进行融合，fused 实现则在此基础上在垂直方向上进行融合。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通常情况下，这三种实现方式的性能排序为：fused -> foreach -> for 循环。因此，在适用的情况下，我们会默认选择 foreach 而不是 for 循环。适用的条件是 foreach 实现可用，并且用户没有指定任何与实现相关的 kwargs（例如，fused、foreach、differentiable），且所有张量都是本地的并在 CUDA 上。请注意，虽然 fused 实现比 foreach 还要快，但这些实现是较新的，我们希望在完全切换之前给它们更多的时间来适应。不过，您可以随时尝试它们！<br>




