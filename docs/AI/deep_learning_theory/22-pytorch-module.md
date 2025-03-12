# torch.nn.Module 
- [src code](https://github.com/pytorch/pytorch/tree/270111b7b611d174967ed204776985cefca9c144/torch/nn)

**实践：找到本机pytorch 库 中对应的torch.nn 的位置。** <br>

# 1 pytorch 自带的 torch.nn layer
- [pytorch user guide](https://pytorch.org/docs/stable/nn.html)

**思考：为什么要定义torch.nn 模块呢？？？** <br>

## 1.1 用 torch.nn 解决之前的问题
```python
def nn_demo():
    '''
    1. 数据准备：输入数据 + lable 数据
    2. 网络结构的搭建：激活函数 + 损失函数 + 权重初始化；
    3. 优化器选择；
    4. 训练策略：学习率的控制 + 梯度清0 + 更新权重 + 正则化；
    '''
    input = torch.tensor([5, 10]).reshape(1, 2).to(torch.float32)
    linear_1 = torch.nn.Linear(2, 3)
    act_1 = torch.nn.Sigmoid()
    linear_2 = torch.nn.Linear(3, 2)
    act_2 = torch.nn.Sigmoid()
    criteration = torch.nn.MSELoss()
    
    optimizer = torch.optim.SGD([{"params": linear_1.parameters()},
                                 {"params": linear_2.parameters()}], lr=0.5)
    label = torch.tensor([0.01, 0.99]).reshape(1, 2)
    
    for i in range(100):
        optimizer.zero_grad()
        x = linear_1(input)
        x = act_1(x)
        x = linear_2(x)
        output = act_2(x)
        loss = criteration(output, label)
        loss.backward()
        optimizer.step() # 更新权重      
        print(loss)
```

## 1.2 Tensor 和 Parameter 的区别
- 通过查阅代码解决；

**思考：上述做法还有优化空间吗？？？** <br>

# 2 定义我们自己的module
## 2.1 代码案例
- 代码案例1：<br>
``` python
class FullConnect(nn.Module):
    def __init__(self, k, n):
        super(FullConnect, self).__init__() # 初始化父类
        self.full_connect1 = nn.Linear(30, 20) 
        self.full_connect2 = nn.Linear(20, 10)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()
        
    def forward(self, input):
        x = self.full_connect1(input) # type(input) = Tensor
        x = self.activation1(x)
        x = self.full_connect2(x)
        x = self.activation2(x)
        
        return x
           
def full_connect_demo():
    model = FullConnect(30, 10) # model的实例化
    input = torch.rand(4, 30) # 我们拿不到input的梯度
    
    loss_function = nn.CrossEntropyLoss()
    lable = torch.Tensor([3, 4, 2, 4]).to(torch.int64)
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    for i in  range(100):
        model.train()
        optimizer.zero_grad()      
        output = model(input) # output: float
        loss = loss_function(output, lable)
        loss.backward()
        # print("================model weight grad before update: ", model.full_connect1.weight[0])
        optimizer.step()
        print("=======loss: ", loss)
        # print("================model weight grad before update: ", model.full_connect1.weight[0])
```

- 代码案例2：<br>
```python
class ModuleDemo(torch.nn.Module):
    def __init__(self):
        super(ModuleDemo, self).__init__()
        self.linear_1 = torch.nn.Linear(2, 3)
        self.act_1 = torch.nn.LeakyReLU()
        self.linear_2 = torch.nn.Linear(3, 2)
        self.act_2 = torch.nn.LeakyReLU()
        
    def forward(self, input):
        x = self.linear_1(input)
        x = self.act_1(x)
        x = self.linear_2(x)
        output = self.act_2(x)
        # loss = self.criteration(output, label)
        return output
    
def module_train():
    torch.manual_seed(0)
    input = torch.tensor([5, 10]).reshape(1, 2).to(torch.float32)
    label = torch.tensor([0.01, 0.99]).reshape(1, 2)
    model = ModuleDemo()
    criteration = torch.nn.MSELoss()    
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    
    for i in range(100):
        optimizer.zero_grad()
        output = model(input)
        loss = criteration(output, label)
        loss.backward()
        optimizer.step()
        print(f"=========loss[{i}]: {loss}")
```

## 2.2 customer layer 要点
1. 自定义的模型应该继承自基类 nn.Module；
2. custom 模块还可以包含其他模块，从而可以将它们嵌套在树形结构中, 子模块为当前模块的普通属性；
3. 上述子模块在 custom modle 中的__init__函数初始化；
4. 在子类上进行赋值之前，必须先调用父类的 __init__()；
5. 需要有 forward函数: 我们具体的实现过程，计算过程。

# 3 nn.Module 中的容器
```python
def sequential_demo():
    seq_modules = nn.Sequential(
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
        
    input_image = torch.rand(3,28*28)
    logits = seq_modules(input_image)
    print("logits: ", logits.shape)
```

# 4 nn.Module 属性详解
```python
# 是否生成模块的补丁列表
# 补丁列表记录了模块中所有可导操作的详细信息，包括输入张量、输出张量、参数、缓冲区等
dump_patches: bool = False # 
_version: int = 1 # 版本号，对应向后兼容很重要
training: bool # 是否处于训练模式
_parameters: Dict[str, Optional[Parameter]] # 模型的当前类的_parameters
_buffers: Dict[str, Optional[Tensor]] # 模型的 buffers，可通过register_buffers 来定义
# 存储模块中的非持久性缓冲区（non-persistent buffer）的集合，非持久性缓冲区不会被保存为模型的状态字典（state_dict）的缓冲区
_non_persistent_buffers_set: Set[str] # register_buffer 的 persistent = False 时触发
# 反向传播过程中，在梯度计算之前执行的钩子函数， register_full_backward_pre_hook 来注册
_backward_pre_hooks: Dict[int, Callable] 
# 反向传播过程中，在梯度计算之后执行的钩子函数，
_backward_hooks: Dict[int, Callable] # register_full_backward_hook 来注册, register_backward_hook 来注册被禁用
_is_full_backward_hook: Optional[bool]  # 判断反向传播钩子函数是否完全覆盖了模块的所有梯度输出
# 前向钩子函数，前向执行完后调用，用register_forward_hook来注册
_forward_hooks: Dict[int, Callable] # 
# 判断前向钩子函数是否带参数
_forward_hooks_with_kwargs: Dict[int, bool] 
# 前向钩子函数是否always 被调用
_forward_hooks_always_called: Dict[int, bool]
# 前向 pre 钩子函数，register_forward_pre_hook来注册
_forward_pre_hooks: Dict[int, Callable]
# pre 前向钩子函数是否带参数
_forward_pre_hooks_with_kwargs: Dict[int, bool]
# state_dict 函数内获取状态后执行, _register_state_dict_hook 来注册
_state_dict_hooks: Dict[int, Callable] 
# 加载状态的pre 钩子函数，_register_load_state_dict_pre_hook 来注册
_load_state_dict_pre_hooks: Dict[int, Callable] # 
# state_dict 函数内获取状态前执行，register_state_dict_pre_hook 来注册
_state_dict_pre_hooks: Dict[int, Callable] 
# 加载状态时的后向钩子函数， register_load_state_dict_post_hook 来注册
_load_state_dict_post_hooks: Dict[int, Callable]
# 存储子模块，子模块还可以有子模块
_modules: Dict[str, Optional['Module']] 
# False 时不能传参数，True时 可以传参数 给到父类
call_super_init: bool = False 
# torch.compile: Optimizes given model/function using TorchDynamo and specified backend.
# self._compiled_call_impl = torch.compile(self._call_impl, *args, **kwargs)
_compiled_call_impl : Optional[Callable] = None 
```
# 5 torch.nn.Module 常用功能
## 5.1  _parameters 设置机制
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;nn.Module 使用了 Python 的 __setattr__ 机制，当在类中定义成员时，__setattr__ 会检测成员的 type 派生于哪些类型。如果派生于 Parameter 类，则被归于 _parameters ；如果派生于 Module ，则划归于 _modules。因此，如果类中定义的成员被封装到Python的普通数据类型中，则不会自动归类，比如：self.layers = [nn.Linear(1024, 80), nn.Linear(80, 10]，检测到是list类型，则会视为普通属性。<br>

## 5.2 _buffers 功能展示
```python
import torch
import torch.nn as nn

'''通常，这用于注册一个不被视为模型参数的缓冲区。例如，BatchNorm 的 running_mean 不是一个参数，但它是模块的状态的一部分。
缓冲区默认是持久的，并将与参数一起保存。通过将 persistent 属性设置为 False，可以改变这种行为。
持久性缓冲区和非持久性缓冲区之间唯一的区别是后者不会成为该模块的 state_dict 的一部分。
可以通过给定的名称将缓冲区作为属性进行访问。
self.register_buffer('running_mean', torch.zeros(num_features))
'''

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('batch_mean', None)
        self.register_buffer('batch_var', None, persistent=False)
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))

    def forward(self, x):
        # 计算当前批次的均值和方差
        batch_mean = torch.mean(x)
        batch_var = torch.var(x)

        # 更新运行时的均值和方差
        if self.training:
            if self.batch_mean is None:
                self.batch_mean = batch_mean.detach()
                self.batch_var = batch_var.detach()
            else:
                self.batch_mean = (1 - 0.1) * self.batch_mean + 0.1 * batch_mean.detach()
                self.batch_var = (1 - 0.1) * self.batch_var + 0.1 * batch_var.detach()
            self.running_mean = (1 - 0.1) * self.running_mean + 0.1 * batch_mean.detach()
            self.running_var = (1 - 0.1) * self.running_var + 0.1 * batch_var.detach()

        # 使用运行时的均值和方差进行归一化
        normalized_x = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)

        return normalized_x

# 创建模型实例
module = MyModule()

# 模拟训练过程
for epoch in range(5):
    # 模拟每个批次的输入数据
    input_data = torch.randn(10)

    # 前向传播
    output = module(input_data)

    # 输出当前批次的均值和方差
    print(f"Batch Mean: {module.batch_mean.item()}, Batch Variance: {module.batch_var.item()}")

# 输出最终的运行时均值和方差
print(f"Running Mean: {module.running_mean.item()}, Running Variance: {module.running_var.item()}")
```

## 5.3 前向钩子函数展示
```python
import torch
    import torch.nn as nn

    def hook_fn(module, input, output):
        # 钩子函数的自定义操作
        print("Executing hook function...")
        print("Module:", module)
        print("Input:", input)
        print("Output:", output)
        print("-------------------")

    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
            
        def forward(self, input):
            return self.linear(input)

    module = MyModule()

    # 注册前向钩子函数
    handle = module.register_forward_hook(hook_fn)

    # 执行前向传播
    input_tensor = torch.randn(1, 10)
    output = module(input_tensor)

    # 移除钩子函数
    handle.remove()
```

## 5.4 反向钩子函数展示
```python
import torch
import torch.nn as nn

# 反向传播钩子函数
def backward_hook_fn(module, grad_input, grad_output):
    # 自定义操作
    print("Executing backward hook function...")
    print("Module:", module)
    print("Gradient Input:", grad_input)
    print("Gradient Output:", grad_output)
    print("-------------------")

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = MyModel()

# 注册反向传播钩子函数
hook_handle = model.register_backward_hook(backward_hook_fn)

# 定义输入数据和目标标签
input_data = torch.randn(1, 10)
target = torch.randn(1, 1)

# 前向传播
output = model(input_data)

# 计算损失
loss = output.sum()

# 反向传播
loss.backward()

# 移除钩子函数
hook_handle.remove()
```

## 6 nn.Module 方法全解
```python
# 构造函数 --> 一系列属性初始化
def __init__(self, *args, **kwargs) -> None:

# forward 函数
forward: Callable[..., Any] = _forward_unimplemented
# 添加一个buffer 到 module，注册一个不被视为模型参数的缓冲区，但可能是模型状态的一部分
def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
# Adds a parameter to the module
def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
# 向module 添加 子module，
# 提供了一种标准化的方式来将子模块添加到模型中。
# 通过使用 add_module，您可以确保模型的子模块被正确地注册和命名，并在模型的其他部分中进行访问。
def add_module(self, name: str, module: Optional['Module']) -> None:
# add_module 的别名
def register_module(self, name: str, module: Optional['Module']) -> None:
# 通过名称获取子模块
def get_submodule(self, target: str) -> "Module":
# 通过名称获取parameter
def get_parameter(self, target: str) -> "Parameter":
# 通过名称获取 buffer
def get_buffer(self, target: str) -> "Tensor":
# 获取额外状态，用户应该实现此函数，否则直接报错，额外的状态应该是可 picklable 的
def get_extra_state(self) -> Any:
# 设置额外状态，子类实现此函数，否则直接报错
def set_extra_state(self, state: Any):
# 内部使用，递归的将所有子模块的_parameters 和 _buffers 应用用户指定的函数 fn 进行操作
def _apply(self, fn, recurse=True):
# 外部使用，递归的将所有子模块应用 fn 进行处理
def apply(self: T, fn: Callable[['Module'], None]) -> T:
# _parameters 和 _buffers 转到cuda 设备上
def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:
# 将所有模型参数和缓冲区移动到 IPU（Intel Processor for AI）上。
def ipu(self: T, device: Optional[Union[int, device]] = None) -> T:
#将所有模型参数和缓冲区移动到 XPU 可扩展处理单元（extensible Processing Unit）上。
def xpu(self: T, device: Optional[Union[int, device]] = None) -> T:
# 将所有模型参数和缓冲区移动到 CPU 上。
def cpu(self: T) -> T:
# in-place 的 将所有参数和缓冲区转换为 dst_type 类型
def type(self: T, dst_type: Union[dtype, str]) -> T:
# inplace 的将所有参数和缓冲区转换为 float 类型
def float(self: T) -> T:
# inplace 的将所有参数和缓冲区转换为 double 类型
def double(self: T) -> T:
# inplace 的将所有参数和缓冲区转换为 half 类型
def half(self: T) -> T:
# inplace 的将所有参数和缓冲区转换为 bfloat16 类型
def bfloat16(self: T) -> T:
# 将参数和缓冲区移动到指定的设备，而无需复制存储空间。
def to_empty(self: T, *, device: Union[str, device], recurse: bool = True) -> T:
# 将参数和缓冲区移动到指定的设备
@overload
def to(self: T, device: Optional[Union[int, device]] = ..., dtype: Optional[Union[dtype, str]]= ..., non_blocking: bool = ...) -> T:
# 将参数和缓冲区转成指定的数据类型
@overload
def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T:
# 将参数和缓冲区转成Tensor 对应的数据类型 和 device类型
@overload
def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T:
# 调用以上三种to
def to(self, *args, **kwargs):
   
# 在模块上注册一个反向传播 pre 钩子。
def register_full_backward_pre_hook(
    self,
    hook: Callable[["Module", _grad_t], Union[None, _grad_t]],
    prepend: bool = False,
    ) -> RemovableHandle:
# 在模块上注册一个反向传播钩子。
# 这个函数已被弃用，建议改用 torch.nn.Module.register_full_backward_hook 方法。
# 将来版本中，该函数的行为将会发生变化。
def register_backward_hook(
    self, hook: Callable[['Module', _grad_t, _grad_t], Union[None, _grad_t]]
    ) -> RemovableHandle:

# 在模块上注册一个反向传播钩子。
def register_full_backward_hook(
    self,
    hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]],
    prepend: bool = False,
    ) -> RemovableHandle:

# 返回用于在调用函数中使用的反向传播钩子。
# 它返回两个列表，一个包含完整的反向传播钩子（full backward hooks），
# 另一个包含非完整的反向传播钩子（non-full backward hooks）。
def _get_backward_hooks(self):
# 返回反向传播pre 钩子
def _get_backward_pre_hooks(self):
# 用于在注册非完整反向传播钩子时发出警告：未来版本中可能会对非完整钩子的行为进行更改。
def _maybe_warn_non_full_backward_hook(self, inputs, result, grad_fn):

# 在模块上注册一个前向传播预钩子。该预钩子将在每次调用 forward 函数之前被调用。
def register_forward_pre_hook(
    self,
    hook: Union[
        Callable[[T, Tuple[Any, ...]], Optional[Any]],
        Callable[[T, Tuple[Any, ...], Dict[str, Any]], Optional[Tuple[Any, Dict[str, Any]]]],
    ],
    *,
    prepend: bool = False,
    with_kwargs: bool = False,
    ) -> RemovableHandle:

# 在模块上注册一个前向传播钩子。该钩子将在每次调用 forward 函数计算输出之后被调用。
def register_forward_hook(
    self,
    hook: Union[
        Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
        Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
    ],
    *,
    prepend: bool = False,
    with_kwargs: bool = False,
    always_call: bool = False,
    ) -> RemovableHandle:
# 在forward 时，同时进行 tracing，以捕获静态图，此时eager mode 运行速度会降低
def _slow_forward(self, *input, **kwargs):
# 对前向传播进行装饰：控制forward的调度，可能会处理一些其他的内部机制，如跟踪计算图、梯度传播等。
def _wrapped_call_impl(self, *args, **kwargs):
# 模型真正执行之处
def _call_impl(self, *args, **kwargs):
# __call__ 魔术方法，让Module 成为 Callable 对象
__call__ : Callable[..., Any] = _wrapped_call_impl
# 该方法应返回一个表示对象状态的可序列化对象，通常是一个字典。
# pickle.dump()、pickle.dumps() 时可被自动调用。
def __getstate__(self):
# 更新module的状态
def __setstate__(self, state):
# 在访问不存在的属性时被调用
def __getattr__(self, name: str) -> Any:
# 在module属性设置过程中插入自己的逻辑
def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
# 删除 属性
def __delattr__(self, name):
# 注册 state_dict 钩子函数：
# 这些钩子函数将被传入以下参数：self，state_dict，prefix，local_metadata；
# 在设置完self的state_dict之后调用。
# 请注意，只有self及其子模块的参数和缓存才能保证存在于state_dict中。
# 这些钩子函数可以原地修改state_dict，也可以返回一个新的state_dict。
def _register_state_dict_hook(self, hook):
# 注册 state_dict 的pre 钩子函数；
# 这些钩子函数将在调用self的state_dict之前，使用参数self、prefix和keep_vars进行调用。
# 注册的钩子函数可用于在进行state_dict调用之前执行预处理操作。
def register_state_dict_pre_hook(self, hook):
# 被state_dict 方法内部调用；
# 将模块的状态保存到包含模块状态但不包含其子模块状态的destination字典中。
# 在torch.nn.Module.state_dict中的每个子模块上调用此方法。
# 在极少数情况下，子类可以通过覆盖此方法并添加自定义逻辑来实现特定于类的行为。
def _save_to_state_dict(self, destination, prefix, keep_vars):

# 用户可以选择性地传递一个可映射对象给state_dict，在这种情况下，state_dict将返回相同的对象。
# 但如果未传递任何对象，将创建并返回一个OrderedDict。
T_destination = TypeVar('T_destination', bound=Dict[str, Any])

# 状态字典的重载
@overload
def state_dict(self, *, destination: T_destination, prefix: str = ..., keep_vars: bool = ...) -> T_destination:
@overload
def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Any]:
# 返回一个包含模块整个状态的字典。
# 包括参数和持久性缓冲区（例如运行平均值）。键是相应的参数和缓冲区名称。
# 设置为None的参数和缓冲区不包括在内。
# 返回的对象是一个浅拷贝。它包含对模块参数和缓冲区的引用。
def state_dict(self, *args, destination=None, prefix='', keep_vars=False):

# 注册 state_dict 的前置钩子函数，将执行 state_dict 前 被调用
def _register_load_state_dict_pre_hook(self, hook, with_module=False):
# 注册一个后置钩子函数，在调用模块的load_state_dict方法之后运行。
def register_load_state_dict_post_hook(self, hook):

# 被load_state_dict 调用，
# 将参数和缓冲区从state_dict复制到当前模块，但不包括其子模块。
# 这个方法在torch.nn.Module.load_state_dict中的每个子模块上调用。
# 提供给这个模块的state_dict中保存的元数据作为local_metadata参数。
# 对于没有元数据的状态字典，local_metadata为空。子类可以使用local_metadata.get("version", None)中的版本号实现特定于类的向后兼容加载。
# 此外，local_metadata还可以包含assign_to_params_buffers键，指示是否应该将键分配给其在state_dict中对应的张量。
def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

# 将参数（_parameters）和缓冲区(_buffers)从state_dict复制到当前模块及其子模块。
# 如果strict为True，则state_dict的键必须与该模块的torch.nn.Module.state_dict函数返回的键完全匹配。
def load_state_dict(self, state_dict: Mapping[str, Any],  strict: bool = True, assign: bool = False):
# 用于生成模块的各种名称和成员的辅助方法。
def _named_members(self, get_members_fn, prefix='', recurse=True, remove_duplicate: bool = True):
# 返回一个迭代器，用于遍历模块的参数。通常将其传递给优化器(optimizer)使用。
def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
# 返回一个迭代器，遍历模块的参数，同时产出参数的名称和参数本身。
def named_parameters(
        self,
        prefix: str = '',
        recurse: bool = True,
        remove_duplicate: bool = True
        ) -> Iterator[Tuple[str, Parameter]]:
# 返回一个迭代器，用以遍历模块的buffers
def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
# 返回一个迭代器，遍历模块的buffers，同时产出buffers的名称和参数本身。
def named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, Tensor]]:
# 返回一个迭代器，用于遍历直接子模块。
def children(self) -> Iterator['Module']:
# 返回一个迭代器，用于遍历直接子模块，同时产出模块的名称和模块本身
def named_children(self) -> Iterator[Tuple[str, 'Module']]:
# 返回一个迭代器，用于遍历网络中的所有模块。
def modules(self) -> Iterator['Module']:
# 返回一个迭代器，用于遍历网络中的所有模块，同时产出模块的名称和模块本身。
def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True):
# 将module 设置为 training 模式
def train(self: T, mode: bool = True) -> T:
# 将module 设置为 eval 模式
def eval(self: T) -> T:
# 更改是否自动梯度过程中应记录此module中参数；
# 该方法会in-place 地设置参数的requires_grad属性。
# 该方法对于冻结模块的一部分以进行微调或单独训练模型的部分（例如，GAN训练）非常有用。
# 有关requires_grad_()方法与几种类似机制之间的比较，请参阅:ref:locally-disable-grad-doc。
def requires_grad_(self: T, requires_grad: bool = True) -> T:
# 重置所有模型参数的梯度，可以清为0，也可以直接置为None
def zero_grad(self, set_to_none: bool = True) -> None:
def share_memory(self: T) -> T:# 用于将张量的内存共享给其他进程，cuda默认为True
def _get_name(self): # 返回类实例的名称
# 用于返回模块的额外表示形式（extra representation）。该方法可以用来提供有关模块的附加信息，通常用于调试和打印模块的可读表示。
def extra_repr(self) -> str: 

def __repr__(self): # 定义对象的字符串表示形式。使用 repr() 函数或在交互式环境中直接输出对象时被调用
def __dir__(self): # 返回该对象的属性和方法的名称列表
# 用于创建模块的副本，以便在每个设备上进行并行计算。副本本身不具有参数，副本引用原始module。
def _replicate_for_data_parallel(self):
# 使用torch.compile编译该模块的前向传播。
# 该模块的__call__方法会被编译，并且所有的参数会原样传递给torch.compile。
def compile(self, *args, **kwargs):
```

- [locally-disable-grad-doc](https://pytorch.org/docs/2.1/notes/autograd.html#locally-disable-grad-doc)
