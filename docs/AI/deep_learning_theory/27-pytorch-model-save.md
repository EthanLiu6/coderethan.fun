# 1 tensor 的保存和加载
```python
def tensor_save():
    tensor = torch.ones(5, 5)
    torch.save(tensor, "tensor.t")
    tensor_new = torch.load("tensor.t")
    print(tensor_new)
    
def load_to_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('mnist.pt', map_location=device)
    print(f"model device: {model}")
```

# 2 模型状态的保存
## 2.1 定义一个模型
```python
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

## 2.2 保存模型的状态
```python
def save_para_demo():
    model = Net()
    torch.save(model.state_dict(), "mnist_para.pth")
```

## 2.3 加载模型的状态
```python
def load_para_demo():
    param = torch.load("mnist_para.pth")
    model = Net()
    model.load_state_dict(param)
    input = torch.rand(1, 1, 28, 28)
    output = model(input)   
    print(f"output shape: {output.shape}")
```

## 2.4 思考与尝试
- 不同的后缀名，对模型的保存和加载有影响吗？？？
- 保存模型状态后，可以直接给别人使用吗？？？ （前提：他人本地没有这个模型）
- 模型状态中都保存了什么内容呢？？？

# 3 保存与加载模型
## 3.1 保存模型
```
def save_demo_v1():
    model = Net()
    input = torch.rand(1, 1, 28, 28)
    output = model(input)
    torch.save(model, "mnist.pt") # 4.6M : 保存
```
## 3.2 加载模型
```python
def load_demo_v1():
    model = torch.load("mnist.pt")

    device = torch.device("cuda")
    model = torch.load('mnist.pt', map_location=device) # 直接将模型加载到device上
    
    input = torch.rand(1, 1, 28, 28)
    output = model(input)
    print(f"output shape: {output.shape}")
```

## 3.3 思考与尝试
- 此时保存后的模型可以直接给他人使用吗？？？ （前提：他人本地没有这个模型）
- 此时保存的模型中都包含哪些内容呢？？？
    

# 4 训练中的保存和加载

**思考： 训练过程中断，我们需要保存哪些内容以在之前基础上继续训练，而无需重新开始整个训练过程 ？？？**

## 4.1 保存训练中的状态
```python
def save_ckpt_demo():
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = torch.Tensor([0.25])
    epoch = 10
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': loss.item(),
        # 可以添加其他训练信息
    }

    torch.save(checkpoint, 'mnist.ckpt')
```

## 4.2 加载训练中的状态
```python
def load_ckpt_demo():
    checkpoint = torch.load('model.ckpt')
    model = Net() # 需要事先定义一个net的实例
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    input = torch.rand(1, 1, 28, 28)
    output = model(input)
    print("output shape: ", output.shape)
```

# 5 保存和加载模型的静态图

**思考：如何将模型安全的交付？？？** <br>

## 5.1 保存模型静态图
```python
def save_trace_model():
    model = Net().eval()
    # 通过trace 得到了一个新的model，我们最终保存的是这个新的model
    traced_model = torch.jit.trace(model, torch.randn(1, 1, 28, 28))
    traced_model.save("traced_model.pt")
    # torch.save(traced_model, "mnist_trace.pt")
```

## 5.2 加载模型静态图
```python
def load_trace_model():
    mm = torch.jit.load("traced_model.pt")
    output = mm(torch.randn(1, 1, 28, 28))
    print("load model succsessfully !")
    print("output: ", output)
```

# 6 通用格式onnx的保存
## 6.1 保存onnx 静态图模型
```python
import torch
import torchvision

model = torchvision.models.resnet18()
input_tensor = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, input_tensor, "model.onnx")
```

## 6.2 运行onnx 模型
- 方式1：<br>
```python
def run(model_path, image_path):
  session = onnxruntime.InferenceSession(model_path)
  input_data = []
  input_data.append(image_process(image_path))
  input_name_1 = session.get_inputs()[0].name
  outputs = session.run([],{input_name_1:input_data})
  return outputs
```

- 方式2：<br>
```
def onnx_model_infer(input_data_list : list, onnx_model):
  session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
  input_name = [item.name for item in session.get_inputs()]
  assert(len(input_data_list) == len(input_name))
  input_dict = dict()
  for i, data in enumerate(input_data_list):
    input_dict[input_name[i]] = data

  outputs = session.run([], input_feed = input_dict)
  print("onnx_model_infer run successfully !!!")
  return outputs
```

## 6.3 shape infer
```python
def shape_infer(onnx_model):
  model = onnx.shape_inference.infer_shapes(onnx_model)
  print("shape_infer fun run successfully !!!")
  return model
```

