# 1 tensorboard 介绍
[torch 链接](https://pytorch.org/docs/stable/tensorboard.html?highlight=tensorboard)
- board：展板
- tensorflow 率先采用个
- 效果很好，pytorch 也采用了这个 --> 
- 只要我们把我们需要保存的信息 dump 成tensorboard支持的格式就行；
- pytorch 里面还有一个叫 tensorboardX 的东西，和 tensorboard 很类似，我们用tensorboard就行

# 2 安装方式
- 我们安装好了 tensorflow 的话，tensorboard会自动安装；
- pip install tensorboard


# 3 抓取log

## 3.1 import SummaryWriter
```python
import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
```

## 3.2 plot scalar
```python
def add_scalar():
  writer = SummaryWriter("scalar_log")
  for n_iter in range(200, 300):
      # writer.add_scalars('Loss/train', {"a":n_iter * 2, "b": n_iter*n_iter}, n_iter)
      writer.add_scalar('Loss/test1', 200, n_iter)
      # writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
      # writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
```

## 3.3 plot loss and accuracy
```python

writer = SummaryWriter("run")

# Log the running loss averaged per batch
writer.add_scalars('Training vs. Validation Loss',
                { 'Training' : avg_train_loss, 'Validation' : avg_val_loss },
                epoch * len(training_loader) + i)

```

# 4 执行方式：
tensorboard --logdir=./log  <br>
tensorboard --logdir dir_name   <br>
python -m tensorboard.main --logdir=./logs   <br>

# 5 查看graph
```python
def add_graph():
  import torchvision.models as models
  net = models.resnet50(pretrained=False)
  writer = SummaryWriter("graph_log")
  writer.add_graph(net, torch.rand(16, 3, 224, 224))
  writer.flush()
  writer.close()
```

# 6 查看特征图
```python
def add_image():
  # Writer will output to ./runs/ directory by default
  # --logdir=./runs
  writer = SummaryWriter("mtn_log")

  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
  trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
  model = torchvision.models.resnet50(False)
  torch.onnx.export(model, torch.randn(64, 3, 224, 224), "resnet50_ttt.onnx")
  # Have ResNet model take in grayscale rather than RGB
  model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
  images, labels = next(iter(trainloader)) # 拿到 输入 和label
  
  print("============images shape: ", images.shape)
  output = model.conv1(images)
  output = output[:, 0, :, :].reshape(64, 1, 14, 14).expand(64, 3, 14, 14)
  print("============output shape: ", output.shape)
  
  
  grid = torchvision.utils.make_grid(images)
  grid = torchvision.utils.make_grid(output)
  writer.add_image('output', grid, 0) # 保存图片
  # writer.add_graph(model, images) # 保存模型
  writer.close()
```

# 7 性能分析profiler
```python
# Non-default profiler schedule allows user to turn profiler on and off
# on different iterations of the training loop;
# trace_handler is called every time a new trace becomes available
def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],

    # In this example with wait=1, warmup=1, active=2, repeat=1,
    # profiler will skip the first step/iteration,
    # start warming up on the second, record
    # the third and the forth iterations,
    # after which the trace will become available
    # and on_trace_ready (when set) is called;
    # the cycle repeats starting with the next step

    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2,
        repeat=1),
    on_trace_ready=trace_handler
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    # used when outputting for tensorboard
    ) as p:
        for iter in range(N):
            code_iteration_to_profile(iter)
            # send a signal to the profiler that the next iteration has started
            p.step()
```
