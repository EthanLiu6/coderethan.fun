1. offload.py
   ```python
   import torch
   import torch.nn  as nn
   
   # 自定义OffloadFunction类
   class OffloadFunction(torch.autograd.Function):
       @staticmethod
       def forward(ctx, input):
           # 保存输入到CPU，并保留原始输入用于计算图
           ctx.save_for_backward(input.detach().cpu())
           return input  # 保持前向传播在GPU执行
   
       @staticmethod
       def backward(ctx, grad_output):
           # 从CPU加载输入数据到GPU并启用梯度
           input_cpu, = ctx.saved_tensors
           input_gpu = input_cpu.cuda().requires_grad_()
   
           # 重新执行前向计算以构建局部计算图
           with torch.enable_grad():
               # 此处模拟实际模型的前向计算（例如Decoder层）
               output = input_gpu * 2  # 示例计算（替换为实际模型操作）
               output.backward(grad_output)
   
           return input_gpu.grad   # 返回梯度到上一层
   
   # 定义包含Offload的Decoder模型
   class OffloadDecoder(nn.Module):
       def __init__(self):
           super().__init__()
           self.layer1  = nn.Linear(512, 512)
           self.layer2  = nn.Linear(512, 512)
   
       def forward(self, x):
           # 第一层正常计算
           x = self.layer1(x)
   
           # 对第二层应用OffloadFunction
           x = OffloadFunction.apply(x)
   
           # 继续后续计算（示例）
           x = self.layer2(x)
           return x
   
   # 训练流程
   def train():
       model = OffloadDecoder().cuda()
       optimizer = torch.optim.Adam(model.parameters())
       criterion = nn.MSELoss()
   
       # 模拟数据
       inputs = torch.randn(32,  512).cuda()
       targets = torch.randn(32,  512).cuda()
   
       # 训练循环
       for epoch in range(10):
           optimizer.zero_grad()
           outputs = model(inputs)
           loss = criterion(outputs, targets)
           loss.backward()
           optimizer.step()
           print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
   
   if __name__ == "__main__":
       train()
   ```

   

2. optimizer_offload.py

   ```python
   import torch
   import torch.nn as nn
   from torch.utils.data import Dataset, DataLoader
   from torch.nn.parallel import DistributedDataParallel as DDP
   from torch.cuda.amp import autocast, GradScaler
   import torch.nn.functional as F
   
   # 配置参数
   class Config:
       vocab_size = 10000
       d_model = 512
       nhead = 8
       num_layers = 6
       dim_feedforward = 2048
       max_seq_len = 512
       batch_size = 64
       epochs = 10
       lr = 1e-4
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # Transformer Decoder模型
   class TransformerDecoderModel(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.embedding = nn.Embedding(config.vocab_size, config.d_model)
           self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
   
           decoder_layer = nn.TransformerDecoderLayer(
               d_model=config.d_model,
               nhead=config.nhead,
               dim_feedforward=dim_feedforward,
               batch_first=True
           )
           self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
           self.fc = nn.Linear(config.d_model, config.vocab_size)
   
       def forward(self, x, memory):
           seq_len = x.size(1)
           pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
           x = self.embedding(x) + self.pos_embed(pos)
   
           tgt_mask = nn.Transformer().generate_square_subsequent_mask(seq_len).to(x.device)
           x = self.decoder(x, memory, tgt_mask=tgt_mask)
           return self.fc(x)
   
   # 虚拟数据集
   class DummyDataset(Dataset):
       def __init__(self, size=1000):
           self.data = torch.randint(0, Config.vocab_size, (size, Config.max_seq_len))
   
       def __len__(self):
           return len(self.data)
   
       def __getitem__(self, idx):
           return self.data[idx], self.data[idx]
   
   # 异步数据预取器
   class DataPrefetcher:
       def __init__(self, loader):
           self.loader = loader
           self.stream = torch.cuda.Stream()
           self.preload()
   
       def preload(self):
           try:
               self.next_batch = next(self.iter)
           except StopIteration:
               self.next_batch = None
               return
   
           with torch.cuda.stream(self.stream):
               self.next_batch = [t.to(Config.device, non_blocking=True)
                                  for t in self.next_batch]
   
       def __iter__(self):
           self.iter = iter(self.loader)
           self.preload()
           return self
   
       def __next__(self):
           torch.cuda.current_stream().wait_stream(self.stream)
           batch = self.next_batch
           if batch is None:
               raise StopIteration
           self.preload()
           return batch
   
   # 训练函数
   def train():
       # 初始化
       config = Config()
       torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速
   
       # 数据加载
       train_set = DummyDataset()
       loader = DataLoader(train_set,
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True,
                          persistent_workers=True)
   
       # 模型初始化
       model = TransformerDecoderModel(config)
       model = model.to(config.device)
   
       # 使用CPU Offload（需要PyTorch >= 1.10）
       # if torch.cuda.is_available():
       #     model = DDP(model, device_ids=[config.device])
   
       # 混合精度训练
       # scaler = GradScaler()
       optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
   
       # 训练循环
       for epoch in range(config.epochs):
           prefetcher = DataPrefetcher(loader)
           batch_idx = 0
   
           for src, tgt in prefetcher:
               optimizer.zero_grad(set_to_none=True)  # 更节省内存
   
               # with autocast():
               memory = torch.randn(src.size(0), config.max_seq_len, config.d_model).to(config.device)
               output = model(src, memory)
               loss = F.cross_entropy(output.view(-1, config.vocab_size),
                                       tgt.view(-1),
                                       ignore_index=0)
   
               # 梯度缩放和异步反向传播
               # scaler.scale(loss).backward()
   
               # 梯度裁剪
               torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   
               # 参数更新
               # scaler.step(optimizer)
               # scaler.update()
               optimizer.step()
   
               if batch_idx % 100 == 0:
                   print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
               batch_idx += 1
   
   if __name__ == "__main__":
       train()
   ```

   

3. recompute.py

   ```python
   import torch
   import torch.nn  as nn
   from torch.utils.checkpoint  import checkpoint
   import logging
   
   # 配置 logging，包含文件名、函数名和行号
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
   )
   
   logger = logging.getLogger(__name__)
   
   class DecoderLayer(nn.Module):
       def __init__(self, d_model=512, nhead=8):
           super().__init__()
           self.self_attn  = nn.MultiheadAttention(d_model, nhead)
           self.ffn  = nn.Sequential(
               nn.Linear(d_model, d_model*4),
               nn.ReLU(),
               nn.Linear(d_model*4, d_model)
           )
           self.norm1  = nn.LayerNorm(d_model)
           self.norm2  = nn.LayerNorm(d_model)
   
       def forward(self, x):
           # 修正形状后的自注意力
           x_attn = x.transpose(0,  1)
           attn_output, _ = self.self_attn(x_attn,  x_attn, x_attn)
           attn_output = attn_output.transpose(0,  1)
           x = self.norm1(x  + attn_output)
   
           # 前馈网络部分
           ffn_output = self.ffn(x)
           x = self.norm2(x  + ffn_output)
           return x
   
   class CheckpointedDecoder(nn.Module):
       def __init__(self):
           super().__init__()
           self.decoder  = DecoderLayer()
   
       def forward(self, x):
           # 启用非重入式检查点
           return checkpoint(self.decoder,  x, use_reentrant=False)
   
   # 训练配置
   device = torch.device('cuda')
   model = CheckpointedDecoder().to(device)
   optimizer = torch.optim.Adam(model.parameters(),  lr=1e-4)
   criterion = nn.MSELoss()
   
   # 数据生成（建议实际数据应设置 requires_grad=False）
   x = torch.randn(32,  64, 512, device=device)
   target = torch.randn(32,  64, 512, device=device)
   
   # 训练循环
   for epoch in range(100):
       optimizer.zero_grad()
       output = model(x)
       loss = criterion(output, target)
       loss.backward()
       optimizer.step()
       logger.info(f'Epoch {epoch}, Loss: {loss.item():.4f}')
   ```

   