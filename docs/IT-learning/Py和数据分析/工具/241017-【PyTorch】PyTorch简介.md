---
Title: Introduction to PyTorch
---



# Introduction to PyTorch

::: tip 用途

仅使用于个人的专英课堂翻转，不过也算是基础了解PyTorch吧

:::

## 1. What is PyTorch?

PyTorch is a free, open-source tool developed by Facebook that helps people build and train artificial intelligence (AI) models. It’s especially useful for **deep learning**, a type of AI that tries to imitate how the human brain works to learn from large amounts of data. PyTorch is popular because it is simple to use, flexible, and allows for fast experiments, making it great for both research and real-world applications.

## 2. Key Features of PyTorch

### 2.1. **Dynamic Computation Graphs**

PyTorch uses **dynamic computation graphs**, meaning that it builds the graph of calculations step-by-step as you run the code. This makes it easy to try out new ideas, change the model, and find mistakes in the code while you’re working.

![img](/Users/ethan.liu/Downloads/torch动态图.gif)

### 2.2. **Autograd (Automatic Gradient Calculation)**

PyTorch has a system called **autograd** that automatically calculates the gradients, which are needed to update the model during training. This makes training easier because you don’t have to do the math manually.

### 2.3. **Tensors**

At the core of PyTorch are **tensors**, which are like special types of arrays (or lists of numbers). Tensors are used to store and process data. They can run on both a computer’s CPU or GPU, making it faster to train models, especially when working with big data.

### 2.4. **Easy to Use with Python**

PyTorch works really well with Python, a popular programming language. It’s designed to be intuitive and easy to write, so you can quickly test new ideas and build deep learning models.

### 2.5. **Rich Ecosystem of Libraries**

PyTorch has many **extra tools** to help with different kinds of tasks.**TorchVision**, **TorchText**, and **TorchAudio** are specialized PyTorch libraries that make working with images, text, and audio much easier. 

- **TorchVision** provides pre-trained models, image datasets, and transformation tools for tasks like image classification, object detection, and video analysis. 
- **TorchText** simplifies natural language processing (NLP) by offering text datasets, pre-trained word embeddings, and tools for processing raw text data for tasks such as sentiment analysis and text classification.
-  **TorchAudio** supports audio and speech-related tasks, providing pre-trained models, audio transformations, and datasets for tasks like speech recognition and audio classification. Together, these libraries enhance PyTorch’s flexibility across different types of machine learning projects.

## 3. Where is PyTorch Used?

### 3.1. **Research**

PyTorch is popular with researchers because it’s flexible and allows them to test new ideas easily. It’s used in many cutting-edge AI areas, such as language models, image recognition, and reinforcement learning.

### 3.2. **Industry**

Companies like Facebook, Uber, and Tesla use PyTorch for tasks such as image recognition, self-driving cars, and recommendation systems.

### 3.3. **Education**

Many universities and online courses use PyTorch to teach deep learning because it is simple, easy to understand, and works well with Python.

## 4. How To Use PyTorch？

### 4.1 Official Tutorials

Link to [**PyTorch official website**](https://pytorch.org/).

### 4.2 Basic Framework

Here’s a simple example of building a neural network in PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy data (32 samples of size 784)
inputs = torch.randn(32, 784)
labels = torch.randint(0, 10, (32,))

# Forward pass and calculate loss
outputs = model(inputs)
loss = criterion(outputs, labels)

# Backward pass and optimize
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
```

 **Explanation:**

- This code creates a simple neural network with three layers.
- It uses random data to train the model.
- The system calculates the loss (how wrong the model is) and then updates the model’s parameters automatically using **autograd**.

## 5. Conclusion

PyTorch is a powerful and easy-to-use tool for building AI models, especially for deep learning. It is widely used in both research and industry due to its flexibility and strong community support. Whether you’re just starting out or working on advanced AI projects, PyTorch makes deep learning more accessible.