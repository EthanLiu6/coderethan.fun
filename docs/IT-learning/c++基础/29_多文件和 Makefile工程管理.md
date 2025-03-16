## 27 课 多文件和 Makefile工程管理 

### 一、C++ 多文件编程 9

**1. 引入：为什么要使用多文件编程？ 1**

* **代码规模**: 随着项目的规模扩大，将所有代码放在一个文件中会变得难以维护和理解。
* **模块化**: 多文件编程允许将代码分解成独立的模块，提高代码的可重用性和可维护性。
* **协作开发**: 多个开发者可以独立开发不同的模块，然后将它们组合在一起。
* **编译效率**: 只需重新编译修改过的文件，提高编译效率。

**示例：** 假设我们要开发一个计算器程序，可以将其分解成以下模块：

  * `calculator.cpp`: 主程序文件，负责用户交互和调用其他模块。
  * `addition.cpp`: 加法模块，实现加法运算。
  * `subtraction.cpp`: 减法模块，实现减法运算。

**2. 头文件和源文件 3**

* **头文件 (.h)**: 
    * 声明函数、类、变量等，提供接口信息。
    * 使用 `#include` 指令包含在其他文件中。
    * **避免在头文件中定义变量或函数**，会导致重复定义错误。
    * 使用 `#ifndef`、`#define`、`#endif` 预处理指令防止头文件重复包含。

**示例：`addition.h`**

```c++
#ifndef ADDITION_H
#define ADDITION_H

int add(int a, int b);

#endif
```

* **源文件 (.cpp)**:
    * 包含头文件。
    * 定义函数、类、变量等，实现具体功能。

**示例：`addition.cpp`**
```c++
#include "addition.h"

int add(int a, int b) {
  return a + b;
}
```

**3. 声明与定义 2**

* **声明**: 告诉编译器某个函数、类或变量的存在，但不提供具体实现。
* **定义**: 提供函数、类或变量的具体实现。

**示例：**
```c++
// 声明
int add(int a, int b);

// 定义
int add(int a, int b) {
  return a + b;
}
```

**4. 编译和链接 3**

* **编译**: 将每个源文件编译成目标文件 (.o)。
* **链接**: 将多个目标文件和库文件链接成可执行文件。

**示例 (使用 g++ 编译器):**

```bash
g++ -c addition.cpp -o addition.o
g++ -c subtraction.cpp -o subtraction.o
g++ -c calculator.cpp -o calculator.o
g++ calculator.o addition.o subtraction.o -o calculator
```

### 二、Makefile 15

**1. 引入：什么是 Makefile？1**

* Makefile 是一个文本文件，描述了项目中文件之间的依赖关系和编译规则。
* 使用 make 工具可以根据 Makefile 自动编译项目。
* 提高编译效率，简化编译过程。

**2. Makefile 基本语法4**

* **目标 (target)**: 要生成的文件或执行的操作。
* **依赖 (dependencies)**: 生成目标文件所依赖的其他文件。
* **命令 (commands)**: 生成目标文件需要执行的命令，**必须以 Tab 键开头**。

**基本格式:**
```makefile
target: dependencies
<Tab>commands
```

**示例:**

```makefile
calculator: calculator.o addition.o subtraction.o
<Tab>g++ calculator.o addition.o subtraction.o -o calculator

calculator.o: calculator.cpp
<Tab>g++ -c calculator.cpp -o calculator.o

addition.o: addition.cpp addition.h
<Tab>g++ -c addition.cpp -o addition.o

subtraction.o: subtraction.cpp subtraction.h
<Tab>g++ -c subtraction.cpp -o subtraction.o
```

**3. 变量3**

* **定义变量**: `variable = value`
* **使用变量**: `$(variable)`

**示例:**
```makefile
CXX = g++
CXXFLAGS = -Wall -g

calculator: calculator.o addition.o subtraction.o
<Tab>$(CXX) $(CXXFLAGS) calculator.o addition.o subtraction.o -o calculator

# ... 其他规则 ...
```

**4. 自动化推导规则 2**

* make 工具可以根据文件后缀自动推导编译规则。

**示例:**

```makefile
CXX = g++
CXXFLAGS = -Wall -g

calculator: calculator.o addition.o subtraction.o
<Tab>$(CXX) $(CXXFLAGS) $^ -o $@

# ... (不需要手动编写 .o 文件的规则) ...
```

**5. 常用特殊目标 (10分钟)**

* **clean**: 清理生成的文件
* **all**: 编译所有目标

**示例:**

```makefile
# ... 其他规则 ...

clean:
<Tab>rm -f *.o calculator

all: calculator
```

**6. 一个简易的Makefile 模板 3**

```makefile
# 可执行文件名
TARGET = myprogram

# 源文件
SOURCES = main.cpp file1.cpp file2.cpp

# 对象文件
OBJECTS = $(SOURCES:.cpp=.o)

# 编译器
CXX = g++

# 编译选项
CXXFLAGS = -Wall -g -std=c++11

# 链接选项
LDFLAGS = 

# 默认目标
all: $(TARGET)

# 链接规则
$(TARGET): $(OBJECTS)
<Tab>$(CXX) $(LDFLAGS) $(OBJECTS) -o $@

# 编译规则
%.o: %.cpp
<Tab>$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理规则
clean:
<Tab>rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
```

##  **作业** 

根据以上知识，尝试编写一个 Makefile 来编译一个简单的 C++ 项目，包含至少两个源文件和一个头文件。

**总结:** 
* 多文件编程是大型 C++ 项目开发的必要手段。
* Makefile 可以自动化编译过程，提高开发效率。
* 掌握 Makefile 的基本语法和常用技巧，可以更好地管理 C++ 项目。

(1) main.cpp, 2个 cpp文件，2 个头文件

(2) 简易模板

(3) 可以增加一些注释



