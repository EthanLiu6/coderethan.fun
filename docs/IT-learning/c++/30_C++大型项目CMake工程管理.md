# C++28课：高性能部署之CMake工程管理

## 1. 什么是 CMake？
CMake 是一个跨平台的构建系统生成工具，它帮助我们管理编译过程，可以生成特定于编译器的项目文件（比如 Makefile、Visual Studio 的 Solution 文件等），从而简化编译、链接等过程。

### 1.1. 为什么选择 CMake？
- 跨平台支持
- 自动处理依赖关系
- 强大的脚本和配置语言
- 与现代 C++ 标准协作良好

## 2. 基本结构
CMake 使用 `CMakeLists.txt` 文件描述如何构建您的项目。

### 2.1. 最简单的 CMakeLists.txt 示例
创建一个名为 `hello_world` 的简单项目：

```cmake
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(HelloWorld)

# 指定要编译的源文件
add_executable(hello_world main.cpp)
```

### 2.2. 生成与构建
#### 2.2.1. Linux/ macOS
```bash
mkdir build
cd build
cmake ..
make
```

#### 2.2.2. Windows
使用 Visual Studio：

```bash
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019"
```
然后用 Visual Studio 打开生成的 `.sln` 文件。

## 3. CMake 基本命令详解
### 3.1. cmake_minimum_required
指定 CMake 的最低版本：

```cmake
cmake_minimum_required(VERSION 3.10)
```

### 3.2. project
定义项目的名称和版本号：

```cmake
project(MyProject VERSION 1.0)
```

### 3.3. add_executable
添加一个可执行文件，并指定其源文件：

```cmake
add_executable(hello_world main.cpp)
```

### 3.4. add_library
添加一个库文件，并指定其源文件：

```cmake
add_library(MyLib STATIC mylib.cpp mylib.h)
```

### 3.5. target_link_libraries
将库文件链接到可执行文件：

```cmake
target_link_libraries(hello_world PRIVATE MyLib)
```

### 3.6. include_directories
指定包含目录：

```cmake
include_directories(${PROJECT_SOURCE_DIR}/include)
```

### 3.7. find_package
查找其他库或包，举例如下：

```cmake
find_package(Boost 1.60 REQUIRED)
```

## 4. 组织大型项目
### 4.1. 子目录和子模块
大型项目通常分为多个子模块，可以通过`add_subdirectory`命令进行管理。

#### 4.1.1. 主目录的 CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.10)
project(SuperProject)

add_subdirectory(ModuleA)
add_subdirectory(ModuleB)

add_executable(my_project main.cpp)
target_link_libraries(my_project PRIVATE ModuleA ModuleB)
```

#### 4.1.2. 子模块的 CMakeLists.txt (例如 ModuleA)
```cmake
# ModuleA/CMakeLists.txt
add_library(ModuleA STATIC moduleA.cpp moduleA.h)
```

### 4.2. 设置全局属性
您可以用 `set` 命令设置全局属性，比如设置全局的C++ 标准：

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
```

## 5. CMake 高级用法
### 5.1. 定义编译选项
使用 `target_compile_options` 指定编译选项：

```cmake
target_compile_options(MyLib PRIVATE -Wall -Werror)
```

### 5.2. 条件编译
通过 `if` 语句实现条件编译：

```cmake
if (WIN32)
    target_compile_definitions(MyLib PRIVATE WINDOWS)
elseif (UNIX)
    target_compile_definitions(MyLib PRIVATE LINUX)
endif()
```

### 5.3. 测试支持
CMake支持集成测试，可以使用 `add_test` 添加测试：

```cmake
enable_testing()

add_executable(my_tests test.cpp)
add_test(NAME MyTest COMMAND my_tests)
```

### 5.4. 安装目标
指定安装规则：

```cmake
install(TARGETS hello_world DESTINATION bin)
install(FILES "${PROJECT_SOURCE_DIR}/config.txt" DESTINATION etc)
```

使用以下命令进行安装：

```bash
cmake --install .
```

## 6. CMake 与第三方依赖管理
### 6.1. 使用 ExternalProject_Add
用于管理外部项目的构建：

```cmake
include(ExternalProject)
ExternalProject_Add(external_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    PREFIX ${CMAKE_BINARY_DIR}/external/json
)
```

### 6.2. 使用 FetchContent
用于更方便地管理外部依赖：

```cmake
include(FetchContent)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.9.1
)
FetchContent_MakeAvailable(json)
```

### 6.3. 使用 find_package
依赖管理时最常见的方式：

```cmake
find_package(Boost 1.60 REQUIRED COMPONENTS filesystem system)
target_link_libraries(my_project PRIVATE Boost::filesystem Boost::system)
```

## 7. CMake 与自定义命令
### 7.1. 添加自定义命令
通过 `add_custom_command` 添加原始命令：

```cmake
add_custom_command(
    OUTPUT output.txt
    COMMAND ${CMAKE_COMMAND} -E echo "Hello" > output.txt
    DEPENDS input.txt
)
```

### 7.2. 自定义目标
通过 `add_custom_target` 定义一个伪目标：

```cmake
add_custom_target(MyTarget ALL DEPENDS output.txt)
```

## 8. 生成与构建
### 8.1. 常见命令总结
- 配置项目: `cmake ..`
- 编译项目: `make` 或 `cmake --build .`
- 运行测试: `ctest`
- 安装项目: `cmake --install .`

## CMake 课后作业

**目标:**

- 熟悉 CMake 的基本使用方法
- 掌握使用 CMake 构建包含多个源文件和库的项目

**任务:**

设计并实现一个简单的计算器程序，该程序包含以下功能：

1. **加法:** 两个整数相加。
2. **减法:** 两个整数相减。
3. **乘法:** 两个整数相乘。
4. **除法:** 两个整数相除。

**要求:**

1. 使用 CMake 构建项目。
2. 将计算器的功能实现放在一个单独的库 (例如 `calculator_lib`) 中。
3. 创建一个可执行文件 (例如 `calculator`)，调用 `calculator_lib` 中的功能实现计算器的功能。

**示例代码结构:**



```txt
calculator_project/
├── CMakeLists.txt
├── src/
│   ├── CMakeLists.txt
│   ├── calculator.cpp  # 可执行文件源代码
│   └── ...
├── lib/
│   ├── CMakeLists.txt
│   ├── calculator_lib.cpp  # 库源代码
│   └── ...

```

## 作业：

## CMake 课后作业参考答案

以下是 CMake 课后作业的参考答案，包含代码结构、CMakeLists.txt 文件、源代码和测试用例。

**代码结构:**

```
calculator_project/
├── CMakeLists.txt
├── src/
│   ├── CMakeLists.txt
│   └── calculator.cpp
├── lib/
│   ├── CMakeLists.txt
│   └── calculator_lib.cpp
```

**CMakeLists.txt (根目录):**

```cmake
cmake_minimum_required(VERSION 3.10)
project(CalculatorProject)

add_subdirectory(lib)
add_subdirectory(src)

# 可选: 添加安装规则
# install(TARGETS calculator DESTINATION bin)
# install(TARGETS calculator_lib DESTINATION lib)
```

**CMakeLists.txt (lib 目录):**

```cmake
add_library(calculator_lib calculator_lib.cpp)
```

**CMakeLists.txt (src 目录):**

```cmake
add_executable(calculator calculator.cpp)
target_link_libraries(calculator calculator_lib)
```


**calculator_lib.cpp:**

```cpp
#include "calculator_lib.h"

int add(int a, int b) {
  return a + b;
}

int subtract(int a, int b) {
  return a - b;
}

int multiply(int a, int b) {
  return a * b;
}

int divide(int a, int b) {
  if (b == 0) {
    return 0; // 处理除零错误
  }
  return a / b;
}
```

**calculator_lib.h:**

```cpp
#ifndef CALCULATOR_LIB_H
#define CALCULATOR_LIB_H

int add(int a, int b);
int subtract(int a, int b);
int multiply(int a, int b);
int divide(int a, int b);

#endif // CALCULATOR_LIB_H
```

**calculator.cpp:**

```cpp
#include <iostream>
#include "calculator_lib.h"

using namespace std;

int main() {
  int a = 10;
  int b = 5;

  cout << "a + b = " << add(a, b) << endl;
  cout << "a - b = " << subtract(a, b) << endl;
  cout << "a * b = " << multiply(a, b) << endl;
  cout << "a / b = " << divide(a, b) << endl;

  return 0;
}
```

**构建和运行:**

1. 创建 `build` 目录: `mkdir build`
2. 进入 `build` 目录: `cd build`
3. 运行 CMake: `cmake ..` -G "Unix Makefiles"
4. 构建项目: make
5. 运行可执行文件: `./src/calculator`

