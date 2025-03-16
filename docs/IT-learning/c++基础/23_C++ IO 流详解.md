# 第 21 课：C++ IO 流详解

## 1. 什么是 IO 流？

在 C++ 中，IO 流（Input/Output Stream）是一种用于处理程序与外部世界（如文件、键盘、屏幕等）之间数据交互的机制。可以将 IO 流想象成一个“流动的管道”，数据从一端“流入”程序，或从程序“流出”到另一端。

**核心思想：** 将输入输出操作抽象为字节序列的流动。

**主要优点：**

*   **统一接口：** 无论数据来源是文件、键盘还是网络，都可以使用相同的 IO 流操作。
*   **类型安全：** C++ IO 流具有类型检查，可以避免许多常见的输入输出错误。
*   **可扩展性：** 可以自定义 IO 流，以处理特殊的数据格式或设备。
*    **支持缓冲：** 提高读写效率

## 2. C++ 标准 IO 流对象

C++ 提供了四个预定义的标准 IO 流对象，它们都包含在 `<iostream>` 头文件中：

| 对象   | 描述                                 |
| ------ | ------------------------------------ |
| `cin`  | 标准输入流（通常与键盘关联）         |
| `cout` | 标准输出流（通常与屏幕关联）         |
| `cerr` | 标准错误流（无缓冲，通常与屏幕关联） |
| `clog` | 标准日志流（有缓冲，通常与屏幕关联） |

**示例：**

```cpp
#include <iostream>

int main() {
  int age;
  std::cout << "请输入您的年龄：";  // 提示用户输入
  std::cin >> age;                // 从标准输入读取整数

  std::cout << "您的年龄是：" << age << std::endl; // 输出到标准输出

  if (age < 0) {
    std::cerr << "错误：年龄不能为负数！" << std::endl; // 输出错误信息
  }

  return 0;
}
```

## 3. 输入流（Input Stream）

输入流用于从外部设备（如键盘、文件）读取数据。`cin` 是最常用的标准输入流对象。

### 3.1. `cin` 常用方法

*   `operator>>`：从输入流中提取数据，并根据目标变量的类型进行转换。
*   `get()`：
    *   `cin.get()`：读取一个字符（包括空白字符）。
    *   `cin.get(char& ch)`：读取一个字符到 `ch`。
    *   `cin.get(char* str, streamsize n)`：读取最多 `n-1` 个字符到 `str`，遇到换行符或文件结束符停止。
    *    `cin.get(char* str, streamsize n, char delim)`: 读取最多 `n-1` 个字符到 `str`，遇到 `delim` 或换行符或文件结束符停止
*   `getline()`：
    *   `cin.getline(char* str, streamsize n)`：读取一行文本（最多 `n-1` 个字符），遇到换行符或文件结束符停止。
    *   `cin.getline(char* str, streamsize n, char delim)`：读取一行文本（最多 `n-1` 个字符），遇到 `delim` 或换行符或文件结束符停止。
*   `ignore()`：
    *    `cin.ignore()`： 忽略输入流中的一个字符。
    *   `cin.ignore(streamsize n)`: 忽略输入流中的 `n` 个字符
    *    `cin.ignore(streamsize n, int delim)`: 忽略 `n` 个字符，或者在遇到定界符 `delim` 时停止
*   `peek()`：查看输入流中的下一个字符，但不将其从流中移除。
*   `gcount()`：返回上一次非格式化输入操作读取的字符数。

**示例：**

```cpp
#include <iostream>
#include <string>

int main() {
  char ch;
  std::cout << "请输入一个字符：";
  std::cin.get(ch);  // 读取一个字符（包括空格）
  std::cout << "您输入的字符是：" << ch << std::endl;

  char buffer[80];
  std::cout << "请输入一行文本：";
  //std::cin.ignore();
  std::cin.getline(buffer, 80);  // 读取一行文本
  std::cout << "您输入的文本是：" << buffer << std::endl;

    // 使用 string 和 getline() 读取一行
  std::string line;
  std::cout << "请输入另一行文本 (string): ";
  //std::cin.ignore(); //有多个cin时，有时需要清除缓冲区，防止上一次的输入影响当前读取行
  std::getline(std::cin, line);
  std::cout << "您输入的文本是: " << line << std::endl;
    
  return 0;
}
```
### 3.2 输入流的状态
输入流对象（如 `cin`）具有内部状态标志，用于指示流的状态。可以使用以下成员函数检查这些状态：
- `good()`: 如果流处于正常状态（没有错误发生），返回 `true`。
- `eof()`: 如果到达文件末尾（End-of-File），返回 `true`。
- `fail()`: 如果发生可恢复的错误（例如，试图读取一个整数，但输入了非数字字符），返回 `true`。
- `bad()`: 如果发生不可恢复的错误（例如，硬件故障），返回 `true`。

在处理输入时，检查流状态非常重要，以确保输入操作成功完成，并处理可能出现的错误。

```c++
#include <iostream>
#include <limits> // 用于 numeric_limits

int main() {
    int num;

    std::cout << "请输入一个整数: ";
    while (!(std::cin >> num)) {
        // 输入失败，检查原因
        if (std::cin.eof()) {
            std::cerr << "已到达输入末尾！" << std::endl;
            break; // 退出循环
        } else if (std::cin.fail()) {
            std::cerr << "输入错误，请重新输入一个整数: ";
            std::cin.clear(); // 清除错误状态
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // 忽略无效输入
        } else if (std::cin.bad()) {
            std::cerr << "发生严重错误，无法继续！" << std::endl;
            return 1; // 退出程序
        }
    }

    if (std::cin.good()) {
        std::cout << "您输入的整数是: " << num << std::endl;
    }

    return 0;
}
```

## 4. 输出流（Output Stream）

输出流用于向外部设备（如屏幕、文件）写入数据。`cout` 是最常用的标准输出流对象。

### 4.1. `cout` 常用方法

*   `operator<<`：将数据插入到输出流中。
*   `put()`：输出一个字符。
*   `write()`：输出指定数量的字符。

**示例：**

```cpp
#include <iostream>

int main() {
  char ch = 'A';
  std::cout.put(ch);  // 输出字符 'A'
  std::cout << std::endl;

  const char* str = "Hello, world!";
  std::cout.write(str, 5);  // 输出字符串的前 5 个字符 "Hello"
  std::cout << std::endl;

  return 0;
}
```

### 4.2. 格式化输出

C++ 提供了多种方式来控制输出的格式，包括：

*   **使用操纵符（Manipulators）：** 包含在 `<iomanip>` 头文件中。

    *   `std::setw(n)`：设置字段宽度为 `n`。
    *   `std::setprecision(n)`：设置浮点数精度为 `n` 位小数。
    *   `std::fixed`：以固定点表示法显示浮点数。
    *   `std::scientific`：以科学计数法表示浮点数。
    *   `std::left`：左对齐。
    *   `std::right`：右对齐（默认）。
    *   `std::internal`：将填充字符插入到符号和数字之间（对于数值）。
    *   `std::hex`：以十六进制显示整数。
    *   `std::oct`：以八进制显示整数。
    *   `std::dec`：以十进制显示整数（默认）。
    *   `std::boolalpha`：将 `bool` 值显示为 `true` 或 `false`。
    *    `std::noboolalpha`: 将 `bool` 值显示为 1或0

**示例：**

```cpp
#include <iostream>
#include <iomanip>

int main() {
  double pi = 3.14159265358979323846;

  std::cout << "默认精度：" << pi << std::endl;
  std::cout << "精度为 5：" << std::setprecision(5) << pi << std::endl;
  std::cout << "固定点表示法：" << std::fixed << pi << std::endl;
  std::cout << "科学计数法：" << std::scientific << pi << std::endl;

  int num = 255;
  std::cout << "十进制：" << std::dec << num << std::endl;
  std::cout << "十六进制：" << std::hex << num << std::endl;
  std::cout << "八进制：" << std::oct << num << std::endl;
  std::cout << "十进制：" << std::dec << num << std::endl; //恢复到十进制

  bool flag = true;
  std::cout << "boolalpha: " << std::boolalpha << flag << std::endl;   //输出 true
  std::cout << "noboolalpha: "<< std::noboolalpha << flag << std::endl; //输出 1
  return 0;
}
```

## 5. 文件 IO

除了标准输入输出流，C++ 还提供了文件流，用于读写文件。

### 5.1. 文件流类

*   `ifstream`：输入文件流，用于从文件读取数据。
*   `ofstream`：输出文件流，用于向文件写入数据。
*   `fstream`：文件流，既可以读取也可以写入数据。

这些类都包含在 `<fstream>` 头文件中。

### 5.2. 文件操作步骤

1.  **打开文件：** 使用 `open()` 方法打开文件，指定文件名和打开模式。
2.  **读/写文件：** 使用 `operator>>`、`operator<<`、`get()`、`getline()`、`read()`、`write()` 等方法进行读写操作。
3.  **关闭文件：** 使用 `close()` 方法关闭文件。

**打开模式：**

| 模式          | 描述                                                     |
| ------------- | -------------------------------------------------------- |
| `ios::in`     | 以读取模式打开文件（默认）                               |
| `ios::out`    | 以写入模式打开文件（默认），如果文件存在，则清空文件内容 |
| `ios::app`    | 以追加模式打开文件，在文件末尾添加内容                   |
| `ios::ate`    | 打开文件后，将文件指针定位到文件末尾                     |
| `ios::trunc`  | 如果文件存在，则清空文件内容                             |
| `ios::binary` | 以二进制模式打开文件                                     |
| 可以用        | 来组合多种模式                                           |

**示例：读取文件**

```cpp
#include <iostream>
#include <fstream>
#include <string>

int main() {
  std::ifstream inputFile("example.txt"); // 打开文件

  if (inputFile.is_open()) {  // 检查文件是否成功打开
    std::string line;
    while (std::getline(inputFile, line)) {  // 逐行读取
      std::cout << line << std::endl;        // 输出每一行
    }
    inputFile.close();  // 关闭文件
  } else {
    std::cerr << "无法打开文件！" << std::endl;
  }

  return 0;
}
```

**示例：写入文件**

```cpp
#include <iostream>
#include <fstream>

int main() {
  std::ofstream outputFile("output.txt");  // 以写入模式打开文件
  //等效于 std::ofstream outputFile("output.txt", std::ios::out);
    
  if (outputFile.is_open()) {
    outputFile << "Hello, file!" << std::endl;
    outputFile << "This is a new line." << std::endl;
    outputFile.close();
  } else {
    std::cerr << "无法创建文件！" << std::endl;
  }
     // 以二进制模式写入文件
    std::ofstream binaryFile("data.bin", std::ios::binary);
    if(binaryFile.is_open()){
       int data[] = {1,2,3,4,5};
       binaryFile.write(reinterpret_cast<char*>(data), sizeof(data));
       binaryFile.close();
    } else {
         std::cerr << "无法创建文件！" << std::endl;
    }
  return 0;
}
```
### 5.3 文件的随机访问
`seekg()` 和 `seekp()` 允许随机访问文件中的数据。`seekg()` 用于输入流 (如 `ifstream`)，而 `seekp()` 用于输出流 (如 `ofstream`)。

- `seekg(offset, origin)` 和 `seekp(offset, origin)`
  - `offset`: 相对于 `origin` 的偏移量（可以是正数或负数）。
  - `origin`: 参考点，可以是以下值之一：
    - `ios::beg`: 文件开头。
    - `ios::cur`: 当前位置。
    - `ios::end`: 文件结尾。
-  `tellg()` 和 `tellp()`:
    -  这两个函数用于获取当前文件指针位置,`tellg()` 用于输入流，`tellp()` 用于输出流。

示例
```cpp
#include <iostream>
#include <fstream>

int main() {
    // 写入一些数据到文件
    std::ofstream outFile("example.txt");
    outFile << "abcdefghijklmnopqrstuvwxyz";
    outFile.close();

    // 打开文件进行读取
    std::ifstream inFile("example.txt");
    if (!inFile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    // 获取文件大小
    inFile.seekg(0, std::ios::end);
    long fileSize = inFile.tellg();
    std::cout << "File size: " << fileSize << " bytes" << std::endl;

    // 移动到文件中间
    inFile.seekg(fileSize / 2, std::ios::beg);
    char ch;
    inFile.get(ch);
    std::cout << "Character at the middle: " << ch << std::endl;  //输出m

    // 从当前位置向后移动 3 个字符
    inFile.seekg(3, std::ios::cur);
    inFile.get(ch);
    std::cout << "Character 3 positions forward: " << ch << std::endl;  //输出p

    // 移动到文件末尾之前的 5 个字符
    inFile.seekg(-5, std::ios::end);
    std::string lastFive;
    char buffer[6]; //  包含空终止符
    inFile.read(buffer, 5);
    buffer[5] = '\0'; // 确保以空字符结尾
    lastFive = buffer;
    std::cout << "Last five characters: " << lastFive << std::endl; //输出 vwxyz
    inFile.close();
    return 0;
}
```

## 6. 字符串流（String Stream）

字符串流提供了一种在内存中操作字符串的 IO 流方式。它允许你像处理文件或标准输入输出一样处理字符串。

### 6.1. 字符串流类

*   `istringstream`：输入字符串流，用于从字符串读取数据。
*   `ostringstream`：输出字符串流，用于向字符串写入数据。
*   `stringstream`：字符串流，既可以读取也可以写入数据。

这些类都包含在 `<sstream>` 头文件中。

### 6.2. 字符串流用法
主要操作：
-   `str()`：
    -   无参数：返回字符串流中当前包含的字符串的副本。
    -   有参数（`const std::string& s`）：用字符串 `s` 的内容替换字符串流的当前内容。
-   `operator<<`（插入运算符）：将各种类型的数据插入到字符串流中。这些数据会被转换为字符串形式。
-   `operator>>`（提取运算符）：从字符串流中提取数据，并将其转换为适当的类型。
-   `clear()`: 清除字符串流的错误状态。
-   `rdbuf()`: 用于访问或操作与流关联的底层缓冲区。

```cpp
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

int main() {
// 将整数转换为字符串
int num = 123;
stringstream ss;
ss << num;
string str = ss.str();
cout << "String: " << str << std::endl;

// 将字符串转换为整数
string str2 = "456";
stringstream ss2(str2);
int num2;
ss2 >> num2;
cout << "Integer: " << num2 << std::endl;

// 字符串拼接
stringstream ss3;
ss3 << "Hello" << " " << "World!";
string result = ss3.str();
cout << "Concatenated string: " << result << std::endl;

return 0;
}
```



## 7、作业

**题目：**

1. 读取文件中的数据（学生的成绩），根据姓名构造 `Student` 类。
2. 计算均分，并换算成 A-E 等级后，再写入另一个文件。
   * 等级划分：A[90,100] B[80,90) C[70,80) D[60,70) E[0,60)

**提示：**

* 可以使用 `fstream` 读取和写入文件。
* 可以使用 `stringstream` 辅助字符串解析和数据类型转换。

**下面是参考代码 :**

```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

class Student {
public:
    string name;
    int chinese;
    int math;
    int english;
    char grade;

    Student(string name, int chinese, int math, int english) :
        name(name), chinese(chinese), math(math), english(english) {
        calculateGrade();
    }

    void calculateGrade() {
        double average = (chinese + math + english) / 3.
        if (average >= 90) {
            grade = 'A';
        } else if (average >= 80) {
            grade = 'B';
        } else if (average >= 70) {
            grade = 'C';
        } else if (average >= 60) {
            grade = 'D';
        } else {
            grade = 'E';
        }
    }
};

int main() {
    // 读取学生成绩数据
    ifstream inputFile("student_scores.txt");
    if (!inputFile.is_open()) {
        cerr << "Error opening input file!" << endl;
        return 1;
    }

    vector<Student> students;
    string line;
    while (getline(inputFile, line)) {
        stringstream ss(line);
        string name;
        int chinese, math, english;
        ss >> name >> chinese >> math >> english;
        students.push_back(Student(name, chinese, math, english));
    }
    inputFile.close();

    // 写入学生成绩和等级
    ofstream outputFile("student_grades.txt");
    if (!outputFile.is_open()) {
        cerr << "Error opening output file!" << endl;
        return 1;
    }

    for (const Student& student : students) {
        outputFile << student.name << " " << student.chinese << " " 
                   << student.math << " " << student.english << " " 
                   << student.grade << endl;
    }
    outputFile.close();

    cout << "Grades calculated and written to student_grades.txt" << endl;

    return 0;
}
```


**作业要求:**

* 创建两个文件 `student_scores.txt` 和 `student_grades.txt`。
* 在 `student_scores.txt` 中输入学生成绩数据，格式为：`姓名 语文 数学 英语`，每行一个学生。
* 运行程序后，`student_grades.txt` 中将包含学生的成绩和对应的等级。

---


## 8. 总结

本次课程我们详细讲解了 C++ IO 流的各个方面，包括：

*   IO 流的基本概念和标准 IO 流对象 (`cin`, `cout`, `cerr`, `clog`)。
*   输入流 (`cin`) 和输出流 (`cout`) 的常用方法和格式化输出。
*   文件 IO (`ifstream`, `ofstream`, `fstream`) 的操作步骤和打开模式。
*   文件的随机访问。
*   字符串流 (`istringstream`, `ostringstream`, `stringstream`) 的用法。

掌握 IO 流是 C++ 编程的基础，希望大家通过本次课程的学习，能够熟练运用 IO 流处理各种输入输出任务。

