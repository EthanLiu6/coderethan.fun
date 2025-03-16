## C++ 第 14 课：运算符重载与 String 类详解

**课程目标：**

* 理解运算符重载的概念和意义，掌握其基本语法和使用方法。
* 深入理解 `std::string` 类的特性和常用成员函数，能够灵活运用其进行字符串操作。
* 区分静态数组和动态数组，了解它们的特点和应用场景，并掌握动态数组的内存管理。
* 了解 C 风格字符串及其与 `std::string` 的区别。

---

### 1. 运算符重载

**1.1 运算符重载的概念和意义**

C++ 允许我们为自定义的数据类型重新定义现有运算符的行为。这意味着你可以使用像 `+`, `-`, `*`, `/` 等运算符来操作你的类对象，就像操作内置类型一样。运算符重载使得代码更加直观和易于理解。例如，对于表示点的 `Point` 类，我们可以重载 `+` 运算符来实现两个点的向量相加。

**1.2 运算符重载的语法**

运算符重载通过定义特殊的函数来实现，这些函数的名字是 `operator` 后面跟着要重载的运算符。

```cpp
返回类型 operator运算符(参数列表) {
  // 运算符的具体操作
}
```

**1.3 可重载与不可重载的运算符**

大多数 C++ 运算符都可以重载，但也有一些运算符是不能重载的，例如：

* `.` (成员访问运算符)
* `.*` (成员指针访问运算符)
* `::` (作用域解析运算符)
* `sizeof` (大小运算符)
* `typeid` (类型信息运算符)
* `?:` (条件运算符)
* `#` (预处理运算符)
* `##` (预处理连接运算符)

**1.4 重载运算符的两种方式：成员函数和友元函数**

* **成员函数重载：** 运算符函数作为类的成员。左操作数隐式地绑定到 `this` 指针。
* **友元函数重载：** 运算符函数定义在类外部，但被声明为类的友元。可以访问类的私有和保护成员。通常用于需要访问两个操作数私有成员的情况，或者当左操作数不是类的对象时（例如重载流运算符）。

**例子：使用成员函数重载 `+` 运算符**

```cpp
class Complex {
public:
  double real;
  double imag;

  Complex(double r = 0, double i = 0) : real(r), imag(i) {}

  // 重载 + 运算符 (作为成员函数)
  Complex operator+(const Complex& other) const {
    return Complex(real + other.real, imag + other.imag);
  }
};

int main() {
  Complex c1(2, 3);
  Complex c2(4, 5);
  Complex c3 = c1 + c2; // 等价于 c1.operator+(c2);
  cout << c3.real << " + " << c3.imag << "i" << endl;
  return 0;
}
```

**例子：使用友元函数重载输出流运算符 `<<`**

```cpp
#include <iostream>

class Complex {
public:
  double real;
  double imag;

  Complex(double r = 0, double i = 0) : real(r), imag(i) {}

  friend std::ostream& operator<<(std::ostream& os, const Complex& c);
};

std::ostream& operator<<(std::ostream& os, const Complex& c) {
  os << c.real << " + " << c.imag << "i";
  return os;
}

int main() {
  Complex c(2, 3);
  std::cout << c << std::endl; // 输出：2 + 3i
  return 0;
}
```

**1.5 不同类型的运算符重载示例**

* **算术运算符重载 (+, -, *, /):**  如上面 `Complex` 类的例子。
* **关系运算符重载 (==, !=, <, >, <=, >=):**  用于比较对象。
* **赋值运算符重载 (=):**  控制对象间的赋值行为，通常需要处理深拷贝。
* **复合赋值运算符重载 (+=, -=, *=, /=):**  结合了运算和赋值。
* **输入/输出流运算符重载 (<<, >>):**  用于自定义对象的输入和输出格式。
* **自增自减运算符重载 (++, --):**  区分前缀和后缀形式需要不同的函数签名。
* **下标运算符重载 ([]):**  允许像访问数组一样访问对象的元素。
* **函数调用运算符重载 (()):**  使得对象可以像函数一样被调用 (函数对象或仿函数)。

**1.6 运算符重载的注意事项和最佳实践**

* **保持运算符的自然语义：** 重载运算符的行为应该与内置类型的对应运算符保持一致。
* **不要滥用运算符重载：** 只在能够提高代码可读性和直观性的情况下使用。
* **考虑返回值类型：**  对于算术运算符，通常返回一个新的对象；对于赋值运算符，通常返回对象的引用 (`*this`)。
* **使用 `const` 修饰符：**  对于不会修改对象状态的运算符重载，应该使用 `const` 修饰。
* **注意运算符的优先级和结合性：** 重载运算符不能改变其优先级和结合性。

---

### 2. String 类详解

**2.1 `std::string` 类的引入和优势**

`std::string` 类是 C++ 标准库中用于处理字符串的类。相比于 C 风格的字符数组，`std::string` 提供了以下优势：

* **自动内存管理：** 无需手动分配和释放内存，避免了内存泄漏和缓冲区溢出的风险。
* **动态大小：** 字符串的长度可以根据需要动态增长。
* **丰富的成员函数：** 提供了各种方便的字符串操作函数。
* **更安全易用：**  避免了 C 风格字符串操作中常见的错误。

**2.2 `std::string` 类的常用构造函数**

```cpp
#include <string>
#include <iostream>

int main() {
  std::string s1;                      // 默认构造函数，创建一个空字符串
  std::string s2("Hello");             // 使用 C 风格字符串初始化
  std::string s3 = "World";            // 拷贝初始化
  std::string s4(s2);                  // 拷贝构造函数
  std::string s5(s2, 1, 3);            // 使用 s2 的子串初始化 (从索引 1 开始，长度为 3)
  std::string s6(10, 'a');             // 创建包含 10 个 'a' 的字符串
  std::string s7(s2.begin(), s2.end()); // 使用迭代器初始化

  std::cout << "s1: " << s1 << std::endl;
  std::cout << "s2: " << s2 << std::endl;
  std::cout << "s3: " << s3 << std::endl;
  std::cout << "s4: " << s4 << std::endl;
  std::cout << "s5: " << s5 << std::endl;
  std::cout << "s6: " << s6 << std::endl;
  std::cout << "s7: " << s7 << std::endl;

  return 0;
}
```

**2.3 `std::string` 类的常用操作函数**

| 函数           | 描述                                                         | 示例                                                              |
| -------------- | ------------------------------------------------------------ | ----------------------------------------------------------------- |
| `+` (连接)     | 连接两个字符串或字符串与字符/C 风格字符串                    | `std::string s = "Hello" + " " + "World";`                         |
| `+=` (追加)    | 将字符串或字符/C 风格字符串追加到现有字符串末尾                | `std::string s = "Hello"; s += " World!";`                       |
| `append`       | 在字符串末尾追加字符串或其子串                               | `std::string s = "Hello"; s.append(" World");`                    |
| `find`         | 查找子串第一次出现的位置，找不到返回 `std::string::npos`      | `std::string s = "abcdefg"; size_t pos = s.find("cde");`          |
| `rfind`        | 查找子串最后一次出现的位置                                     | `std::string s = "abcdecdef"; size_t pos = s.rfind("cde");`      |
| `find_first_of` | 查找字符串中第一个与指定字符集合中任何字符匹配的字符的位置    | `std::string s = "abcdefg"; size_t pos = s.find_first_of("ce");` |
| `substr`       | 返回指定范围的子串                                           | `std::string s = "abcdefg"; std::string sub = s.substr(1, 3);`  |
| `insert`       | 在指定位置插入字符串或字符                                   | `std::string s = "abc"; s.insert(1, "xyz");`                     |
| `replace`      | 替换指定范围内的子串                                         | `std::string s = "abcdefg"; s.replace(1, 3, "XYZ");`           |
| `erase`        | 删除指定位置或范围内的字符                                   | `std::string s = "abcdefg"; s.erase(1, 3);`                      |
| `length` / `size` | 返回字符串的长度 (字符个数)                                   | `std::string s = "Hello"; int len = s.length();`                 |
| `capacity`     | 返回当前已分配的存储空间大小 (可能大于实际长度)               | `std::string s = "Hello"; int cap = s.capacity();`               |
| `empty`        | 检查字符串是否为空                                           | `std::string s; bool isEmpty = s.empty();`                      |
| `clear`        | 清空字符串内容，使其长度变为 0                                  | `std::string s = "Hello"; s.clear();`                            |
| `compare`      | 比较两个字符串                                               | `std::string s1 = "abc"; std::string s2 = "abd"; int res = s1.compare(s2);` |
| `c_str`        | 返回一个指向以 null 结尾的 C 风格字符串的常量指针 (const char*) | `std::string s = "Hello"; const char* cstr = s.c_str();`
示例：字符串的操作 |

```cpp
#include <iostream>
#include <string>

using namespace std;

int main() {
  string str = "Hello";
  string world = " World";

  // 连接字符串
  string greeting = str + world;
  cout << "Greeting: " << greeting << endl; // 输出：Hello World

  // 追加字符串
  greeting += "!";
  cout << "Greeting: " << greeting << endl; // 输出：Hello World!

  // 查找子串
  size_t pos = greeting.find("World");
  if (pos != string::npos) {
    cout << "'World' found at position: " << pos << endl; // 输出：'World' found at position: 6
  }

  // 截取子串
  string sub = greeting.substr(0, 5);
  cout << "Substring: " << sub << endl; // 输出：Hello

  // 替换子串
  greeting.replace(6, 5, "Universe");
  cout << "Greeting after replace: " << greeting << endl; // 输出：Hello Universe!

  return 0;
}
```

**2.4 访问字符串元素**

与 C 风格字符串类似，`std::string` 允许通过下标访问单个字符，并提供了更安全的 `at()` 方法。

| 方法    | 描述                                             | 示例                       |
| ------- | ------------------------------------------------ | -------------------------- |
| `[]`    | 通过下标访问字符串元素，**不进行越界检查**，可能导致未定义行为 | `char c = str[0];`          |
| `at()`  | 通过下标访问字符串元素，**进行越界检查**，越界会抛出 `std::out_of_range` 异常 | `char c = str.at(0);`       |
| `front()` | 返回字符串的第一个字符的引用                         | `char c = str.front();`     |
| `back()`  | 返回字符串的最后一个字符的引用                         | `char c = str.back();`      |

**示例：访问字符串元素**

```cpp
#include <iostream>
#include <string>
#include <stdexcept> // 需要包含这个头文件来处理异常

using namespace std;

int main() {
  string str = "Hello";

  cout << "First character: " << str[0] << endl;       // 输出：H
  cout << "First character: " << str.at(0) << endl;     // 输出：H
  cout << "Last character: " << str.back() << endl;      // 输出：o

  try {
    // str.at(10); // 这会抛出 std::out_of_range 异常
  } catch (const out_of_range& e) {
    cerr << "Caught an out_of_range exception: " << e.what() << endl;
  }

  return 0;
}
```

**2.5 `std::string` 的迭代器**

`std::string` 也支持迭代器，可以用于遍历字符串中的字符。

* `begin()`: 返回指向字符串第一个字符的迭代器。
* `end()`: 返回指向字符串末尾之后一个位置的迭代器。
* `rbegin()`: 返回指向字符串最后一个字符的反向迭代器。
* `rend()`: 返回指向字符串开头之前一个位置的反向迭代器。

**示例：使用迭代器遍历字符串**

```cpp
#include <iostream>
#include <string>

using namespace std;

int main() {
  string str = "Hello";

  cout << "Using forward iterators: ";
  for (string::iterator it = str.begin(); it != str.end(); ++it) {
    cout << *it << " "; // 输出：H e l l o
  }
  cout << endl;

  cout << "Using reverse iterators: ";
  for (string::reverse_iterator rit = str.rbegin(); rit != str.rend(); ++rit) {
    cout << *rit << " "; // 输出：o l l e H
  }
  cout << endl;

  // C++11 引入的基于范围的 for 循环更加简洁
  cout << "Using range-based for loop: ";
  for (char c : str) {
    cout << c << " "; // 输出：H e l l o
  }
  cout << endl;

  return 0;
}
```

**2.6 `std::string` 的内存管理**

`std::string` 负责管理其内部存储字符串数据的内存。当你对 `std::string` 对象进行操作（如追加、插入等）时，如果当前的内存空间不足以容纳新的字符串，`std::string` 会自动分配更大的内存空间，并将原有数据复制到新的空间。 这就是为什么使用 `std::string` 可以避免手动内存管理带来的问题。

---

### 3. 静态数组和动态数组

**3.1 静态数组**

* **3.1.1 定义和初始化：**  在声明时指定数组的大小。

```cpp
int arr1[5]; // 声明一个包含 5 个整数的数组 (未初始化)
int arr2[5] = {1, 2, 3, 4, 5}; // 声明并初始化
int arr3[] = {1, 2, 3}; // 编译器自动推断大小为 3
```

* **3.1.2 内存分配：栈区:** 静态数组的内存在栈上分配。
* **3.1.3 大小固定，编译时确定:** 数组的大小必须在编译时已知，不能在运行时改变。
* **3.1.4 访问速度快:**  可以直接通过内存地址计算访问元素，速度较快。
* **3.1.5 示例:**  

**3.2 动态数组**

* **3.2.1 使用 `new` 运算符分配内存:**  在堆上动态分配内存。

```cpp
int* ptr = new int[10]; // 分配包含 10 个整数的动态数组
```

* **3.2.2 内存分配：堆区:** 动态数组的内存在堆上分配。需要程序员手动管理。
* **3.2.3 大小可在运行时确定:** 可以在程序运行时根据需要分配不同大小的数组。
* **3.2.4 需要手动使用 `delete` 或 `delete[]` 释放内存，防止内存泄漏:**

```cpp
delete[] ptr; // 释放动态分配的数组内存
ptr = nullptr; // 良好的编程习惯，防止悬空指针
```

* **3.2.5 灵活性高:**  可以根据程序的需要动态地调整数组的大小。
* **3.2.6 示例:** 

**3.3 `std::vector`：动态数组的更安全选择**

* **3.3.1 `std::vector` 的优势：自动内存管理、动态扩容:** `std::vector` 是 C++ 标准库提供的动态数组容器。它封装了动态内存管理，你无需手动分配和释放内存。当 `vector` 的容量不足以容纳新元素时，它会自动重新分配更大的内存空间。
* **3.3.2 基本用法和常用操作:**

```cpp
#include <iostream>
#include <vector>

using namespace std;

int main() {
  // 创建一个存储整数的 vector
  vector<int> vec;

  // 添加元素
  vec.push_back(10);
  vec.push_back(20);
  vec.push_back(30);

  // 访问元素
  cout << "First element: " << vec[0] << endl;     // 输出：10
  cout << "Second element: " << vec.at(1) << endl;    // 输出：20

  // 获取大小和容量
  cout << "Size: " << vec.size() << endl;         // 输出：3
  cout << "Capacity: " << vec.capacity() << endl;     // 输出：可能是 3 或更大

  // 遍历元素
  cout << "Elements: ";
  for (int i = 0; i < vec.size(); ++i) {
    cout << vec[i] << " "; // 输出：10 20 30
  }
  cout << endl;

  // 删除元素
  vec.pop_back(); // 删除最后一个元素
  cout << "Size after pop_back: " << vec.size() << endl; // 输出：2

  // 使用迭代器遍历
  cout << "Elements using iterator: ";
  for (vector<int>::iterator it = vec.begin(); it != vec.end(); ++it) {
    cout << *it << " "; // 输出：10 20
  }
  cout << endl;

  return 0;
}
```

---

### 4. 补充：C 风格字符串

* **4.1 C 风格字符串的定义和特点:**  C 风格字符串是以 null 字符 `'\0'` 结尾的 `char` 类型数组。

```cpp
char c_str[] = {'H', 'e', 'l', 'l', 'o', '\0'}; // 显式声明 null 结尾
char c_str2[] = "World"; // 编译器会自动添加 null 结尾
```

* **4.2 `char*` 与 `char[]`:**
    * `char[]` 定义了一个字符数组，可以直接操作数组中的元素。
    * `char*` 是一个指向字符的指针，可以指向字符数组的首地址。

```cpp
char arr[] = "Hello";
char* ptr = arr; // ptr 指向 'H'
```

* **4.3 C 风格字符串的常用函数 (`strlen`, `strcpy`, `strcat`, `strcmp`):** 这些函数定义在 `<cstring>` 头文件中。

```cpp
#include <cstring>
#include <iostream>

using namespace std;

int main() {
  char str1[] = "Hello";
  char str2[20];

  // 获取长度
  cout << "Length of str1: " << strlen(str1) << endl; // 输出：5

  // 复制字符串
  strcpy(str2, str1);
  cout << "str2: " << str2 << endl; // 输出：Hello

  // 连接字符串
  char str3[50] = "Hello ";
  strcat(str3, "World");
  cout << "str3: " << str3 << endl; // 输出：Hello World

  // 比较字符串
  char s1[] = "apple";
  char s2[] = "banana";
  int result = strcmp(s1, s2);
  if (result < 0) {
    cout << "s1 comes before s2" << endl;
  } else if (result > 0) {
    cout << "s1 comes after s2" << endl;
  } else {
    cout << "s1 and s2 are equal" << endl;
  }

  return 0;
}
```

* **4.4 `std::string` 与 C 风格字符串的转换:**
    * `std::string` to C 风格字符串: 使用 `string::c_str()` 方法。
    * C 风格字符串 to `std::string`:  可以直接赋值或在构造函数中使用。

```cpp
#include <string>
#include <iostream>

using namespace std;

int main() {
  string cpp_str = "Hello from string";
  const char* c_str = cpp_str.c_str(); // 获取 C 风格字符串 (const char*)
  cout << "C-style string: " << c_str << endl;

  char c_arr[] = "Hello from C-array";
  string cpp_str2 = c_arr; // C 风格字符串转换为 string
  cout << "C++ string: " << cpp_str2 << endl;

  return 0;
}
```

* **4.5 为什么推荐使用 `std::string`:**
    * **安全性:**  `std::string` 自动管理内存，避免了缓冲区溢出等安全问题。
    * **方便性:**  提供了丰富的成员函数，简化了字符串操作。
    * **效率:**  现代 `std::string` 的实现通常经过优化，性能良好。
    * **避免手动内存管理:**  减少了出错的可能性，提高了开发效率。

**总结：**

本节课我们学习了运算符重载、`std::string` 类的使用以及静态数组和动态数组的概念。运算符重载使得我们可以自定义运算符对类对象的操作行为，`std::string` 类提供了方便安全的字符串处理方式，而对数组的理解是内存管理的基础。在实际开发中，优先推荐使用 `std::string` 和 `std::vector`，因为它们能提供更好的安全性和便捷性。


## 作业：

编写一个程序，实现以下功能：

1. 要求用户输入一个整数 `n`，表示数组的大小。
2. 动态分配一个大小为 `n` 的整型数组。
3. 要求用户输入 `n` 个整数，并将其存储到数组中。
4. 计算数组中所有元素的平均值，并输出。
5. 释放动态分配的内存。