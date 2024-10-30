# JavaSE 简介

JavaSE (Java Standard Edition) 是 Java 平台的核心版本，提供了开发桌面应用程序、服务端应用和基础库的标准 API。它是 Java 生态系统的基础。

## JavaSE 核心概念

### 1. JDK、JRE、JVM

- **JDK (Java Development Kit)**: Java 开发工具包，包含编译器、JRE 等工具，开发 Java 应用所需。
- **JRE (Java Runtime Environment)**: Java 运行环境，包含 JVM 和运行 Java 程序所需的类库。
- **JVM (Java Virtual Machine)**: Java 虚拟机，是 Java 程序运行的核心，它负责将字节码转化为机器码并执行。

### 2. Java 基础语法

- **变量**：`int age = 25;` 用于存储数据。
- **数据类型**：`int`, `float`, `double`, `char`, `boolean` 等。
- **运算符**：加法 `+`, 减法 `-`, 乘法 `*`, 除法 `/` 等。
- **条件语句**：`if`, `else`, `switch`。
- **循环**：`for`, `while`, `do-while`。

```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, JavaSE!");
    }
}
