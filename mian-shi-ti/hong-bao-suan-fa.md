---
description: 发红包算法
---

# 红包算法

## 题目

输入一个金额和红包个数，返回每个红包的金额。

要实现两种方法，一种是点击一次发一个，另一种是一次性发放完毕。

```java
/** 每次分配一个红包 **/
int allocate(int amount, int remainAmount, int count, int remainCount);
/** 一次性分配完毕  **/
int[] allocate(int amount, int count);
```

## 分析

红包分配的金额应该平均，单次的随机性比较大，多次分配后，每个红包的平均值应该一样。



