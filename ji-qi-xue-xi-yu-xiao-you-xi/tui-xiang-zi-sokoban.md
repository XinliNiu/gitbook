---
description: 本文介绍推箱子问题点常规解法以及如何用机器学习优化推箱子
---

# 推箱子\(Sokoban\)

## 推箱子游戏介绍

推箱子是一个很常见点小游戏，在功能机时代，几乎每个手机上都装了推箱子。

## 推箱子中的对象

![](../.gitbook/assets/gameandml_sokoban_3.png)



| 对象 | 字符表示 | 图标 |
| :--- | :--- | :--- |
| PlayerOnly | @ | ![](../.gitbook/assets/gameandml_sokoban_4.png)  |
| PlayerInGoal | + | ![](../.gitbook/assets/gameandml_sokoban_5.png)  |
| BoxOnly | $ | ![](../.gitbook/assets/gameandml_sokoban_6.png)  |
| BoxInGoal | \* | ![](../.gitbook/assets/gameandml_sokoban_7.png)  |
| Wall | \# | ![](../.gitbook/assets/gameandml_sokoban_8.png)  |
| Road | - | ![](../.gitbook/assets/gameandml_sokoban_10.png)  |
| GoalOnly | . \(注：点） | ![](../.gitbook/assets/gameandml_sokoban_9.png)  |

一个推箱子中的对象包含上述内容，因此我们要构造一个地图，用符号即可。

```text
##########
#--------#
#.$@-----#
#--------#
#.$------#
#--------#
##########
```

上述字符就对应着如下点推箱子问题：

![](../.gitbook/assets/gameandml_sokoban_11.png)



## 推箱子问题常规解法

推箱子问题是一个典型的搜索问题。最开始我们拿到一个初始状态，最终我们要达到一个终态，而终态就是所有的箱子都在目标中。



