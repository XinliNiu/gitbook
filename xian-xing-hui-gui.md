---
description: 介绍线性回归算法
---

# 线性回归\(Linear Regression\)

回归问题是监督学习的一大类，线性回归要解决的问题是，



| 学习时长x\(小时\) | 考试分数 |
| :--- | :--- |
| 5 | 40 |
| 6 | 49 |
| 4 | 30 |
| 5 | 39 |
| 7 | 58 |
| 9 | 70 |
| 8 | 62 |
| 2 | 15 |
| 10 | 85 |
| 12 | 97 |







$$
J(w,b) =\sum_{i=1}^{m}(\hat{y}^{i}-y^i)^2
$$

 



$$
J(w,b) =\frac{1}2\sum_{i=1}^{m}(wx^i+b-y^i)^2
$$

我们的目标就是找到w和b，使得cost函数能取得最小值，代价函数中除了w和b都是已知量，所以这个函数的参数就两个\(此处w是一个实数，不是向量\)，我们可以用梯度下降的方法计算代价函数取得最小值时的w和b。这就需要对w和b求偏导数。我们把代价函数展开。

$$
J(w,b) =\frac{1}2((wx^1+b-y^1)^2  + (wx^2+b-y^2)^2+(wx^3+b-y^3)^2+\\
...+(wx^m+b-y^m)^2)
$$

如果我们让

$$
J_i(w,b)=\frac{1}2(wx^i+b-y^i)^2\\
J(w,b)=J_1(w,b)+J_2(w,b)+...+J_m(w,b)
$$

  它对w的导数是

$$
\frac{\partial J_1}{\partial{w}}=(wx^i+b-y^i)x^i
$$

  

$$
\frac{\partial J}{\partial{w}}=\frac{\partial J_1}{\partial{w}}+\frac{\partial J_2}{\partial{w}}+...+\frac{\partial J_m}{\partial{w}}\\=(wx^1+b-y^1)x^i+(wx^2+b-y^2)x^i+...+(wx^m+b-y^m)x^m
$$



