# Python基础

### Python的安装



### 第一个Python程序







## 函数的参数

Python的函数参数特别灵活，有位置参数，默认参数，可变参数，关键字参数等。

### 位置参数

位置参数就是最普通的参数，比如要计算两个数的和，函数定义如下，有两个参数，使用的时候必须全部传入，而且要按函数定义中的顺序传入。

```python
#定义
def add(a, b):
  return a + b
#使用
sum = add(1, 2)
```

### 默认参数

假设有一个函数是打印温度，有两个参数，一个是数值，一个是温度单位，温度单位传入C表示摄氏度，F表示华氏度。

```python
#定义
def print_temperature(num, type):
   if type == 'C':
      print (num,'摄氏度')
   if type == 'F':
      print (num,'华氏度')
#使用
print_temperature(23,'C')
```

假设这个使用这个函数的场景大部分是摄氏度的情况，我们可以省略一个参数，而在函数定义中指定一个默认值。

```python
#定义
def print_temperature(num, type='C'):
   if type == 'C':
      print (num,'摄氏度')
   if type == 'F':
      print (num,'华氏度')
#使用
print_temperature(23) #打印 '23摄氏度'
print_temperature(23,'C') #打印 '23摄氏度'
print_temperature(70,'F') #打印 '70华氏度'
```

{% hint style="warning" %}
在定义函数的时候，如果要使用默认参数，那么默认参数后面不能再出现没有默认值的参数了，这时候会有编译错误，因为如果允许这样，python解释器就不知道参数的对应关系了。具体可以参见下例。
{% endhint %}

```python
#错误的默认参数定义
#如果允许这样定义，那么只传一个参数的时候编译器没法知道对应关系
def print_temperature(type='C', num):
   if type == 'C':
      print (num,'摄氏度')
   if type == 'F':
      print (num,'华氏度')
      
#运行报错
    def print_temperature(type='C', num):
                         ^
SyntaxError: non-default argument follows default argument
```

如果是用IDE编写代码，那么这种错误直接能在IDE中显示出来，因为这属于语法错误，也就是编译时错误。

![](.gitbook/assets/python1.png)

### 可变参数

前面定义了一个计算两个数的和的函数add，假如要实现三个数相加，需要重新定义一个add\(a, b, c\)，四个相加要定义add\(a, b, c, d\)。如果要实现一个任意个数相加的函数呢？这时候需要使用可变参数。

{% hint style="warning" %}
注意，一个函数只能有一个可变参数，否则就是语法错误。
{% endhint %}

```python
#可变参数定义
def sum(*args):
    count = len(args)
    sum = 0
    for i in range(0, count):
        sum = sum + args[i]
    return sum
#使用
x = sum(1,2,3) #1,2,3被封装成一个tuple放入args中
```

以上函数定义允许不传参数，假如我们要求至少传入两个参数，那么就使用位置参数和可变参数结合。

```python
#至少传入两个参数
def sum(a, b, *args):
   sum = a + b
   count = len(args)
   for i in range(0, count):
     sum = sum + args[i]
   return sum 
```

{% hint style="danger" %}
在语法上，并没有强制位置参数和默认参数必须在可变参数之前，但是假如在可变参数后面有位置参数和默认参数，那么它们都会被转换成keyword参数，也就是调用的时候必须写上参数名，因此建议不要这样做。
{% endhint %}

```python
#可变参数后有位置参数和默认参数
#不建议这样使用
def sum(*args, x, y=1):
    count = len(args)
    sum = 0
    for i in range(0, count):
        sum = sum + args[i]
    sum = sum + x + y
    return sum

#错误使用
sum(1,2,3)
#报错信息
    sum(1,2,3)
TypeError: sum() missing 1 required keyword-only argument: 'x'

#正确使用
sum(1,2,x=3) #y=1默认参数
sum(1,2,x=3,y=4) #修改了默认参数
```



### 关键字参数

关键字参数允许你传入0个或任意个含参数名的参数，这些关键字参数在函数内部自动组装为一个dict。

{% hint style="info" %}
一个函数只能有一个关键字参数，而且必须放到最后！
{% endhint %}

```python
#定义含有关键字参数的函数
#kwargs对于函数来说就是一个dict
def print_personal_info(name, age, **kwargs):
    print ("name,", name)
    print ("age,", age)
    if 'gender' in kwargs:
        print ("gender,", kwargs["gender"])
    if 'height' in kwargs:
        print("height,", kwargs["height"])

#使用关键字参数
print_personal_info("niuxinli", 29, gender='male')
```

### 参数的顺序

即使语法没有强制要求定义函数的时候按位置参数，默认参数，可变参数，关键字参数来传，但是我们一般都按这个顺序来传，否则容易出问题。

