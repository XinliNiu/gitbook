# 强化学习\(Reinforcement Learning\)

## 什么是强化学习？

强化学习是不同于监督学习与无监督学习的一种机器学习方法，它能让一个代理\(agent\)在一个交互式的环境\(environment\)通过自己的行动\(action\)得到的反馈\(reward\)不断的学习并完成任务。

与监督学习不一样，

Reinforcement Learning\(RL\) is a type of machine learning technique that enables an agent to learn in an interactive environment by trial and error using feedback from its own actions and experiences.

Though both supervised and reinforcement learning use mapping between input and output, unlike supervised learning where feedback provided to the agent is correct set of actions for performing a task, reinforcement learning uses rewards and punishment as signals for positive and negative behavior.

As compared to unsupervised learning, reinforcement learning is different in terms of goals. While the goal in unsupervised learning is to find similarities and differences between data points, in reinforcement learning the goal is to find a suitable action model that would maximize the total cumulative reward of the agent. The figure below represents the basic idea and elements involved in a reinforcement learning model.

![&#x5F3A;&#x5316;&#x5B66;&#x4E60;&#x6A21;&#x578B;](.gitbook/assets/reinforcement-learning-figure.jpg)

举一个极简单的例子，一个机器人要完成一项简单的任务，就是从A点到B点。机器人能完成的操作也很简单，就是朝某个方向前进有限的距离，而机器人每走完一次，能够测量到它与目的地的距离。如何让机器人从一开始什么都不知道，到能够以最快的速度从A点走到B点？也就是如何让机器人学会从A点走直线到B点。

（图）

这个问题看似简单，却能通俗易懂的说明强化学习的思想。我们把这个问题格式化一下。

机器人一开始位于A点，目的地为B点，我们用平面坐标系来表示A点B点分别为 $$(x_a, y_a)$$ , $$(x_b,y_b)$$ 。

机器人可以执行的操作就是转动 $$\theta$$ 度（只能逆时针转，最大360度\)，并且前进 S 的距离，机器人能测量它到目的地的距离D，以及正前方与目的地的夹角 $$\delta$$ 。

（图）

我们把这个问题具体一点，假设机器人一开始位于 $$(0,0)$$ ，目的地位于 $$(10,0)$$ ，机器人每次前进的最大距离为0.5。

最开始机器人掌握的信息有

 $$(D_0,\delta_0)=(10,0)$$ 

经过一次随机的前进，假如为

 $$(\theta_1,S_1)=(90,0.3)$$ 

那么此时的距离与角度为

$$(D_1,\delta_1)=(10.0045,90)$$ 

这次前进是好是坏呢？我们可以用回报\(reward\)来衡量，这里的回报可以用与目的地减少的距离来表示，也就是 $$R_1=-0.0045$$ ，可以看到，这次前进的回报很小，甚至是负的。

经过这一次前进，机器人能获得如下信息：在环境为\_\_时，做了——的动作，得到的回报为\_\_。现在机器人还没有得到足够的信息能让它做出好的决策。

接着机器人继续随机前进，

$$(\theta_2,S_2)=(300,0.5)  $$   =&gt; $$(D_2,\delta_2)=(9.5828,75.42)$$  =&gt; $$R2=0.4217$$ 

这次前进，机器人获得了如下信息：在环境为\_\_时，做了——的动作，得到的回报为\_\_。这次的回报是正的，说明这一次操作比上一次操作好。

经过不断的尝试，机器人得到了越来越多的信息，对环境的理解不断增强，所以称之为强化学习。而在这个过程中得到的数据，就类似于监督学习中的训练集，每一组数据就是在当前环境\(environment\)下做出某个行动\(action\)得到的回报\(reward\)。在强化学习中，我们不需要使用训练集，但是我们需要把环境，行动以及回报定义出来，agent通过强化学习算法不断的获取新的数据，并用于增强学习。

实际中应用强化学习的问题，要比上面的问题复杂的多，尤其对于reward的定义。比如著名的AlphaGo，它的environment很简单，就是一个棋盘加上棋子，action也很简单，就是选一个位置，最难的是如何判定reward。基本的思路就是计算棋局能赢的概率，比如在某个棋局下，选定了一个位置，赢棋的概率为95％，我们可以用95％来代表reward，但是如何计算出95％这个值是非常难的，AlphaGo通过监督学习\(深度神经网络\)为棋局加落子确定赢棋概率，并通过强化学习不断产生新的数据，使得它的能力不断提升，最终战胜了人类围棋高手。

对于其他的小游戏，例如扫雷，可以比较容易的建出模型，environment是当前地图，可以用一个矩阵表示，action是一个坐标，选一个未被挖掘的点，reward就是减少的雷点，当然如果遇上雷了，reward就清零了，这样的一个模型是比较简单的。

对于推箱子，问题与围棋其实类似，但是复杂程度远低于围棋。对于reward的定义，我们可以采用搜索算法中的启发式算法，比如计算每个箱子跟目标的距离，也可以采用监督学习，如果使用监督学习，它的数据集应该是这样的：

$$X=(State,Action)$$ $$(State \in R^n, Action \in \{1,2,3,4\})$$ \(1234对应上下左右\)

$$Y \in \{0,1\}$$ 

X是一个地图加上一个方向，Action代表一个方向。而Y的含义是：在当前State下，方向为Action时，是不是最优action，如何理解呢？

假如现在机器人处在某个位置，它如果向上走一步，那么接下来无论如何努力，也需要再走10步才能完成任务，如果向下走一步，它接下来最快需要5步，以此类推，向左需要8步，向右没法走，那可以假设向右需要一个特别大的步数，例如100000步。那么我们就能得到4个带有标签的数据：

$$X = \{CurrentState, 1), Y=0$$  

$$X = \{CurrentState, 2), Y=1$$ 

$$X = \{CurrentState, 3), Y=0$$ 

$$X = \{CurrentState, 4), Y=0$$ 

将X规范化成向量，就可以用现成的工具进行学习了，关键是如何取得这些数据。我们需要使用暴力算法找到最优路径，再从最优路径中提取训练集。

以上介绍了强化学习的基本模型，我们拿到了数据，接下来我们就来看一下如何使用强化学习算法让agent学习。













```
$ give me super-powers
```

{% hint style="info" %}
 Super-powers are granted randomly so please submit an issue if you're not happy with yours.
{% endhint %}

Once you're strong enough, save the world:

```
// Ain't no code for that yet, sorry
echo 'You got to trust me on this, I saved the world'
```



