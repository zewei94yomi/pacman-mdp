![Pacman](https://thelogicalindian.com/h-upload/2021/03/17/192284-thelogicalindianfb1000x600-1.jpg)



# Pacman | MDP | Value Iteration 

## Introduction

This project is a coursework of KCL MSc AI, Artificial Intelligence & Decision Making module.

It uses **value iteration** to solve **MDP** and helps pacman win the game. 



## Version

Python 2.7



## How to run

1. Download the project

2. Use the following command (make sure you have Python 2.7 installed on your machine)

   1. **small grid**

      ```shell
      python pacman.py -p MDPAgent -l smallGrid
      ```

   2. medium classic

      ```shell
      python pacman.py -p MDPAgent -l mediumClassic
      ```

3. Evaluate 'mdpAgent'

   1. **small grid**

      ```shell
      python pacman.py -q -n 25 -p MDPAgent -l smallGrid
      ```

   2. medium classic

      ```shell
      python pacman.py -q -n 25 -p MDPAgent -l mediumClassic
      ```




> If you use MacOS **Monterey** to run the command `python pacman.py -p MDPAgent -l smallGrid` or `python pacman.py -p MDPAgent -l mediumClassic`, then you might find it fail to display graphics. But it's ok with evaluation commands which only display text in your terminal. 
>
> Two options for Monterey users:
>
> 1. delete default Python 2.7 and install a new Python 2.7 from fficial Python.org website
> 2. run on a virtual machine



## Demo

demo video for my MDPAgent on mediumClassic layout

https://www.youtube.com/watch?v=j-k60swUKuE





# Analysis

## 1. 概要

## 1.0 前言

最近刚写完AI Reasoning & Decision Making的Coursework，占这门课总成绩的20%。因为不要求写Report，所以在这里单独做一个记录和总结。从CW放出到ddl大概有一个月的时间，工作量不大，难度也较低，在Github上有很多Pacman基于MDP的代码可以参考借鉴。个人认为难点是在于调整项目中的各类参数，如Reward，和制定策略来优化游戏结果（胜率、得分）。



## 1.1 游戏

### 1.1.1 原版

[吃豆人Pacman](https://www.google.com/logos/2010/pacman10-i.html)

原版是需要玩家手动控制，而我们这门课叫'AI Reasoning & Decision Making'，那自然是要编写AI来自动决策的。



### 1.1.2 小地图

![4v6bjWINtqe5Emk](https://s2.loli.net/2021/12/06/4v6bjWINtqe5Emk.png)



### 1.1.3 中地图

![dqPCZx6jlGioa5E](https://s2.loli.net/2021/12/06/dqPCZx6jlGioa5E.png)



## 1.2 要求

- **要求0**：整个Coursework拿到时是可运行的，无需重构或实现除了Pacman外的任何功能。

- **要求1**：为Pacman编写代码，使得Pacman可以基于**MDP**来做出决策，躲避Ghost，并成功完成游戏。

- **要求2**：不可以显式地告诉Pacman要往哪走，即不可以直接修改utility来影响Pacman的决策，或直接控制Pacman的移动。

- **要求3**：只得通过Pacman项目提供的api来获取数据和信息，不得访问除该方式外获得的数据或修改任何除mdpAgents.py外的文件。不得使用除了MDP外的算法完成游戏，如Reinforcement Learning中的Q-Learning。

- > Your code must be based on solvin g the Pacman environment as an MDP. If y ou don’t submit a p ro g ram that contains a reco g nisable MDP solver, y ou will lose marks.

  > The onl y MDP solvers we will allow are the ones p resented in the lecture, i.e., Value iteration, Polic y iteration and Modiﬁed p olic y iteration. In p articular, Q -Learnin g is unacce p table.

  > Your code must only use the results of the MDP solver to decide what to do. If y ou submit code which makes decisions about what to do that uses other information in addition to what the MDP-solver g enerates ( like ad-hoc g host avoidin g code, for exam p le ) , y ou will lose marks. This is to ensure that y our MDP-solver is the thin g that can win enou g h g ames to p ass the functionalit y test.

- **评估方式**：在小地图和中等地图中各运行Pacman游戏25次，Pacman需要在小地图和中等地图中取得较高胜率，并且在中等地图中获胜的场次中取得较高分数。即**小地图**要取得**高胜率**，**中地图**要取得**高胜率**和**高分数**。

- 总体评分标准：![RWeFLZ7Of4BCPSh](https://s2.loli.net/2021/12/06/RWeFLZ7Of4BCPSh.png)

- 得分计算公式：![8MzIxR4oalC29h3](https://s2.loli.net/2021/12/06/8MzIxR4oalC29h3.png)

- **补充**：

  - 游戏设定为non-deterministic，Pacman做出决策向一个方向移动有0.8的概率成功，有0.1的概率向目标方向的左边，有0.1的概率向目标方向的右边。如果移动到的位置是墙壁，那么就回到原地。
  - ![Bgr4AI6LMielHN2](https://s2.loli.net/2021/12/06/Bgr4AI6LMielHN2.png)
  - Pacman具有global visibility，即可以看到所有Ghost、Wall、Food和Capsule的位置。在Coursework发布之前的几个lab中做过更难的版本，Pacman只有partial visibility，即只能获取面朝方向、一定距离内的环境信息，难度大。



# 2. 方法



## 2.1 Bellman Equation

在环境中不同的事物有不同的reward，比如Pacman吃到food加分，food的reward通常就是正数；Pacman遇到ghost游戏结束，ghost的reward通常就是负数。在一个状态上所有的动作的发生的概率乘以该动作导致的状态的utility（效用）的和，再加上这个状态的reward，就是这个状态的utility。这就是**Bellman Equation**：

![GeJizQnrUvCmNb1](https://s2.loli.net/2021/12/06/GeJizQnrUvCmNb1.png)

***s***: current state.

***U(s)***: expected utility of current state

***R(s)***: reward of state ***s***.

***γ***: discount factior, is used to weaken the effect of utility of other state

***A(s)***: all actions that can be taken by Pacman at state ***s*** .

***P(s'|s, a)***: at state ***s***, the probability of whether Pacman can be state ***s'*** at the next state if taking action ***a***.

***U(s')***: utility of next state ***s'*** 



如下，假如现在在(1, 1)状态，那么U(1, 1)的计算过程如下，注意如果移动的方向是墙壁就留在原地：

![GZf3XoOHCcLvqRi](https://s2.loli.net/2021/12/06/GZf3XoOHCcLvqRi.png)



## 2.2 Optimal Policy

Policy就是采取的行动/动作，即Bellman Equation中的action。Optimal Policy，顾名思义，就是采取最优选择。假如对所有状态都有正确的utility values，那么Pacman的决策就非常简单：只需要选择一个action可以最大化下一个状态的expected utility（期望效用）：

![kzYd5CuIjJ8bvtK](https://s2.loli.net/2021/12/06/kzYd5CuIjJ8bvtK.png)

在上面的例子中，当Pacman在(1, 1)状态时，optimal policy就是***Up***。



## 2.2 Markov Decision Process

[马尔可夫决策过程 (Markov decision process)](https://en.wikipedia.org/wiki/Markov_decision_process)是这个Coursework的核心，Pacman的所有决策本质就是MDP solver。解决MDP问题可以采用：Value iteration，Policy iteration或者Modified policy iteration。我采用的是Value iteration，并且取得较好的表现。



### 2.2.1 Value Iteration

为了选择optimal policy，我们需要正确的utility。而要获得在所有状态在一个瞬间的正确utility，我们可以进行value iteration：

1. 随机初始化每个状态的utility（通常初始化为0）
2. 使用Bellman Equation来更新一个状态的utility（同时对所有状态进行更新），每次更新时计算的utility应当是上一轮的utility。
3. 不断使用Bellman Equation更新utility，直至最终拟合，停止。
4. （PPT里说value iteration最终保证可以拟合在最优的值上，在代码中为减少运行压力和时间，我设定了当误差小于一定值即认为拟合）



**pseudocode** ：

![4MDzqXj5fIZtvEh](https://s2.loli.net/2021/12/06/4MDzqXj5fIZtvEh.png)





# 3. 实践

## 3.1 过程

Pacman的核心代码其实很简单，做的事情主要是：

1. 在每一回合开始时，根据当前环境情况，更新不同物体/状态的reward
2. 根据reward，进行value iteration，直至拟合；更新每个状态的MEU（Maximum Expected Utility）
3. Pacman基于所在位置的四个方向计算并得出optimal policy，并做出该行动



## 3.2 调优 

在整个Value iteration的过程中，utility无需我们改变，而reward和gamma是需要我们人为设置和调整的。这个coursework实现value iteration很简单，但是要拿高分就需要fine tune整个模型。需要fine tune的参数就是**reward**（还有gamma）。



### 3.2.1 Reward类型

允许调整的参数基本就是以下物体的reward：

- Empty state：即空白格，没有任何物体的格子
- Food：食物，吃了会加分，应当设为正数
- Capsule：胶囊，吃了会加分，并且ghost减速，Pacman可以在一定时间内吃掉ghost并加分，应当设为正数
- Ghost：ghost分为两种，一种是初始的，会抓住Pacman结束游戏，应当设为负数；还有一种是Pacman吃了Capsule后的ghost，我称之为scaredGhost，Pacman可以吃掉，并加很多分，应当设为正数



显然，同一局游戏中，同一个时刻，不同的物体因为有着不同属性，reward应当区别设定；随着游戏进行，游戏环境和状态的不断变化，reward也应该动态地变化。这就引出了reward设定的strategy。



### 3.2.2 Reward设定策略

> 虽然说是我们规定策略，但从Pacman的角度，reward是它对世界/环境的认知，根据这个认知，结合他自己的思考和计算（Bellman equation + value itertion），得出对自己利益最大化的选择（optimal policy）。所以这里的strategy也可以解释成Pacman自己的策略。这个策略可以认为是Pacman如何提取、解释世界/环境中的信息与知识。

#### 1. 辐射Ghost的Reward

这一策略的主要目的是让Pacman提前躲避ghost，防止Pacman进入潜在的危险区域，提前避险，降低Pacman死亡率，从而提高游戏胜率。

通过debug和实验发现，对ghost设定非常小（负）的reward并没有对Pacman躲避ghost起到很好作用，原因是在value iteration的过程中，其utility被max函数消除了，导致ghost的utility不能很好传导出去。所以策略之一就是以ghost所在位置为起点，将ghost的reward辐射到附近区域。这里需要使用曼哈顿距离，同时考虑到墙壁的遮挡。可以使用BFS来实现。此外，距离ghost越远，辐射的reward越小；辐射值应当累加到当地的reward上，避免两个ghost之间互相覆盖。

![65QBxl8nFMbj2X9](https://s2.loli.net/2021/12/06/65QBxl8nFMbj2X9.jpg)



#### 2. 追逐Ghost

这一策略的主要目的是提高Pacman得分率。

当Pacman吃下胶囊后，理论上应当马上吃掉两个ghost，然后吃食物，再接着吃掉一个胶囊，再吃掉两个ghost，最后完成游戏，并且要尽可能快，这样才能在单局游戏中拿到高分。而要激励Pacman在吃掉胶囊后去吃ghost的方法就是提高scared ghost的reward。但也要考虑一个问题，就是能否赶在胶囊药效内追上ghost（赶到ghost所在位置）。这里我采用BFS来计算pacman在地图里到ghost的曼哈顿距离，并根据药效时间，判断能否赶上。如果可以就提高ghost的reward（此时同样辐射这个正值的reward），Pacman就会跟着utility一路追逐ghost。如果发现时间不够，就设置为普通的scared ghost reward（同样是正值，比food稍小）即可。

![aHqeUcumYZVGfx5](https://s2.loli.net/2021/12/06/aHqeUcumYZVGfx5.png)



#### 3. 食物Reward的抑制

这一策略的主要目的是提高Pacman在小地图的胜率。

一些食物处在四通八达的位置，即十字路口，这样的位置为Pacman的路线和动作提供了很多选择，如果ghost正在追击，Pacman也有更高概率逃脱；在双向的通道内，Pacman较为窘困；在死胡同内（只在小地图中有），Pacman更是非常容易死亡。

如下面两幅图中绿色的food，就是位置不太好，流通性不好的food；红色的food，就是处于三岔路口或十字路口，变通性高的food。经测试，不采用该策略，在小地图中，Pacman经常走进死胡同吃正中间的food，而被ghost堵截。所以我们可以提升被一面墙围绕或没有墙围绕的food的reward，而抑制/减小被两面或三面墙围绕的food的reward：`reward = foodReward / (1 + wallCount) `

而采用该策略后，Pacman通常会优先吃外面的food，胜率得到提升。



![Rxs9NofUaYpkIi3](https://s2.loli.net/2021/12/06/Rxs9NofUaYpkIi3.png)

![6WZpqFUPcA5EGX1](https://s2.loli.net/2021/12/06/6WZpqFUPcA5EGX1.png)



#### 4. Terminal State

这一策略的主要目的是提高Pacman在小地图的胜率。

在使用策略3后发现，Pacman先吃左下角food的概率确实提高了，但是也容易因为死胡同中的food的reward被抑制，而哪怕经过胡同也不进去。此时就可以把最后这个food作为terminal state，提高他的reward。因为Pacman在吃完后就不需要管后面的事情，哪怕ghost就堵在口子上也没关系。这也算是一个小trick。



# 4. 结果与评估

## 4.1 小地图

运行25次：

对于小地图，因为ghost是随机运动的，Pacman的运动也是non-deterministic的，运行25次获胜的场次从14～24次不等，这里截取随机一次的结果：

![ob8q7RVwMnxA3O2](https://s2.loli.net/2021/12/06/ob8q7RVwMnxA3O2.png)



运行1000次：

运行1000次比较可以用来评估前面调参的好坏，小地图的胜率基本稳定在0.74：

![69trxBEFRTVomky](https://s2.loli.net/2021/12/06/69trxBEFRTVomky.png)



## 4.2 中地图

运行25次：

因为增加了追逐Ghost的策略，所以胜率较低。如果不加的话胜率可以稳定在0.6左右。

![NIiyRqHzt6e7fJx](https://s2.loli.net/2021/12/06/NIiyRqHzt6e7fJx.png)



因为中地图较大，运行起来较慢，这里就不运行多次了。

最终25局得分可以稳定在4000分左右。
