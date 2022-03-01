[TOC]



注：灰色代码框内的文字为本人口语化的总结，可以直接用来回答问题。

⭐️：提问频率比较高

⚠️：待补充/不确定



## 模型

### word2vec

https://www.zhihu.com/question/44832436/answer/131725613

它将每个词映射到一个固定长度的向量，这些向量能更好地表达不同**词之间的相似性和类比关系**。（使得语义相似的单词在嵌入式空间中的距离很近）

word2vec 本质上是一种**降维**操作——把词语从 one-hot encoder 形式的表示降维到更低维度的向量的表示。

<!--神经网络迭代训练一定次数，得到**输入层到隐藏层的参数矩阵**，矩阵中每一行的转置即是对应词的词向量-->



**实施过程**

https://www.cnblogs.com/zhangyang520/p/10969975.html



#### skip-gram

跳元模型假设**一个词**可以用来在文本序列中**生成其周围的单词**

**跳元模型考虑了在给定中心词的情况下生成周围上下文词的条件概率。**

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxksw8th39j31iy0icq4d.jpg" alt="截屏2021-12-21 01.14.36" style="zoom:33%;" />



<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkswol66nj31n60bkn00.jpg" alt="截屏2021-12-21 01.15.04" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxksyecyjnj31n00eimzu.jpg" alt="截屏2021-12-21 01.16.42" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxksyxl4r6j31be0u0wjb.jpg" alt="截屏2021-12-21 01.17.08" style="zoom: 67%;" />

对词典中索引为 i 的词进行训练后，得到 vi（作为中心词）和 ui（作为上下文词）两个词向量。在自然语言处理应用中，**跳元模型的中心词向量通常用作词表示。**



**Skip-gram处理步骤**

1. 确定窗口大小window，对每个词生成2*window个训练样本，(i, i-window)，(i, i-window+1)，...，(i, i+window-1)，(i, i+window)

2. 确定batch_size，注意batch_size的大小必须是2*window的整数倍，这确保每个batch包含了一个词汇对应的所有样本

3. 训练算法有两种：层次Softmax和Negative Sampling

4. 神经网络迭代训练一定次数，得到**输入层到隐藏层的参数矩阵**，矩阵中每一行的转置即是对应词的词向量





#### CBOW

连续词袋模型假设**中心词**是基于其在文本序列中的**周围上下文词生成的**

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkt163k6fj31260igdgt.jpg" alt="截屏2021-12-21 01.19.22" style="zoom:50%;" />

**连续词袋模型考虑了给定周围上下文词生成中心词条件概率**

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkt29is9wj31l60u00z6.jpg" alt="截屏2021-12-21 01.20.25" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkt2uilacj31ny0req7u.jpg" alt="截屏2021-12-21 01.20.56" style="zoom:50%;" />



**CBOW的处理步骤**

1. 确定窗口大小window，对每个词生成2*window个训练样本，(i-window, i)，(i-window+1, i)，...，(i+window-1, i)，(i+window, i)

2. 确定batch_size，注意batch_size的大小必须是2*window的整数倍，这确保每个batch包含了一个词汇对应的所有样本

3. 训练算法有两种：层次Softmax和Negative Sampling

4. 神经网络迭代训练一定次数，得到输入层到隐藏层的参数矩阵，矩阵中每一行的转置即是对应词的词向量



**两种优化方法：分层softmax、负采样**

由于softmax操作的性质，上下文词可以是词表V中的任意项。

计算log条件概率时，包含与整个词表大小一样多的项的求和，而在一个词典上（通常有几十万或数百万个单词）求和的梯度的计算成本是巨大的。



为了解决这个问题，我们直观的想法就是**限制每次必须更新的输出向量的数量**。一种有效的手段就是采用**分层softmax**；另一种可行的方法是通过**负采样**。



#### hierarchical softmax

```
我自己的理解：原有的softmax，需要与词表中每一个单词做内积。
分层softmax就是把词表的单词放在叶子结点上，每个叶子节点生成词的概率可以通过从根节点到叶子结点的唯一路径进行计算，就是每次向左和向右走的概率。
这样只需经过logV次向量内积运算，模型复杂度从V降到了logV
```

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gzcf2hxcemj30yu044weq.jpg" alt="截屏2022-02-14 01.52.21" style="zoom:50%;" />

本质是把 N 分类问题变成 log(N)次二分类

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkteq0ab1j311w0lyjta.jpg" alt="截屏2021-12-21 01.32.23" style="zoom:50%;" />

用二叉树来表示词汇表中的所有单词。V个单词必须存储于二叉树的叶子单元。可以被证明一共有V-1个内单元。对于每个叶子节点，**有一条唯一的路径可以从根节点到达该叶子节点；该路径被用来计算该叶子结点所代表的单词的概率。**

一个单词作为输出词的概率被定义为：

从根节点出发到叶子结点的随机路径。在每个隐节点（包含根节点），我们需要分配往左走或往右走的概率。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxktgw6r8sj31d00b6gnz.jpg" alt="截屏2021-12-21 01.34.27" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxktj7mqutj31q20foadc.jpg" alt="截屏2021-12-21 01.36.39" style="zoom:40%;" />

训练模型的计算复杂度从 ![[公式]](https://www.zhihu.com/equation?tex=O%28V%29) 降至 ![[公式]](https://www.zhihu.com/equation?tex=O%28logV%29) ，这在效率上是一个巨大的提升。而且我们仍然有差不多同样的模型参数（原始模型： ![[公式]](https://www.zhihu.com/equation?tex=V) 个单词的输出向量，分层softmax： ![[公式]](https://www.zhihu.com/equation?tex=V-1) 个隐节点的输出向量)。



#### ⚠️negative sampling

```
Negative Sampling是对于给定的词，生成其负采样词集合的一种策略。
已知有一个词,这个词可以看做一个正例,而它的上下文词集可以看做是负例,但是负例的样本太多。
而在语料库中,各个词出现的频率是不一样的,所以在采样时可以要求高频词选中的概率较大,低频词选中的概率较小,这样就转化为一个带权采样问题,大幅度提高了模型的性能。
```

本质是预测总体类别的一个子集

负采样的思想更加直观：为了解决数量太过庞大的输出向量的更新问题，我们就不更新全部向量，而只更新他们的一些样本。



显然正确的输出单词（也就是正样本）应该出现在我们的样本中，另外，我们需要采集几个单词作为负样本（因此该技术被称为“负采样”）。采样的过程需要指定总体的概率分布，我们可以任意选择一个分布。我们把这个分布叫做噪声分布，标记为 ![[公式]](https://www.zhihu.com/equation?tex=P_%7Bn%7D%28w%29) 。可以凭经验选择一个好的分布。

![[公式]](https://www.zhihu.com/equation?tex=P_%7Bn%7D%28w%29) 中采样得到的单词集合，也就是负样本。![[公式]](https://www.zhihu.com/equation?tex=t_%7Bj%7D)是单词![[公式]](https://www.zhihu.com/equation?tex=w_%7Bj%7D)的标签。t=1时， ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bj%7D)是正样本；t=0时，![[公式]](https://www.zhihu.com/equation?tex=w_%7Bj%7D)为负样本。

只需要将此公式作用于 ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bj%7D%5Cin%5Cleft%5C%7B+w_%7BO%7D+%5Cright%5C%7D%5Ccup+W_%7Bneg%7D) 而不用更新词汇表的所有单词。这也解释了为什么我们可以在一次迭代中节省巨大的计算量。



相关公式：

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxktumsdraj31be0k0772.jpg" alt="截屏2021-12-21 01.47.40" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxktv79628j31bk0jmtcx.jpg" alt="截屏2021-12-21 01.48.13" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxktve8qqfj31bg0hytbr.jpg" alt="截屏2021-12-21 01.48.26" style="zoom:50%;" />



负采样通过考虑相互独立的事件来构造损失函数，这些事件同时涉及正例和负例。训练的计算量与每一步的噪声词数成线性关系。





### Seq2Seq

**简介**

Encoder-Decoder架构

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkandj3v9j31240eqdh7.jpg" alt="截屏2021-12-20 14.43.17" style="zoom:50%;" />

- 编码器是一个RNN，读取句子输入
  - 可以是双向
- 解码器使用另外一个RNN来输入
  - 必须是单向

循环神经网络编码器使用**长度可变的序列作为输入**， 将其转换为固定形状的**隐状态**。

为了连续生成输出序列的词元， 独立的循环神经网络解码器是基于 **输入序列的编码信息** 和 **输出序列已经看见的或者生成的词元** 来预测下一个词元。

**使用循环神经网络编码器最终的隐状态来初始化解码器的隐状态**





**训练**

训练时解码器使用目标句子作为输入。（即使预测错了 仍然拿正确的词放进去，这与推理时不同）



#### BLEU

衡量生成序列的好坏

计算预测中所有 n-gram 的精度。

惩罚过短的预测 + 长匹配有高权重

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkas8d5ycj31100q00vs.jpg" alt="截屏2021-12-20 14.47.59" style="zoom:33%;" />





**总结**

Seq2Seq是从一个句子生成另一个句子

编码器和解码器都是RNN

将编码器最后时间隐状态来初始化解码器隐状态来完成信息传递

常用BLEU来衡量生成序列的好坏





### RNN

循环神经网络的输出取决于当下输入和前一时间的隐变量。

应用到语言模型中时，循环神经网络根据当前词预测下一时刻词。

通常使用困惑度来衡量语言模型的好坏。





**相比于n-gram，为什么要用RNN？**

```
n元语法模型中，一个单词x在时间t步下的条件概率仅取决于前面n-1个单词，如果想利用更多的单词，就需要增加n的大小，那么模型的参数数量也会指数级增长。
如果使用隐状态，那么t-1时刻的隐状态就存储了到时间步t-1的序列信息。那么条件概率就可以写成在t-1时刻隐变量条件下，当前词xt的概率。
```

![截屏2021-12-21 10.50.34](https://tva1.sinaimg.cn/large/008i3skNly1gxl9jhskrjj31ty0imafl.jpg)

当前时间步隐藏变量由当前时间步的输入 与前一个时间步的隐藏变量一起计算得出。



计算逻辑：

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxl9pv7qxzj31dj0u0799.jpg" alt="截屏2021-12-21 10.56.40" style="zoom:50%;" />





损失：

通过一个序列中所有的n个词元的交叉熵损失的平均值来衡量：

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxl9sb80nsj31me0k40w6.jpg" alt="截屏2021-12-21 10.59.01" style="zoom:50%;" />



- 在最好的情况下，模型总是完美地估计标签词元的概率为1。 在这种情况下，模型的困惑度为1。
- 在最坏的情况下，模型总是预测标签词元的概率为0。 在这种情况下，困惑度是正无穷大。
- 在基线上，该模型的预测是词表的所有可用词元上的均匀分布。 在这种情况下，困惑度等于词表中唯一词元的数量。 事实上，如果我们在没有任何压缩的情况下存储序列， 这将是我们能做的最好的编码方式。 因此，这种方式提供了一个重要的上限， 而任何实际模型都必须超越这个上限。





### GRU

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxlde8fxenj319e0psmzx.jpg" alt="截屏2021-12-21 12.58.36" style="zoom:50%;" />

眼熟这两个式子！！！

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gzcf9n2p3oj30s406cq3b.jpg" alt="截屏2022-02-14 01.59.15" style="zoom: 33%;" />

```
GRU有两个门，一个是重置门，一个是更新门。
它们都是由上一时刻的隐状态和当前时刻的输入计算得到的。
当重置门接近于1的时候，t时刻的候选隐状态更接近于普通RNN计算的结果。 当重置门接近于0的时候，t时刻的候选隐状态就相当于X作为输入变量到多层感知机的结果。
更新门的作用就是新的隐状态Ht在多大程度上来自旧的隐状态，和刚才新的候选隐状态。
当更新门接近于1的时候，新的隐状态就倾向于保留旧的隐状态，此时来自输入X的信息基本上就被忽略。
当更新门接近于0的时候，新的隐状态就会接近于候选隐状态。
```



<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxldnv9dlij31cy0ekjvg.jpg" alt="截屏2021-12-21 13.13.10" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxldo3bn5mj31c20dswj4.jpg" alt="截屏2021-12-21 13.13.22" style="zoom:50%;" />



小结：

- 门控循环神经网络可以更好地捕获时间步距离很长的序列上的依赖关系。
- 重置门有助于捕获序列中的短期依赖关系。
- 更新门有助于捕获序列中的长期依赖关系。
- 重置门打开时，门控循环单元包含基本循环神经网络；更新门打开时，门控循环单元可以跳过子序列。





### LSTM

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxldvykp7dj319y0oi775.jpg" alt="截屏2021-12-21 13.19.42" style="zoom:50%;" />

眼熟这两个式子！！！

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gzcfbduuhzj30ho0523yl.jpg" alt="截屏2022-02-14 02.00.56" style="zoom:33%;" />

```
LSTM有一个记忆单元还有三个门，分别是遗忘门，输入门和输出门。
对于候选的记忆单元，和三个门都是由上一时刻的隐状态和当前时刻的输入得到的。
对于记忆单元，输入门 控制采用多少来自于 候选记忆单元的内容，遗忘门就是控制保留多少过去的记忆单元 Ct-1的内容。（引入这种设计是用来缓解梯度消失问题，以及更好地捕获序列中长距离依赖关系）
输出门控制关于隐状态的计算。
输出门接近1，就能将所有记忆信息传递给预测部分；输出门接近0，只保留记忆元内的所有信息，不需要更新隐状态。
```

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxle2dx9i3j31cs08qmz0.jpg" alt="截屏2021-12-21 13.27.06" style="zoom:50%;" />



<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxle2qgycfj31d40b6q5o.jpg" alt="截屏2021-12-21 13.27.27" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxle36cs5nj31cq0a20ur.jpg" alt="截屏2021-12-21 13.27.52" style="zoom:50%;" />



小结：

- 长短期记忆网络有三种类型的门：输入门、遗忘门和输出门。
- 长短期记忆网络的隐藏层输出包括“隐状态”和“记忆元”。只有隐状态会传递到输出层，而记忆元完全属于内部信息。
- 长短期记忆网络可以缓解梯度消失和梯度爆炸。



**为什么有效**

门的动态控制（既能捕获到长期的依赖关系，也能捕获到序列短期的依赖关系）

可以缓解梯度消失和梯度爆炸



**LSTM和GRU的区别和使用场景**

GRU的优点是其模型的简单性 ，因此更适用于构建较大的网络。它只有两个门控，从计算角度看，它的效率更高，它的可扩展性有利于构筑较大的模型；但是LSTM更加的强大和灵活，因为它具有三个门控。LSTM是经过历史检验的方法。



**双向LSTM**

- 在双向循环神经网络中，**每个时间步的隐状态由当前时间步的前后数据同时决定**。
- 双向循环神经网络与概率图模型中的“前向-后向”算法具有相似性。
- 双向循环神经网络主要用于序列编码和给定双向上下文的观测估计。
- 由于**梯度链更长**，因此双向循环神经网络的训练代价非常高。



### ⭐️Transformer

**https://zhuanlan.zhihu.com/p/338817680（Transformer模型详解，建议看）**

https://zhuanlan.zhihu.com/p/82312421（Transformer的剖析）

https://zh-v2.d2l.ai/chapter_attention-mechanisms/transformer.html

```
介绍一下Transformer？
Transformer是一个纯使用注意力的编码-解码器模型。
它的编码、解码器都有多个transformer块。
每个transformer块都使用多头注意力、基于位置的前馈网络、层与层之间使用残差连接和层归一化。

在编码器中，输入嵌入-加上位置编码-多个编码器层。
每个层都有两个子层，一个是多头自注意力，一个是基于位置的前馈网络。
在计算多头自注意力时，query key value来自于上一层的输出，然后使用残差连接，再进行layerNorm

在解码器中，也是由多个层堆叠起来的。但是除了编码器中提到的两个子层，解码器还在两个子层之间插入了第三个子层，成为编码器-解码器注意力层。⭐️它的query来自于前一个 解码器层 的输出，key 和 value来自于 整个编码器 的输出。
然后在解码器的多头自注意力层中，它的query key和value来自于上一层的输出。
但是解码器中的每个位置，只能考虑该位置之前的所有位置（的词元）。这种masked多头注意力确保预测仅依赖于已生成的词元。
```



Transformer是一个纯使用注意力的 编码-解码器

编码器和解码器都有n个 transformer 块（一般为6个）

每个块里使用多头注意力，基于位置的前馈网络，和层归一化。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkdudis5aj30s20yewhe.jpg" alt="截屏2021-12-20 16.33.51" style="zoom:40%;" />

从宏观角度来看，transformer 的编码器是由多个相同的层叠加而成的，每个层都有两个子层。第一个子层是**多头自注意力**（multi-head self-attention）汇聚；第二个子层是**基于位置的前馈网络**（positionwise feed-forward network）。

具体来说，在计算编码器的自注意力时，Q、K和V都来自前一个编码器层的输出。每个子层都采用了**残差连接**（residual connection）,紧接着应用 LayerNorm 。**（Layer Normalization 会将每一层神经元的输入都转成均值方差都一样的，这样可以加快收敛）**

第一个 Encoder block 的输入为句子单词的表示向量矩阵，后续 Encoder block 的输入是前一个 Encoder block 的输出，最后一个 Encoder block 输出的矩阵就是**编码信息矩阵 C**，这一矩阵后续会用到 Decoder 中。

<img src="https://pic3.zhimg.com/80/v2-45db05405cb96248aff98ee07a565baa_1440w.jpg" alt="img" style="zoom:33%;" />



解码器也是由多个相同的层叠加而成的，并且层中使用了残差连接和LayerNorm。除了编码器中描述的两个子层之外，解码器还在这两个子层之间插入了第三个子层，称为***编码器－解码器注意力*（encoder-decoder attention）层**。在编码器－解码器注意力中，**query来自前一个解码器层的输出**，而**K, V**矩阵使用 Encoder 的**编码信息矩阵**。

在解码器自注意力中，Q、K和V都来自上一个解码器层的输出。但是，解码器中的<u>每个位置只能考虑该位置之前的所有位置</u>。

<img src="https://tva1.sinaimg.cn/large/e6c9d24ely1gzrcz1el81j21220gamyd.jpg" alt="截屏2022-02-27 00.03.20" style="zoom:33%;" />



**Transformer 总结**

- Transformer 与 RNN 不同，可以比较好地并行训练。
- Transformer 本身是不能利用单词的顺序信息的，因此需要在输入中添加位置 Embedding，否则 Transformer 就是一个词袋模型了。
- Transformer 的重点是 Self-Attention 结构，其中用到的 **Q, K, V**矩阵通过输出进行线性变换得到。
- Transformer 中 Multi-Head Attention 中有多个 Self-Attention，可以捕获单词之间多种维度上的相关系数 attention score。





#### Transformer常见问题

https://zhuanlan.zhihu.com/p/360144789



**Transformer为什么使用多头自注意力机制？**

（学得多+计算量不大。）

多头自注意力可以基于相同的注意力机制学习到不同的行为，然后把不同的行为作为知识组合起来，捕获序列内各种范围的依赖关系。

多头可以使 参数矩阵 形成多个**子空间**，矩阵整体的size不变，对每个head进行降维，学习更丰富的特征信息，但是计算量和单个head差不多。



**为什么在进行多头注意力的时候需要对每个head进行降维？**

在**不增加时间复杂度**的情况下，同时，借鉴**CNN多核**的思想，在**更低的维度**，在**多个独立的特征空间**，**更容易**学习到更丰富的特征信息。



**Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？**

https://www.zhihu.com/question/319339652

Q和K的点乘是为了得到一个attention score 矩阵，用来对V进行提纯（倾向性选择）。K和Q使用了不同的W_k, W_Q来生成，可以理解为是在**不同空间上的投影**，**增加了表达能力**，这样计算得到的attention score矩阵的**泛化能力更高**。

如果令Q和K相同，attention score 矩阵是一个对称矩阵。因为是同样一个矩阵，都投影到了同样一个空间，所以泛化能力很差。

另外，如果Q和K相同，则两个关注程度相同，但实际上在一个句子中两个词的**相互关注程度**不一定相同。





**为什么在进行softmax之前需要对attention进行scaled（为什么除以dk的平方根），并使用公式推导进行讲解**

https://blog.csdn.net/qq_37430422/article/details/105042303

当 $d_k$ 较大时，向量内积容易取很大的值。两个向量的内积均值为 0，而方差为$d_k$。

**方差较大，不同的 key 与同一个 query 算出的对齐分数可能会相差很大，有的远大于 0，有的则远小于 0.**

**很有可能存在某个 key，其与 query 计算出来的对齐分数远大于其他的 key 与该 query 算出的对齐分数。**

softmax函数对各个内积的偏导数都趋近于0，造成参数更新困难。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz7dvmrra6j30n403mq2y.jpg" alt="截屏2022-02-09 17.23.20" style="zoom:50%;" />





**在计算attention score的时候如何对padding做mask操作？**

对需要mask的位置设为负无穷。

如果padding 都是0，e0=1, 但是softmax的函数，也会导致为padding的值占全局一定概率，mask就是让这部分值取负无穷，让它再softmax之后基本也为0，不影响attention score的分布





**简单介绍一下Transformer的位置编码？有什么意义和优缺点？**

https://zhuanlan.zhihu.com/p/106644634

Transformer使用attention，attention是位置无关的，会丢失词序信息，模型没有办法知道每个词在句子中的相对和绝对的位置信息。无论一个句子内的词序如何，通过attention计算的token的hidden embedding都是一样的。

因此需要主动将词序信息喂给模型。模型原先的输入是不含词序信息的词向量，**位置编码需要将词序信息和词向量结合起来形成一种新的表示输入给模型**，这样模型就具备了学习词序信息的能力。



**简单讲一下Transformer中的残差结构以及意义。**

（详见ResNet标题）

残差连接 (Residual Connection) 用于防止网络退化。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gyp8chdceij317e0s2gon.jpg" alt="截屏2022-01-25 00.31.43" style="zoom: 33%;" />

encoder和decoder的self-attention层和ffn层都有残差连接。反向传播的时候不会造成梯度消失。



**为什么transformer块使用LayerNorm而不是BatchNorm？LayerNorm 在Transformer的位置是哪里？**

<!--batchnorm保留了不同样本间的大小关系，抹去了维度间原有的大小关系； layernorm抹去了样本间的大小关系，保留了维度间间原有的大小关系-->

LayerNorm在多头注意力层和激活函数层之间。

CV使用BN是认为**不同channel**的信息有重要意义，需要保留不同channel之间的大小关系。

NLP领域认为句子长度不一致，并且各个batch的信息没什么关系，因此只考虑句子内信息的归一化。



**简单描述一下Transformer中的前馈神经网络？使用了什么激活函数？相关优缺点？**

前馈神经网络采用了两个线性变换，激活函数为Relu。
$$
F F N(x)=\max \left(0, x W_{1}+b_{1}\right) W_{2}+b_{2}
$$
（优缺点即ReLu函数优缺点）



优点：


- 解决了梯度消失问题

- 计算速度非常快，只需要判断是否大于0

- 收敛速度快于sigmoid和tanh，因为ReLu的收敛速度一直为1

缺点：

- 不是0均值的（zero-centered）

- Dead ReLU Problem，指**某些神经元可能永远不会被激活**，导致相应的参数永远不能被更新。

  产生Dead ReLU的原因：1）参数初始化  2）learning rate太高导致在训练过程中参数更新太大

- 原点不可导。解决办法：坐标轴下降法、最小角回归法



⚠️**Decoder阶段的多头自注意力和encoder的多头自注意力有什么区别？**

在解码器自注意力中，QKV都来自上一个解码器层的输出。但是，解码器中的<u>每个位置只能考虑该位置之前的所有位置</u>。这种*掩蔽*（masked）注意力保留了*自回归*（auto-regressive）属性，<u>确保预测仅依赖于已生成的输出词元</u>。



**Transformer的并行化提现在哪个地方？Decoder端可以做并行化吗？**

Transformer的并行化主要体现在self-attention模块，在Encoder端Transformer可以并行处理整个序列，并得到整个输入序列经过Encoder端的输出。

Decoder在训练的时候可以，但是交互的时候不可以。

> Decoder 可以在训练的过程中使用 Teacher Forcing 并且并行化训练，即将正确的单词序列 (\<Begin> I have a cat) 和对应输出 (I have a cat \<end>) 传递到 Decoder。那么在预测第 i 个输出时，就要将第 i+1 之后的单词掩盖住，**注意 Mask 操作是在 Self-Attention 的 Softmax 之前使用的，下面用 0 1 2 3 4 5 分别表示 "\<Begin> I have a cat \<end>"。**



**Bert的mask为何不学习transformer在attention处进行屏蔽score的技巧？**

BERT和transformer的目标不一致，bert是语言的预训练模型，需要充分考虑上下文的关系，而transformer主要考虑句子中第i个元素与前i-1个元素的关系。



❌**简单描述一下wordpiece model 和 byte pair encoding，有实际应用过吗？**

不知道







⚠️**Transformer和传统seq-seq的区别，seq-seq的翻译机制和语言模型的区别**







### ⭐️Bert

https://zh-v2.d2l.ai/chapter_natural-language-processing-pretraining/bert.html

https://zhuanlan.zhihu.com/p/46652512

```
介绍一下bert？
bert是一个基于微调的预训练模型，它抽取了足够多的信息。这些抽取的特征可以复用，可以挪到别的地方去。在面对新的任务时，只需要在后面加一个简单的输出层。

bert的输入 是 单个文本或单个文本对。
当输入为单个文本的时候，前面加cls，然后是文本序列的标记、以及特殊分隔词元sep
当输入为单个文本对的时候，前面是cls，然后是第一个文本序列的标记，加一个分隔词元sep，第二个文本序列标记，以及一个特殊分隔词元sep

输入有三个embedding
1. token embedding: 要将各个词转换成固定维度的向量
2. segment embedding: 使用分隔词进行拼接，然后对第一个句子id为0，第二个id为1
3. position embedding: 让BERT在各个位置上学习一个向量表示，将序列顺序的信息编码进来。

bert的输出就是 对每一个词元 返回一个抽取了上下文信息的特征向量。

特点：上下文敏感，可以用于不可知任务。

预训练任务1: 带掩码的语言模型
Transformer编码器是双向的
每次随机（15%）将一些词元替换成<mask>。每次去预测一下是什么。
在选择mask的15%的词当中
	1) 80%的概率下，将选中的词元变成<mask>
	2) 10%的概率下，替换成随机的词元
	3) 10%的概率下，保留原有的词元

预训练任务2: 下一句子预测
预测一个句子对中的两个句子是不是相邻
训练样本中
	50%的概率选择相邻句子对
	50%的概率选择随机句子对
将<cls>对应的输出放到一个全连接层进行预测，是不是相邻

```

**简介**

动机：

- 基于微调的NLP模型
- 预训练的模型抽取了足够多的信息
- 新的任务只需要增加一个简单的输出层



Bert就是，预训练模型抽取了足够多的信息，足够抓住语义信息。这些特征抽取的权重可以复用的，可以挪到别的任务去。新的任务只需要增加简单的输出层就行了。不需要再加什么RNN，Transformer等网络。

**只要把特征转换成语义label的空间就行了。**



**输入输出？**

BERT输入序列明确地表示单个文本和文本对。

当输入为单个文本时，BERT输入序列是特殊类别词元“\<cls>”、文本序列的标记、以及特殊分隔词元“\<sep>”的连结。

当输入为文本对时，BERT输入序列是“\<cls>”、第一个文本序列的标记、“\<sep>”、第二个文本序列标记、以及“\<sep>”的连结。



三个embedding： **Token Enbedding, Segment Embedding, Position embedding**

- 要将各个词转换成固定维度的向量。在BERT中，每个词会被转换成768维的向量表示。
- Segment Embedding 用来区别两种句子。 给第一个句子 id为0，第二个句子id是1
- 由于出现在文本不同位置的字/词所携带的语义信息存在差异，因此，BERT 模型对不同位置的字/词分别附加一个不同的向量以作区分。**Position Embeddings和之前文章中的Transformer不一样，不是三角函数而是学习出来的**

（与Transformer本身的Encoder端相比，BERT的Transformer Encoder端输入的向量表示，多了Segment Embeddings。）



模型输入：通过查询字向量表将文本中的每个字转换为一维向量

模型输出：每个位置都是融合全文语义信息后的向量表示。（抽取了上下文信息的特征向量）



**特点：**

1. 从上下文无关到上下文敏感
2. 从特定任务到不可知任务

ELMo对上下文进行双向编码，但使用特定于任务的架构；而GPT是任务无关的，但是从左到右编码上下文。

BERT（来自Transformers的双向编码器表示）结合了这两个方面的优点。它对上下文进行双向编码，并且对于大多数的NLP任务只需要最少的架构改变。通过使用预训练的Transformer编码器，BERT能够基于其双向上下文表示任何词元。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz9zxvsjhgj315g0acdhy.jpg" alt="截屏2022-02-11 23.37.50" style="zoom:50%;" />

缺点：

- [MASK]标记在实际预测中不会出现，训练时用过多[MASK]影响模型表现
- 每个batch只有15%的token被预测，所以BERT收敛得比left-to-right模型要慢（它们会预测每个token）





**架构：**

- 只有编码器的Transformer
- 两个版本
  - Base: #blocks = 12, hidden size = 768, #heads = 12, #parameters = 110M
  - Large: #blocks = 24, hidden size = 1024, #heads = 16, #parameters = 340M
- 在大规模数据上训练 > 3B 词



Bert maxposition：512

谷歌开源的预训练模型，那么这个词表的大小将会被限制在512





⚡️<font color=red>**Bert的参数量计算，一层一层说**</font>

（2022.02.28 字节电商NLP二面）

https://zhuanlan.zhihu.com/p/144582114







#### 预训练任务1：带掩码的语言模型

- Transformer的编码器是双向的，Bert是深层，多层的网络结构会暴露预测词，失去学习的意义。
- 每次随机（15%）将一些词元换成\<mask>。每次去预测一下是什么
- 因为微调任务中没有\<mask>【如果一直用标记[MASK]代替（在实际预测时是碰不到这个标记的）会影响模型】
  - 80%概率下，将选中的词元变成\<mask>
  - 10%概率换成一个随机词元
  - 10%概率保持原有的词元

**最终的损失函数只计算被mask掉那个token。**



#### 预训练任务2：下一句子预测

- 预测一个句子对中两个句子是不是相邻。（引入这个任务可以更好地让模型学到连续的文本片段之间的关系）
- 训练样本中
  - 50%概率选择相邻句子对
  - 50%概率选择随机句子对
- **将\<cls>对应的输出放到一个全连接层来预测**（是不是相邻）



#### 预训练总结

- BERT针对微调设计
- 相比于Transformer：模型更大，训练数据更多；输入句子对，片段嵌入，可学习的位置编码；训练时使用两个任务 带掩码的语言模型、下一个句子预测。





#### 微调

https://www.bilibili.com/video/BV15L4y1v7ts?spm_id_from=333.999.0.0

Bert 对每一个词元返回 **抽取了上下文信息的特征向量**。（特征维度大小就是hidden size）

不同的任务会使用不同的特征。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkekqny7gj30l60ekwf2.jpg" alt="截屏2021-12-20 16.59.10" style="zoom:50%;" />



例子：句子分类

将\<cls>对应的向量输入到全连接层进行分类

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkelwgm71j317m0fqjsl.jpg" alt="截屏2021-12-20 17.00.18" style="zoom:50%;" />

为什么？

1. 预训练时判断两个句子是不是一个pair，用的就是cls。
2. 这个无明显语义信息的符号会更“公平”地融合文本中各个字/词的语义信息。



例子：命名实体识别

识别一个词是不是命名实体，例如人名、机构、位置。

将非特殊词（除去了cls，sep）放进全连接层分类。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkepn9mb1j30ia0dyjrq.jpg" alt="截屏2021-12-20 17.03.54" style="zoom:50%;" />



例子：问题问答

给定一个问题，和描述文字，找出一个片段作为回答

对片段中的每个词元预测它是不是回答的开头或结束

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkeqi6robj30pi0f40tb.jpg" alt="截屏2021-12-20 17.04.44" style="zoom:50%;" />





#### 微调总结

即使下游任务各有不同，使用Bert微调时均只需要增加输出层。

但根据任务的不同，输入的表示，和使用的Bert特征也会不一样。



#### ⚠️Embedding





#### ⚠️优化器

不是标准的Adam？



#### Bert常见问题

https://mp.weixin.qq.com/s/TDoK3xdXZ6xIZsw9SFxoHQ

**Bert为什么只使用了Transformer的Encoder？**

那么Decoder去哪了呢？显然是被BERT改造了。Transformer其实是个完整地seq-to-seq模型，可以解决诸如机器翻译、生成式QA这种输入输出为不定长句子的任务，在Transformer中，它使用Encoder做特征提取器，然后用Decoder做解析，输出我们想要的结果。

而对于BERT，它作为一个预训练模型，它使用固定的任务——language modeling来对整个模型的参数进行训练，这个language modeling的任务就是masked language model，所以它是一个用上下文去推测中心词[MASK]的任务，故和Encoder-Decoder架构无关，它的输入输出不是句子，其输入是这句话的上下文单词，输出是[MASK]的softmax后的结果，最终计算Negative Log Likelihood Loss，并在一次次迭代中以此更新参数。

**所以说，BERT的预训练过程，其实就是将Transformer的Decoder拿掉，仅使用Encoder做特征抽取器，再使用抽取得到的“特征”做Masked language modeling的任务，通过这个任务进行参数的修正。**

当然了，BERT不仅仅做了MLM任务，还有Next Sequence Prediction，这个由于后序验证对模型的效果提升不明显，所以没有赘述。

注意：我们常说，xxx使用Transformer作为特征抽取器，这其实在说用Transformer的Encoder(主要是Self-Attention和短路连接等模块)做特征抽取器，和Decoder啥关系也没有



**BERT 的输入和输出分别是什么？**

BERT 模型的主要输入是文本中各个字/词(或者称为 token)的原始词向量，该向量既可以随机初始化，也可以利用 word2vec 等算法进行预训练以作为初始值；输出是文本中各个字/词融合了全文语义信息后的向量表示。

模型输入除了字向量(英文中对应的是 Token Embeddings)，还包含另外两个部分：

1. 文本向量(英文中对应的是 Segment Embeddings)：该向量的取值在模型训练过程中自动学习，用于刻画文本的全局语义信息，并与单字/词的语义信息相融合
2. 位置向量(英文中对应的是 Position Embeddings)：由于出现在文本不同位置的字/词所携带的语义信息存在差异（比如：“我爱你”和“你爱我”），因此，BERT 模型对不同位置的字/词分别附加一个不同的向量以作区分

最后，BERT 模型将字向量、文本向量和位置向量的加和作为模型输入。



⚠️**BERT 的三个 Embedding 直接相加会对语义有影响吗？**

https://www.zhihu.com/question/374835153

Embedding 的数学本质，就是以 one hot 为输入的单层全连接。

现在我们将 token, position, segment 三者都用 one hot 表示，然后 concat 起来，然后才去过一个单层全连接，等价的效果就是三个 Embedding 相加。

因此，BERT 的三个 Embedding 相加，其实可以理解为 token, position, segment 三个用 one hot 表示的特征的 concat。Embedding 就是以 one hot 为输入的单层全连接。



**BERT 的MASK方式的优缺点？**

BERT的mask方式：在选择mask的15%的词当中，80%情况下使用mask掉这个词，10%情况下采用一个任意词替换，剩余10%情况下保持原词汇不变。

优点：

1. 被随机选择15%的词当中以10%的概率用任意词替换去预测正确的词，相当于文本纠错任务，为BERT模型赋予了一定的文本纠错能力；
2. 被随机选择15%的词当中以10%的概率保持不变，缓解了finetune时候与预训练时候输入不匹配的问题（预训练时候输入句子当中有mask，而finetune时候输入是完整无缺的句子，即为输入不匹配问题）。

缺点：

**针对有两个及两个以上连续字组成的词，随机mask字割裂了连续字之间的相关性，使模型不太容易学习到词的语义信息。**



**BERT深度双向的特点，双向体现在哪儿？**

BERT的预训练模型中，预训练任务是一个mask LM ，通过随机的把句子中的单词替换成mask标签， 然后对单词进行预测。

这里注意到，对于模型，输入的是一个被挖了空的句子， 而由于Transformer的特性（Self-attention机制）， 它是会注意到所有的单词的，这就导致模型会根据挖空的上下文来进行预测， 这就实现了双向表示， 说明BERT是一个双向的语言模型。



**BERT深度双向的特点，深度体现在哪儿？**

针对特征提取器，Transformer只用了self-attention，没有使用RNN、CNN，并且**使用了残差连接有效防止了梯度消失的问题**，使之可以构建更深层的网络，所以BERT构建了多层深度Transformer来提高模型性能。

模型中的ADD即为残差相加，思路与CV的残差网络相同，使用残差可以使得模型做的很深。



**BERT中并行计算体现在哪儿？**

不同于RNN计算当前词的特征要依赖于前文计算，有时序这个概念，是按照时序计算的，而BERT的Transformer-encoder中的self-attention计算当前词的特征时候，没有时序这个概念，是同时利用上下文信息来计算的，一句话的token特征是通过矩阵并行‘瞬间’完成运算的，故，并行就体现在self-attention。



**BERT中Transformer中的Q、K、V存在的意义？**

在使用self-attention通过上下文词语计算当前词特征的时候，X先通过$W_Q、W_K、W_V$线性变换为QKV，然后使用QK计算得分，最后与V计算加权和而得。

倘若不变换为QKV，直接使用每个token的向量表示点积计算重要性得分，那在softmax后的加权平均中，该词本身所占的比重将会是最大的，使得其他词的比重很少，无法有效利用上下文信息来增强当前词的语义表示。而变换为QKV再进行计算，能有效利用上下文信息，很大程度上减轻上述的影响。

**BERT中的注意力计算方式是点乘**

<img src="https://mmbiz.qpic.cn/mmbiz_png/HIKk7OFDjoTOKcjqp0Dh8BlibvG8WbJehicJjfN7NSoUrib3ibTcDnC8vZ9Dfp8qCE6Ciciau4ts6iaex9JjzI4yByl0g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:50%;" />



❌**BERT中Transformer中Self-attention后为什么要加前馈网络？**

<u>由于self-attention中的计算都是线性了，为了提高模型的非线性拟合能力，需要在其后接上前馈网络？？？？？？</u>（存疑⚠️）

<u>前馈网络不也是线性的吗？？另外attention不是有softmax吗咋是线性的？？</u>

（前馈网络是两个线性变换+一个relu）
$$
F F N(x)=\max \left(0, x W_{1}+b_{1}\right) W_{2}+b_{2}
$$



**BERT的优缺点**

优点：

- 并行
- 解决长时依赖
- 双向特征表示，特征提取能力强，有效捕获上下文的全局信息
- 缓解梯度消失的问题等，BERT擅长解决的NLU任务。

缺点：

- 生成任务表现不佳：预训练过程和生成过程的不一致，导致在生成任务上效果不佳；
- 采取独立性假设：没有考虑预测[MASK]之间的相关性，是对语言模型联合概率的有偏估计（不是密度估计）；
- 输入噪声[MASK]，造成预训练-微调两阶段之间的差异；
- 无法文档级别的NLP任务，只适合于句子和段落级别的任务；





### 基于Bert的改进模型

https://zhuanlan.zhihu.com/p/444588460

https://blog.csdn.net/qq_38556984/article/details/107501414

（albert，roberta， electra做了什么改进）



#### RoBERTa

RoBERTa并没有去更改BERT的网络结构，只是修改了一些预训练时的方法，其中包括：动态掩码（Dynamic Masking）、舍弃NSP（Next Sentence Predict）任务。



**动态掩码技术（Dynamic Masking）**

原始BERT中，训练数据集中的[MASK]位置是预先就生成好的，但RoBERTa中，**[MASK]的位置是在模型训练的时候实时计算决定的**。这样的好处在于，无论你训练多少个Epoch，都能**极大程度上避免训练数据的重复**，如果只使用一份静态数据的话，每个Epoch训练的数据都是一样的（[MASK]的位置也是一样的）。



**舍弃了NSP任务**

原始BERT的预训练过程中，有一个阶段会使用「文本对」作为输入，让模型判断这两个文本段是否为前后文关系，该任务被称为NSP（Next Sentence Predict）任务。但是RoBERTa的实验结论却指出，在预训练过程中去掉这个NSP任务效果反而会更好。

> 之前在网上有看到过一个听起来比较合理的解释：BERT在选择NSP样本时，正样本是同一篇文章中的连续段落；负样本是不同文章的两个段落。这样就会造成模型不仅会关注句子的连通性，同样还会关注句子的主题（topic）。而不同文章下的句子大部分描述的主题都是相差很大的，因此模型很容易更倾向于依赖主题来完成任务，而没有真正的学习到句子之间的**连通性**。

训练序列更长。RoBERTa去除了NSP，而是每次输入连续的多个句子，直到最大长度512（可以跨文章）



**使用更大的数据集，更长的训练步数**

RoBERTa使用数据集由BERT的16G扩大到了160G，扩大的10倍。





#### ALBERT

ALBERT认为BERT模型的参数量实在太大（一个中文Bert模型有411M），这在存储&训练上都比较消耗资源。因此，ALBERT提出利用「词向量因式分解」、「跨层参数共享」以及「句子顺序预测任务」来减小模型大小并提升模型训练速度。



**词向量因式分解（Factorized Embedding Parameterization）**

在BERT当中**输入层每一个token的embedding层和transformer的hidden_layer是直接相连**的，这就导致**embedded_size(E)就必须等于hidden_size(H)**。

那么要想**从输入token的size**（等于词表的vocab_size，V）**映射到隐层H**，那么我们就要建立一个映射矩阵：

$(1, V) · (V, H) = (1, H)$

那么中间这个（V，H）的矩阵就是我们需要构建的矩阵，矩阵运算的时间复杂度是O(VH)。

但是今天往往我们的词表V会非常大（毕竟要尽可能多的囊括所有可能出现的字符，中文BERT模型的vocab大小是21128），所以就会造成O(VH)的值特别的大。

ALBERT的作者认为，BERT之所以这么强大，主要是因为**隐层**能够学到**「上下文」**之间的关系，且**本身词向量矩阵就是比较稀疏的**（因为输入的是one-hot，只有对应词索引的部分才会被激活），所以**word embedding的维度并不用和transformer的hidden要一样大**，H应该是要远大于（>>）E的。

因此，作者所说的因式分解就是在这里做了一个分解：让word embedding不直接和hidden layer相连，而是**先将词表（V）压缩到一个较小的Embedding维度（E），再从较小的Embedding（E）连接到hidden层。**

<img src="https://tva1.sinaimg.cn/large/e6c9d24ely1gzpkbzu6ulj212u0swgnd.jpg" alt="截屏2022-02-25 10.46.48" style="zoom: 33%;" />

这样一来，我们就需要构建两个小矩阵，并进行前后两次矩阵运算 O(V, E) + O(E, H)。那从今天来看O(V, H)会是远远大于O(V, E) + O(E, H)的。



**跨层参数共享（Cross-layer parameter sharing）**

原始BERT中，每一层Transformer的参数是不共享的，而在ALBERT中所有Transformer Layer的参数都是一样的



**句子顺序预测任务（Sentence Order Prediction，SOP）**

我们在RoBERTa任务中有提到过，将原始BERT任务中的NSP任务去掉反而会有更好的效果。当时我们分析可能是因为从不同的文章中选择的句子，其话题（topic）相差甚远，因此模型并不会学习到句子之间的「连贯性」，而是依赖「话题」进行区分。

因此，在ALBERT中提出了一种新的「句子关联性学习」的任务：顺序预测。

顺序预测任务中，Positive Sample是连续的两段文本，Negative Sample是这两段文本调换顺序，最后让模型学习句子顺序是否正确。

通过调整正负样本的获取方式去除主题识别的影响，使预训练更关注于句子关系一致性预测。





#### Electra

https://zhuanlan.zhihu.com/p/89763176

ELECTRA最主要的贡献是提出了新的预训练任务和框架，把生成式的Masked language model(MLM)预训练任务改成了判别式的Replaced token detection(RTD)任务，判断当前token是否被语言模型替换过。

<img src="https://tva1.sinaimg.cn/large/e6c9d24ely1gzpkjtc3izj215k0bodh8.jpg" alt="截屏2022-02-25 10.54.20" style="zoom:50%;" />





#### ⚠️XLNet






### ⚠️GPT





### ResNet

核心思想：每个附加层都应该更容易地包含原始函数作为其元素之一。

f(x) = x + g(x)

设计残差网络的目的是为了避免梯度消失。

> 原论文讲过梯度爆炸和梯度消失在引入bn层之后基本解决，残差是为了解决网络退化





残差块结构：

残差块里首先有2个有输出通道数相同的3×3卷积层。 每个卷积层后接一个batchNorm层和ReLU激活函数。 然后我们通过跨层数据通路，跳过这2个卷积运算，将输入直接加在最后的ReLU激活函数前。 

这样的设计要求2个卷积层的输出与输入形状一样，从而使它们可以相加。 如果想改变通道数，就需要引入一个额外的1×1卷积层来将输入变换成需要的形状后再做相加运算。



<img src="https://tva1.sinaimg.cn/large/008i3skNly1gyp8chdceij317e0s2gon.jpg" alt="截屏2022-01-25 00.31.43" style="zoom:50%;" />

利用残差块（residual blocks）可以训练出一个有效的深层神经网络：输入可以通过层间的残差连接更快地向前传播。





### ⚠️TextCNN

https://zhuanlan.zhihu.com/p/129808195



### SVM

与分类平面距离最近的样本点 称为 **支持向量**，进而构成支持平面。

分类器的分类间距*ρ*指的是支持平面之间的距离

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxlhjsen9mj31dc0u0aco.jpg" alt="截屏2021-12-21 15.27.37" style="zoom: 33%;" />

**支持向量机的核心思想:最大化分类间距*ρ***

分类间隔*ρ*是样本点到分类平面的最小几何距离的两倍:
$$
\rho=2 \min _{i} \frac{\left|\mathbf{w}^{\top} \mathbf{x}_{i}+b\right|}{|| \mathbf{w}||}
$$
因此，支持向量的目标可以形式化定义为

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxlhodnmf5j30rs09kjrt.jpg" alt="截屏2021-12-21 15.32.03" style="zoom:33%;" />

其含义是:**在确保分类正确(使用函数距离)的前提条件下，最大化分类间隔(使用几何距离)**



**松弛变量**

在实际应用中，存在着大量线性不可分的数据。

引入松弛变量，容忍部分不可分数据。





**核函数**

如果线性不可分情况非常严重，需要进一步进行空间映射，将低维空间的线性不可分问题转为高维空间的线性可分问题。



⚠️**SVM的理论依据，如何推导？**







### ⚠️隐马尔可夫

#### Viterbi算法

维特比算法(Viterbi Algorithm)是一种**动态规划算法**。

Viterbi算法是用来解决：

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gy34nic4x5j317006iabj.jpg" alt="截屏2022-01-05 21.41.35" style="zoom:33%;" />

就是说，我知道 隐状态初始化概率、隐状态转移概率、观测状态生成概率，也知道当前的观测序列。来推测这几天最可能是什么天气。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gy34mvevvyj318g0pkgoj.jpg" alt="截屏2022-01-05 21.40.58" style="zoom:33%;" />

<font color=red>从图的角度来看，计算最优隐状态序列概率等价于计算最大路径。</font>

```
比如，要求某一节点的最长路径。可以通过这个节点的所有前继节点计算。
对于任意一个前继节点，前继节点的最长路径 + 前继节点到该节点的路径长度。选出一个最大值，那么这个最大值就是当前节点的最长路径。对应的前继节点就是最优前继。

类比一下，t时刻某一节点的隐状态序列概率，就是 t-1时刻节点的序列概率 * 隐状态转移概率 * 观测状态生成概率，选一个最大值，作为该节点的隐状态序列概率。
```


$$
P(\mathbf{x} ; \boldsymbol{\theta})=\operatorname{argmax}_{\mathbf{z}}\{P(\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta})\}
$$
<img src="https://tva1.sinaimg.cn/large/008i3skNly1gy34qjgt9ej30ug0i4abq.jpg" alt="截屏2022-01-05 21.44.31" style="zoom:33%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gy34rgp4f8j318i0b0q5s.jpg" alt="截屏2022-01-05 21.45.23" style="zoom:33%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gy34rrx23rj318m0r2gov.jpg" alt="截屏2022-01-05 21.45.41" style="zoom:33%;" />

**就相当于把前向概率的求和换成了最大值！**

<!--因为乘的 b是一样的，所以max的作用范围只是大括号内部就行了-->

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gy34sdtcdzj318a0qq0w3.jpg" alt="截屏2022-01-05 21.46.17" style="zoom:33%;" />





你前向概率都会算，那么Viterbi肯定也会算

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gy34sm8amuj31860owgpu.jpg" alt="截屏2022-01-05 21.46.30" style="zoom:33%;" />





### ⚠️条件随机场



### ⚠️朴素贝叶斯







## 大类

### ⭐️Attention

<!--（下面的随意是指 follow your heart，而不是随便）-->

<!--心理学认为人通过随意线索（自主性）和不随意线索（不自主的）选择注意点。-->

<!--参考人类注意力的方式。非自主性提示下，选择偏向于感官输入。比如卷积、全连接层、池化层都只考虑不自主性的线索。-->

<!--**注意力机制则显式地考虑随意性（有自主）的线索。**-->



**注意力机制通过 注意力汇聚 将*查询*（自主性提示）和*键*（非自主性提示）结合在一起，实现对*值*（感官输入）的选择倾向**



- 随意线索被称为查询 query
- 每个输入是一个 key-value对。key表示不随意性的线索，value表示它的值。
- 通过**注意力汇聚（attention pooling）**来有偏向性的选择某些输入。



<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxka3l8jpwj30la0cc3z3.jpg" alt="截屏2021-12-20 14.24.17" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkb8z7cmlj30wo0lytak.jpg" alt="截屏2021-12-20 15.04.02" style="zoom: 33%;" />

#### 评分函数

**非参注意力池化层**





<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkaa8n8w0j30wa0h8q4r.jpg" alt="截屏2021-12-20 14.30.43" style="zoom: 33%;" />

其中 K 叫做核函数。衡量 x 和 xi 距离的函数。

如果一个键 xi越是接近给定的查询 x， 那么分配给这个键对应值 yi的注意力权重就会越大， 也就“获得了更多的注意力”。



高斯核 + softmax

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkadi5fxyj30t00jaq46.jpg" alt="截屏2021-12-20 14.33.50" style="zoom:33%;" />



**参数化的注意力机制**

在之前的基础上加上一个可学习的w

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkaf482rij30vu0di0tn.jpg" alt="截屏2021-12-20 14.35.24" style="zoom:33%;" />



<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkaj9kf0dj314o04i3yz.jpg" alt="截屏2021-12-20 14.39.20" style="zoom:33%;" />





**加性注意力**

当查询和键是不同长度的矢量时， 我们可以使用加性注意力作为评分函数。 
$$
a(\mathbf{q}, \mathbf{k})=\mathbf{w}_{v}^{\top} \tanh \left(\mathbf{W}_{q} \mathbf{q}+\mathbf{W}_{k} \mathbf{k}\right) \in \mathbb{R}
$$
将查询和键连结起来后输入到一个多层感知机（MLP）中， 感知机包含一个隐藏层，其隐藏单元数是一个超参数h。 通过使用tanh作为激活函数，并且禁用偏置项。



**缩放点注意力**

点积操作要求查询和键具有相同的长度
$$
a(\mathbf{q}, \mathbf{k})=\mathbf{q}^{\top} \mathbf{k} / \sqrt{d}
$$
<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkbfadektj31vg0aiach.jpg" alt="截屏2021-12-20 15.10.09" style="zoom:50%;" />



#### ⭐️多头注意力

https://zh-v2.d2l.ai/chapter_attention-mechanisms/multihead-attention.html

当**给定相同的查询、键和值的集合**时， 我们希望模型可以**基于相同的注意力机制学习到不同的行为**， 然后将不同的行为作为知识组合起来， **捕获序列内各种范围的依赖关系** （例如，短距离依赖和长距离依赖关系）。 

（即，对同一key，value，query，希望抽取不同的信息； 多头注意力使用 h 个独立的 attention pooling）



用独立学习得到的 h 组不同的 ***线性投影*（linear projections）来变换查询、键和值**。 然后，这 h 组**变换后的查询、键和值将并行地送到注意力汇聚**中。 最后，将这 h个注意力汇聚的**输出拼接在一起**， 并且**通过另一个可以学习的线性投影进行变换， 以产生最终输出**。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkbjtn1r3j30ws0jmjsy.jpg" alt="截屏2021-12-20 15.14.31" style="zoom:50%;" />



<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkbln7ewmj31hk0ii0we.jpg" alt="截屏2021-12-20 15.16.13" style="zoom:50%;" />

基于这种设计，每个头都可能会关注输入的不同部分， 可以表示比简单加权平均值更复杂的函数。







#### 自注意力

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkcpbx3orj31by06y0tk.jpg" alt="截屏2021-12-20 15.54.24" style="zoom:50%;" />



**对于输入序列中的每一个词，都作为一个query，key，value。来对序列 抽取特征 得到输出序列。**

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkcr44qi9j31ak0mgju7.jpg" alt="image-20211220155610133" style="zoom: 33%;" />



#### ⚠️位置编码





### Boosting

https://zhuanlan.zhihu.com/p/280222403

目的：将多个弱一点的模型，组合在一起变成一个比较强的模型。用来降低**偏差**。

（个体学习器间存在强依赖关系、必须串行生成的序列化方法）



**Boosting 的工作机制**

1. 先从初始训练集训练出一个weak learner
2. 根据weak learner的表现对训练样本分布进行调整，使得先前的learner做错的训练样本在后续受到更多关注
3. 基于调整后的样本分布来训练下一个weak learner。（2和3是AdaBoost的机制。GB的机制是训练weak learner预测残差）
4. 如此反复进行，直至达到指定的数目
5. 将T个weak learner进行加权结合。



<img src="https://tva1.sinaimg.cn/large/008i3skNly1gyp9o089s5j31x90u00xn.jpg" alt="截屏2022-01-25 01.17.28" style="zoom: 33%;" />



#### GBDT

**Gradient Boosting**

在初始时，在原始的样本上进行训练，在之后的时间，把当前时刻boosting出来的模型 **计算残差**（真实的yi减去预测值）再重新去拟合。把新学到的模型，通过一个学习率，加到已有的boosting模型。
$$
H_{t+1}(x) = H_{t}(x)+ \eta f_{t}(x)
$$


<img src="https://tva1.sinaimg.cn/large/008i3skNly1gyp9x1kfwjj312o0l0n02.jpg" alt="截屏2022-01-25 01.26.09" style="zoom: 50%;" />



GBDT 就是用决策树作为weak learner（浅层，防止过拟合）

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz4a6ezjdrj31cg0r041r.jpg" alt="截屏2022-02-07 00.58.38" style="zoom:50%;" />

**GBDT的[例子](https://zhuanlan.zhihu.com/p/280222403)**（给的例1好好看看）

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz4aht55ozj31960pwgp3.jpg" alt="截屏2022-02-07 01.09.37" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz4aikqwijj31dm0u0jw9.jpg" alt="截屏2022-02-07 01.10.20" style="zoom: 50%;" />



#### XGBoost

详见https://zhuanlan.zhihu.com/p/162001079！！！

XGBoost（eXtreme Gradient Boosting）极致梯度提升，是基于GBDT的一种算法。

**XGBoost 相比于 GBDT 的优化**

- 利用二阶泰勒公式展开
  - 优化损失函数，提高计算精确度
- 利用正则项
  - 简化模型，避免过拟合
- 采用Blocks存储结构
  - 可以并行计算等



XGBoost的目标函数由损失函数和正则化项两部分组成。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz4b8ie1lqj319m0mi40t.jpg" alt="截屏2022-02-07 01.35.16" style="zoom:50%;" />

用GBDT梯度提升树表达方式XGBoost。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz4b8ytw87j31aa0d075e.jpg" alt="截屏2022-02-07 01.35.42" style="zoom:50%;" />

接下来，三个步骤优化XGBoost目标函数。

> 第一步：二阶泰勒展开，去除**常数项**，优化损失函数项；
>
> 第二步：正则化项展开，去除**常数项**，优化正则化项；
>
> 第三步：合并**一次项系数、二次项系数**，得到最终目标函数。

最终得到目标函数

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz4bn72praj31cs0km76z.jpg" alt="截屏2022-02-07 01.49.22" style="zoom: 40%;" />



**XGBoost的特点，缺失值如何处理**

https://zhuanlan.zhihu.com/p/382253128

https://zhuanlan.zhihu.com/p/269193235

https://mp.weixin.qq.com/s/a4v9n_hUgxNyKSQ3RgDMLA

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz4bxvxh3dj313q0io0vs.jpg" alt="截屏2022-02-07 01.59.39" style="zoom:40%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz4by7009sj314m0cugn8.jpg" alt="截屏2022-02-07 01.59.57" style="zoom:40%;" />

说法二：

- 在特征k上寻找最佳 split point 时，不会对该列特征 missing 的样本进行遍历，而只对该列特征值为 non-missing 的样本上对应的特征值进行遍历，通过这个技巧来减少了为稀疏离散特征寻找 split point 的时间开销。
- 在逻辑实现上，为了保证完备性，会将该特征值missing的样本分别分配到左叶子结点和右叶子结点，两种情形都计算一遍后，选择分裂后增益最大的那个方向（左分支或是右分支），作为预测时特征值缺失样本的默认分支方向。
- 如果在训练中没有缺失值而在预测中出现缺失，那么会自动将缺失值的划分方向放到右子结点。





**XGBoost是怎么预防过拟合的？**

XGBoost在设计时，为了防止过拟合做了很多优化，具体如下：

- **目标函数添加正则项**：叶子结点个数+叶子节点权重的L2正则化
- **列抽样**：训练的时候只用一部分特征（不考虑剩余的block块即可）
- **子采样**：每轮计算可以不使用全部样本，使算法更加保守
- **shrinkage**: 可以叫学习率或步长，为了给后面的训练留出更多的学习空间





**和LightGBM的差别？**

https://mp.weixin.qq.com/s/a4v9n_hUgxNyKSQ3RgDMLA



#### ⚠️LightGBM





### ⚠️图神经网络



### ⚠️知识图谱

以**结构化三元组**的形式存储现实世界中的实体及其关系，三元组通常描述了一个特定领域中的事实，由头实体、尾实体和描述这两个实体之间的关系组成。

关系有时也称为属性，尾实体被称为属性值。

从图结构的角度看，实体是知识图谱中的节点，关系是连接 两个节点的有向边。



## 一般提问

https://blog.csdn.net/GreatXiang888/article/details/99296607



### 过拟合解决方式

1. 获取更多数据（源头获取、数据增强）。让模型「看见」尽可能多的「例外情况」，它就会不断修正自己，从而得到更好的结果。
2. 选择合适的模型。
   - 减少网络的层数、神经元个数等均可以限制网络的拟合能力
   - 训练时间 Early stopping。在初始化网络的时候一般都是初始为较小的权值。训练时间越长，部分网络权值可能越大。如果我们在合适时间停止训练，就可以将网络的能力限制在一定范围内。
   - 限制权值 Weight-decay，也叫正则化（regularization）。直接将权值的大小加入到 Cost 里
   - 增加噪声 Noise。在输入中、权值中
3. Dropout（详见下一条）



### Dropout

在每个训练批次中，在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为**它不会太依赖某些局部的特征**。

一小批训练样本执行完这个过程后，在没有被删除的神经元上按照随机梯度下降法更新对应的参数（w，b）。



**实现思路：（在训练和测试过程的区别）**

Dropout 在训练时采用，是为了减少神经元对部分上层神经元的依赖，**类似将多个不同网络结构的模型集成起来**，减少过拟合的风险。

而在测试时，应该用整个训练好的模型，因此不需要dropout。



**Dropout 如何平衡训练和测试时的差异**

训练时还要对输出向量除以（1-p）之后再传给输出层神经元，作为神经元失活的补偿，以使得在训练时和测试时每一层输入有大致相同的期望。



**为什么Dropout可以解决过拟合？**

```
1）取平均的作用。如果用相同的数据训练多个不同的网络，会得到不同的结果。对这些结果取平均或者投票策略，可以防止过拟合。Dropout就类似于训练不同的网络，随机删掉比如一半的神经元，导致每批次的网络结构都不同，整个Dropout过程就相当于对多个不同的神经网络取平均。
2）减少神经元之间的共适应关系。Dropout导致两个神经元不一定每次都在网络中出现，这样权值的更新就不再依赖于固定关系的隐含节点的共同作用，减少了对局部特征的依赖。提升了网络的鲁棒性。 
3）类似于性别在生物进化过程中的角色：物种为了生存往往倾向于适应环境，环境突变时物种难以作出及时反应，性别的出现可以繁衍出适应新环境的变种。避免环境改变时物种可能面临的灭绝。
```

（1）取平均的作用： 如果用相同的训练数据去训练5个不同的神经网络，一般会得到5个不同的结果，此时我们可以采用 “5个结果取均值”或者“多数取胜的**投票策略**”去决定最终结果。这种“综合起来取平均”的策略通常可以有效防止过拟合问题。因为不同的网络可能产生不同的过拟合，取平均则有可能让一些“相反的”拟合互相抵消。**dropout就类似在训练不同的网络，随机删掉一半隐藏神经元导致网络结构已经不同，整个dropout过程就相当于对很多个不同的神经网络取平均。**而不同的网络产生不同的过拟合，一些互为“反向”的拟合相互抵消就可以达到整体上减少过拟合。

（2）减少神经元之间复杂的共适应关系： **dropout导致两个神经元不一定每次都在一个dropout网络中出现。这样权值的更新不再依赖于有固定关系的隐含节点的共同作用，阻止了某些特征仅仅在其它特定特征下才有效果的情况** 。迫使网络去学习更加鲁棒的特征 ，这些特征在其它的神经元的随机子集中也存在。**换句话说假如我们的神经网络是在做出某种预测，它不应该对一些特定的线索片段太过敏感，即使丢失特定的线索，它也应该可以从众多其它线索中学习一些共同的特征。**从这个角度看dropout就有点像L1，L2正则，减少权重使得网络对丢失特定神经元连接的鲁棒性提高。

（3）Dropout类似于性别在生物进化中的角色：物种为了生存往往会倾向于适应这种环境，环境突变则会导致物种难以做出及时反应，性别的出现可以繁衍出适应新环境的变种，有效的阻止过拟合，即避免环境改变时物种可能面临的灭绝。





### ⚠️LayerNorm



### BatchNorm

https://zhuanlan.zhihu.com/p/437446744

Batch-Normalization (BN)是一种让神经网络训练更快、更稳定的方法(faster and more stable)。它计算每个mini-batch的均值和方差，并将其拉回到均值为0方差为1的标准正态分布。BN层通常在nonlinear function的前面/后面使用。



**计算方法**

训练阶段：

<img src="https://tva1.sinaimg.cn/large/e6c9d24ely1gzurl5uwijj21400piacg.jpg" alt="img" style="zoom:33%;" />

用(1)(2)式计算一个mini-batch之内的均值和方差

用(3)式来进行normalize。这样，每个神经元的output在整个batch上是标准正态分布。

<img src="https://tva1.sinaimg.cn/large/e6c9d24ely1gzurls03agj21bg0jg767.jpg" alt="截屏2022-03-01 22.46.18" style="zoom:33%;" />

最后，使用可学习的参数  $\gamma$ 和 $\beta$ 来进行线性变换。这是为了让因训练所需而“刻意”加入的BN能够有可能还原最初的输入，从而保证整个network的capacity。

也可以理解为找到线性/非线性的平衡：以Sigmoid函数为例，batchnorm之后数据整体处于函数的非饱和区域，只包含线性变换，破坏了之前学习到的特征分布 。增加  $\gamma$ 和 $\beta$ 能够找到一个线性和非线性的较好平衡点，既能享受非线性的较强表达能力的好处，又避免太靠非线性区两头"死区"（如sigmoid)使得网络收敛速度太慢。



测试阶段：

可能输入就只有一个实例，看不到Mini-Batch其它实例。于是计算

- $\mu_{pop}$ : estimated mean of the studied population ;
- $\sigma_{pop}$ : estimated standard-deviation of the studied population.

$\mu_{pop}$ 和 $\sigma_{pop}$ 是由训练时的 $\mu_{batch}$ 和 $\sigma_{batch}$ 计算得到的。这两个值代替了在(1)(2)中算出的均值和方差，可以直接带入(3)式。



**实际应用**

在全连接网络中是对**每个神经元**进行归一化，也就是每个神经元都会学习一个 $\gamma$ 和 $\beta$



**BN的缺点**

在testing阶段，$\mu_{pop}$ 和 $\sigma_{pop}$ 使用的是估计值。假如训练集和测试集分布不同，会导致测试集经过batchnorm之后并不是服从均值为0，方差为1的分布。



**BN层的作用**

1. **降低了每层网络之间的依赖**
2. BN的存在会引入mini-batch内**其他样本的信息**，就会导致预测一个独立样本时，其他样本信息相当于**正则项**，使得**loss曲面变得更加平滑**，更容易找到最优解。相当于一次独立样本预测可以看多个样本，学到的特征**泛化性**更强





⚡️<font color=red>**BatchNorm的参数和计算过程**</font>

（2022.02.28 字节电商NLP二面深挖）





**LayerNorm和BatchNorm的区别**

（对谁做归一化，谁的大小关系就保留）

Batch Normalization 的处理对象是一批样本， 对这批样本的同一维度特征做归一化。

它去除了不同特征之间的大小关系，但是保留了不同样本间的大小关系，所以在CV领域用的多。

Layer Normalization 的处理对象是单个样本。 是对这单个样本的所有维度特征做归一化。

它去除了不同样本间的大小关系，但是保留了一个样本内不同特征之间的大小关系，所以在NLP领域用的多。（在NLP任务中，**序列的长度大小是不相等的**）



所以，LN不依赖于batch的大小和输入sequence的深度，因此可以用于batchsize为1和RNN中对边长的输入sequence的normalize操作。



举例：

将输入的图像shape记为[N, C, H, W]，这几个方法主要的区别就是在，

- batchNorm是在batch上，对NHW做归一化，对小batchsize效果不好；
- layerNorm在通道方向上，对CHW归一化，主要对RNN作用明显

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkvbnaxs4j30ja0b0dh4.jpg" alt="截屏2021-12-21 02.38.38" style="zoom:50%;" />





**简答讲一下BatchNorm技术，以及它的优缺点。**

BatchNorm是对每一批的数据在进入激活函数前进行归一化。

优点：可以提高收敛速度，防止过拟合，防止梯度消失，增加网络对数据的敏感度。







### 梯度消失和梯度爆炸

解决梯度爆炸：

- 可以通过梯度截断

  <img src="https://tva1.sinaimg.cn/large/008i3skNly1gxl7uu2h0jj30rm08kdgb.jpg" alt="截屏2021-12-21 09.52.13" style="zoom:33%;" />

- 通过添加正则项。

解决梯度消失：

- 将RNN改掉，使用LSTM等自循环和门控制机制。
- 优化激活函数，如将sigmold改为relu
- 使用batchnorm
- 使用残差结构





### 数据集划分

https://zhuanlan.zhihu.com/p/50221679

数据集的划分一般有三种方法：

1. 按一定比例划分为训练集和测试集

   直接将数据随机划分为训练集和测试集，然后使用训练集来生成模型，再用测试集来测试模型的**正确率**和**误差**

2. 训练集、验证集、测试集法

   首先将数据集划分为训练集和测试集，由于模型的构建过程中也需要检验模型，检验模型的配置，以及训练程度，过拟合还是欠拟合，所以会将训练数据再划分为两个部分，一部分是用于训练的训练集，另一部分是进行检验的验证集。当模型“通过”验证集之后，我们再使用测试集测试模型的最终效果，评估模型的准确率，以及误差等。

3. 交叉验证法

   交叉验证一般采用k折交叉验证，即k-fold cross validation。在这种数据集划分法中，我们将数据集划分为k个子集，每个子集均做一次测试集，每次将其余的作为训练集。在交叉验证时，我们重复训练k次，每次选择一个子集作为测试集，并将k次的平均交叉验证的正确率作为最终的结果。





### 数据不平衡怎么处理

https://zhuanlan.zhihu.com/p/56960799

https://zhuanlan.zhihu.com/p/24814085

https://zhuanlan.zhihu.com/p/38183927

https://zhuanlan.zhihu.com/p/260407405



**数据**

- 欠采样：从样本较多的类中再抽取，仅保留这些样本点的一部分；
- 过采样：复制少数类中的一些点，以增加其基数；
- 生成合成数据：从少数类创建新的合成点，以增加其基数。

CV数据的各种拉伸，收缩，翻转，放大和缩小

同义词替换

GAN增强（cycle-GANs，构造领域之间的映射）



**损失函数**

加惩罚项：样本少类别惩罚项（weight）大，所以模型将该类别分错后代价较大，**迫使模型更加注重小样本数据**。（解决正负样本不平衡的问题）

难易样本：[focal loss](https://zhuanlan.zhihu.com/p/80594704)，**把高置信度(p)样本的损失再降低一些**

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz1o9lj9uxj30m004q0su.jpg" alt="截屏2022-02-04 18.49.20" style="zoom:50%;" />



**改变评价指标**

通过 confusion matrix 来计算 precision 和 recall, 然后通过 precision 和 recall 再计算f1 分数

ROC AUC曲线



**使用其他机器学习方法**

比如决策树不会受到不均匀数据的影响（树模型有一个class_weight关于样本权重的参数可以调整）



（2022.02.28 字节NLP 出了样本不均衡场景题）





### 聚类如果不清楚有多少类，有什么方法？

层次聚类法

假设有 n 个待聚类的样本，对于层次聚类算法，它的步骤是：

- 步骤一：（初始化）将每个样本都视为一个聚类；
- 步骤二：计算各个聚类之间的相似度；
- 步骤三：寻找最近的两个聚类，将他们归为一类；
- 步骤四：重复步骤二，步骤三；直到所有样本归为一类。





### ⚠️卷积是什么样的过程？





### beam search

https://zhuanlan.zhihu.com/p/43703136

beam search尝试在广度优先基础上进行进行搜索空间的优化（类似于剪枝）达到减少内存消耗的目的。



定义词表大小是V，beam size是 B，序列长度是L。

假设V=100，B=3：

1. 生成第1个词时，选择概率最大的3个词（假设是a，b，c），即从100个中选了前3个；

2. 生成第2个词时，将当前序列a/b/c分别与词表中的 100个词组合，得到 3*100个序列，从中选 3个概率最大的，作为当前序列（假设现在是am，bq，as）；

3. 持续上述过程，直到结束。最终输出3个得分最高的。



### ⚠️训练和预测时，batch_size不同怎么办？

(2022.02.28字节电商NLP二面)

训练的时候是一组一组放进去的，但预测的时候，一个一个放进去，是怎么处理的？

我也不知道啊。











## 概念/计算

### tf-idf

词频-逆向文件频率

TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

TF-IDF的主要思想是：**如果某个单词在一篇文章中出现的频率TF高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类**。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxcuf26vsnj30eo03wq33.jpg" alt="截屏2021-12-14 04.02.49" style="zoom:40%;" /> <img src="https://tva1.sinaimg.cn/large/008i3skNly1gxcufk2srvj30h003ot8y.jpg" alt="截屏2021-12-14 04.03.16" style="zoom:40%;" />



### 精度、召回率、准确率、F1

> 所谓准确率（accuracy）就是正确预测的数量除以预测总数；
>
> 类别精度（precision）表示当模型判断一个点属于该类的情况下，判断结果的可信程度。
>
> 类别召回率（recall）表示模型能够检测到该类的比率。
>
> 类别的 F1 分数是精度和召回率的调和平均值（F1 = 2×precision×recall / (precision + recall)），F1 能够将一个类的精度和召回率结合在同一个指标当中。



**精度**表示的是**预测为正的样本中有多少是真正的正样本**。那么预测为正就有两种可能了，一种就是把正类预测为正类(TP)，另一种就是把负类预测为正类(FP)，也就是

![[公式]](https://www.zhihu.com/equation?tex=P++%3D+%5Cfrac%7BTP%7D%7BTP%2BFP%7D)

**召回率**表示的是**样本中的正例有多少被预测正确**了。那也有两种可能，一种是把原来的正类预测成正类(TP)，另一种就是把原来的正类预测为负类(FN)。

![[公式]](https://www.zhihu.com/equation?tex=R+%3D+%5Cfrac%7BTP%7D%7BTP%2BFN%7D)

准确率的定义是**预测正确的结果占总样本的百分比**

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%E5%87%86%E7%A1%AE%E7%8E%87+%3D+%5Cfrac%7BTP%2BTN%7D%7BTP%2BTN%2BFP%2BFN%7D+%5C%5C+%5Cend%7Bequation%7D)



F-score相当于precision和recall的调和平均



$$
F1 = \frac{2PR}{P+R}
$$

### ROC和AUC

https://zhuanlan.zhihu.com/p/349366045

**ROC曲线是用来衡量分类器的分类能力。** ROC曲线不容易受数据集中正负样本不平衡的影响。

**AUC表示，随机抽取一个正样本和一个负样本，分类器正确给出正样本的score高于负样本的概率**



ROC (receiver operating characteristic curve)

首先我们需要定义下面两个变量：**FPR、TPR(即为我们常说的召回recall)**。

**FPR**表示，在所有的恶性肿瘤中，被预测成良性的比例。称为伪阳性率。$FPR = \frac{FP}{FP+TN}$

**TPR**表示，在所有良性肿瘤中，被预测为良性的比例。称为真阳性率。 $TPR = \frac{TP}{TP+FN}$

在二分类（0，1）的模型中，最后的输出一般是一个概率值，表示结果是1的概率。需要一个阈值，超过这个阈值则归类为1，低于这个阈值就归类为0。所以，不同的阈值会导致分类的结果不同，也就是混淆矩阵不一样了，FPR和TPR也就不一样了。所以当阈值从0开始慢慢移动到1的过程，就会形成很多对(FPR, TPR)的值，将它们画在坐标系上，就是所谓的ROC曲线了。



AUC (Area under ROC)

ROC曲线下的面积



### L1, L2正则化

https://zhuanlan.zhihu.com/p/137073968

https://zhuanlan.zhihu.com/p/38309692

正则化的目的是限制参数过多或者过大，避免模型更加复杂。

为了防止过拟合，我们可以将其高阶部分的权重 w 限制为 0，这样，就相当于从高阶的形式转换为低阶。为了达到这一目的，最直观的方法就是限制 w 的个数，但是这类条件求解非常困难。所以，一般的做法是寻找更宽松的限定条件：$\sum_{j} w_{j}^{2} \leq C$

即所有w 的平方和不超过参数 C。这时候，**我们的目标就转换为：最小化训练样本误差 Ein，但是要遵循 w 平方和小于 C 的条件。**



**L1正则**

根据权重的绝对值的总和来惩罚权重。
$$
l_{1}: \Omega(w)=\|w\|_{1}=\sum_{i}\left|w_{i}\right|
$$
L1正则常被用来进行特征选择，主要原因在于**L1正则化会使得较多的参数为0，从而产生稀疏解**，我们可以将0对应的特征遗弃，进而用来选择特征。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz1x1xi8aaj30vg06sdgc.jpg" alt="截屏2022-02-04 23.53.25" style="zoom: 50%;" />

**L2正则**

根据权重的平方和来惩罚权重。
$$
l_{2}: \Omega(w)=\|w\|_{2}^{2}=\sum_{i}\left|w_{i}^{2}\right|
$$
主要用来防止模型过拟合

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz1x2iwx86j30iu04eq2y.jpg" alt="截屏2022-02-04 23.53.59" style="zoom:50%;" />



**为什么L1会产生稀疏解？**

https://zhuanlan.zhihu.com/p/129024068

1）从梯度角度

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz1x7lo6ghj30tm0dkgm8.jpg" alt="截屏2022-02-04 23.58.53" style="zoom: 40%;" /> <img src="https://tva1.sinaimg.cn/large/008i3skNly1gz1x7wr9n5j30t20e4aam.jpg" alt="截屏2022-02-04 23.59.11" style="zoom:40%;" /> 

引入L2正则时，代价函数在0处的导数仍是d0，无变化。而引入L1正则后，代价函数在0处的导数有一个突变。从d0+λ到d0−λ，若d0+λ和d0−λ异号，则在0处会是一个极小值点。因此，优化时，很可能优化到该极小值点上，即w=0处。通常越大的λ 可以让代价函数在参数为0时取到最小值。



2）图像的角度

两种正则化，能不能将最优的参数变为0，取决于最原始的损失函数在**0点处的导数**。

施加 L1 正则项后，导数在 w=0 处不可导。不可导点是否是极值点，就是看不可导点左右的单调性。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz1xiutc8oj316i07a75n.jpg" alt="截屏2022-02-05 00.09.41" style="zoom:50%;" />





### 激活函数

#### softmax

**softmax用于多分类过程中**，它将多个神经元的输出，映射到（0,1）区间内

softmax函数将未规范化的预测变换为非负并且总和为1，同时要求模型保持可导。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkrkem764j30og04qmxa.jpg" alt="截屏2021-12-21 00.28.39" style="zoom:50%;" />



**softmax的一个小缺陷：上溢和下溢**

当x=[10000,5000,2000]的时候，$exp(10000)$超过了计算机所能存储的最大范围，就会发生溢出。当x=[-10000,-1000,-34343]的时候，分母很小很小，基本为0，导致计算结果为nan.

解决方法：将原数组变成x-max(x)



**softmax的缺点：计算复杂度高**

考虑负采样和分层softmax



**softmax及其导数**

对于，（这里ai是yi对应的softmax）
$$
L o s s=-\sum_{i} y_{i} \ln a_{i}
$$
<img src="https://pic2.zhimg.com/80/v2-d3a4e22a107052ee998823b24b49db71_1440w.jpg" alt="img" style="zoom:75%;" /> <img src="https://pic3.zhimg.com/80/v2-5eafb4c0a835bc90248766ac0c123dfe_1440w.jpg" alt="img" style="zoom:75%;" />

所以

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkrwf36zcj30te05gjrl.jpg" alt="截屏2021-12-21 00.40.12" style="zoom:50%;" />

即 aj - yj



推导还是看[这个](https://blog.csdn.net/GreatXiang888/article/details/99293507)





#### sigmoid

https://www.jianshu.com/p/c78af484559b
$$
f(z)=\frac{1}{1+e^{-z}}
$$

它能够把输入的连续实值变换为0和1之间的输出，特别的，如果是非常大的负数，那么输出就是0；如果是非常大的正数，输出就是1.

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkv1qipwcj312y0dodgi.jpg" alt="截屏2021-12-21 02.29.04" style="zoom:50%;" />

**缺点**

1. 容易导致梯度消失。每传递一层梯度值都会减小为原来的0.25倍，如果神经网络隐藏层很多，那么梯度在穿过多层后会变得接近0，即出现梯度消失现象。

   > 这里引申出**反向传播**的概念：简要地说，BP算法是一个迭代算法，它的基本思想为：（1）先计算每一层的状态和激活值，直到最后一层（即信号是前向传播的）；（2）计算每一层的误差，误差的计算过程是从最后一层向前推进的（这就是反向传播算法名字的由来）；（3）更新参数（目标是误差变小）。求解梯度用链导法则。迭代前面两个步骤，直到满足停止准则（比如相邻两次迭代的误差的差别很小）。

   **改进**：1、LSTM可以解决梯度消失问题 2、Batchnorm 3、优化激活函数，使用relu 4、使用残差结构

2. 函数输出不是0均值（zero-centered）

   sigmoid 函数的输出均大于 0，使得输出不是 0 均值，这称为偏移现象，这会导致后一层的神经元将得到上一层输出的非 0 均值的信号作为输入。

   **改进**：**数据规范化（normalization）如Layer-Normalization Batch-Normalization等**

3. 解析式中含有幂运算：幂运算对计算机来讲比较耗时，对于规模比较大的深度网络，这会较大地增加训练时间。



#### tanh

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkuvp502mj314g0iewfh.jpg" alt="截屏2021-12-21 02.23.19" style="zoom:50%;" />

特点：和sigmoid差不多，但值域为[-1,1]

此外，解决了Sigmoid函数的不是zero-centered输出问题。



缺点：

仍存在梯度消失问题

仍存在幂运算问题



#### ReLu

$$
ReLu(x) = max(0, x)
$$

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkux8tx5hj313y0dgjrz.jpg" alt="截屏2021-12-21 02.24.47" style="zoom:50%;" />

特点：


- 解决了梯度消失问题

- 计算速度非常快，只需要判断是否大于0

- 收敛速度快于sigmoid和tanh，因为ReLu的收敛速度一直为1

缺点：

- 不是0均值的（zero-centered）

- Dead ReLU Problem，指**某些神经元可能永远不会被激活**，导致相应的参数永远不能被更新。

  Dead ReLU: 输入relu中的值如果存在负数，则最终经过relu之后变成0，极端情况下是输入relu的所有值全都是负数。当神经元中的大多数返回零输出时，梯度在反向传播期间无法流动，并且权重不会更新。

  产生Dead ReLU的原因：1）参数初始化  2）learning rate太高导致在训练过程中参数更新太大

- 原点不可导。解决办法：坐标轴下降法、最小角回归法



#### Leaky ReLu

$$
\text { LeakyReLU }(x)=\max (\alpha x, x)
$$

$\alpha$通常取0.01

对ReLu的一个改进，可以改善relu中x<0部分的dead问题。



### 损失函数

https://zh-v2.d2l.ai/chapter_linear-networks/softmax-regression.html

https://zhuanlan.zhihu.com/p/25723112

https://blog.csdn.net/GreatXiang888/article/details/99293507



#### 交叉熵损失

![截屏2021-12-21 00.38.20](https://tva1.sinaimg.cn/large/008i3skNly1gxkruhf2r4j31jy07s3z2.jpg)



#### Mean Square Loss

$$
M S E=\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-t_{i}\right)^{2}
$$



#### 绝对值损失函数

$$
L(Y, f(x))=|Y-f(x)|
$$





### 优化器

https://www.zhihu.com/question/323747423

#### GD（标准梯度下降）

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkwfuifwjj321209ywg5.jpg" alt="截屏2021-12-21 03.17.14" style="zoom:50%;" />

沿着梯度的方向不断减小模型参数，从而最小化代价函数。



缺点：

- 训练速度慢：每走一步都要要计算调整下一步的方向，并且**每次迭代都要遍历所有的样本**。会使得训练过程及其缓慢，需要花费很长时间才能得到收敛解。
- 容易陷入局部最优解：由于是在有限视距内寻找下山的方向。当陷入平坦的洼地，会误以为到达了山地的最低点，从而不会继续往下走。所谓的局部最优解就是鞍点。落入鞍点，梯度为0，使得模型参数不在继续更新。





#### SGD（随机梯度下降）

随机梯度下降（SGD）可降低每次迭代时的计算代价。在随机梯度下降的每次迭代中，我们对数据样本**随机均匀采样一个样本**，计算梯度。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkwls4etlj30ao02sdfn.jpg" alt="截屏2021-12-21 03.22.56" style="zoom:50%;" />

优点：

- 计算梯度快

缺点：

- SGD在随机选择梯度的同时会**引入噪声**，使得权值更新的方向不一定正确。
- SGD也没能单独克服局部最优解的问题。





#### BGD（批量梯度下降）

在时间t采样一个**随机子集**，大小为b，算梯度时对这b个样本取平均。

批量梯度下降法比标准梯度下降法训练时间短，且每次下降的方向都很正确。



#### Momentum

使用动量(Momentum)的随机梯度下降法(SGD)，主要思想是引入一个**积攒历史梯度信息的动量**来加速SGD。

（$\beta$和$1-\beta$ 的应用是指数加权移动平均法）

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz3uo1n9w7j30j805y3yl.jpg" alt="截屏2022-02-06 16.01.58" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz3qb43uukj30tk0ksjt8.jpg" alt="截屏2022-02-06 13.31.09" style="zoom:33%;" />

动量主要解决SGD的两个问题：

一是随机梯度的方法（引入的噪声）；

二是Hessian矩阵病态问题（可以理解为SGD在收敛过程中和正确梯度相比来回摆动比较大的问题）。



#### Nesterov

```
既然已经走到了新的参数处，那么 梯度可以用跨出一步之后的梯度 ，这样就能用到更多的信息。
当参数向量位于某个位置x时，未来近似的位置可以看做 x + µv，在这个位置计算梯度，然后更新动量，进而更新权重。
```

既然已经走到了新的参数处，那么**梯度可以用跨出一步之后的梯度**，这样就能用到更多的信息。

当参数向量位于某个位置 *x* 时，

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz3usy70frj31kw0q6q6p.jpg" alt="截屏2022-02-06 16.06.42" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz3usbtml0j317k0g8ab7.jpg" alt="截屏2022-02-06 16.06.07" style="zoom:50%;" />



#### AdaGrad

Adaptive Gradient，**学习率要适当地根据每个参数的历史数据来调整**。

学习到的梯度是真实梯度除以梯度内积的开方。Adagrad本质是 **解决各方向导数数值量级的不一致而将梯度数值归一化**。

（这样的好处就是：如果每个参数的振荡幅度不一样，我们这样相当于做了某种归一化，使得它们在自己的范围内做基本一致的变化）

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gz3v6sk3cdj30is0eiwez.jpg" alt="截屏2022-02-06 16.20.00" style="zoom:50%;" />



#### RMSprop

AdaGrad有一个问题：就是随着迭代进行，显然h会越来越大，所以最后更新量会变为0，为了改善这个问题，RMSprop**对过去梯度进行逐步的遗忘**，也就是每次都乘以一个小于1的系数，进行指数移动平均，**呈指数函数式地减少过去梯度的影响**。
$$
\begin{array}{l}
\theta \leftarrow \theta-\frac{\alpha}{\sqrt{h}} \cdot g \\
h \leftarrow \beta h+(1-\beta) g \odot g
\end{array}
$$
通过小于1的 $\beta$ 来进行前面梯度信息的遗忘。





#### Adam

Adam全名为Adaptive Momentum，也就是，既要Adaptive学习率，而且这个Adaptive还不是AdaGrad里那么单纯，其实用的是**RMSprop里这种逐渐遗忘历史的方法，同时还要加入Momentum。**
$$
\begin{array}{l}
v \leftarrow \beta_{1} v+\left(1-\beta_{1}\right) g \\
h \leftarrow \beta_{2} h+\left(1-\beta_{2}\right) g \odot g \\
\theta \leftarrow \theta-\frac{\alpha}{\sqrt{h}} \cdot v
\end{array}
$$










## 场景题

现在有一些新闻，包含军事、体育、经济等，想分出它属于哪个类，该怎么做

```
文本预处理：分词、去停用词、词性标注

贝叶斯
```



一句话中有一个错别字，如何快速找到

<img src="https://tva1.sinaimg.cn/large/e6c9d24ely1gzsn2cycr8j214q08gwfh.jpg" alt="截屏2022-02-28 02.38.03" style="zoom:50%;" />

```
纠错：
第一步通过常用词词典匹配结合统计语言模型的方式进行错误检测；
第二步利用近音字，近形字和混淆字进行候选召回；
最后一步利用统计语言模型进行打分排序。
```



售卖商品收集回来的反馈和评价信息，如何区分正负向，说一个解决方案

https://blog.csdn.net/dingming001/article/details/82935715

下面这个方案不行

```
数据预处理(文本去重、机械压缩、短句删除)、中文分语、停用词过滤

训练得到词向量

利用词向量构建的结果，再进行评论集子集的人工标注。正面评论标为1,负面评价标记为2。将每条评论映射为一个向量，将分词后评论中的所有词语对应的词向量相加做平均，使得一条评论对应一个向量。

Bert
```



推荐答案：

先随便训练个弱一点的，进行预测，然后对于模糊不清的结果（比如认为概率大于0.5的是正样本，小于0.5的是负样本，但有时候0.45的也可能是正样本），手动调节（打标签）一下，再放入训练。



------

如何从大量未标注的图片库中，按照用户的语义查找出要的照片，说一个解决方案

小样本学习

https://blog.csdn.net/weixin_40123108/article/details/89003325





### 概率题

圆上三个点，钝角三角形的概率



一对夫妇有两个孩子，已知其中有一个孩子是出生于周二的男孩，问另一个孩子也是男孩的概率。

https://www.zhihu.com/question/36802129



一组数据同时扩大n倍，方差和均值的变化



## 有什么要问的吗



