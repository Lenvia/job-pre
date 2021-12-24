[TOC]



## 模型

### word2vec

简介

它将每个词映射到一个固定长度的向量，这些向量能更好地表达不同**词之间的相似性和类比关系**。（使得语义相似的单词在嵌入式空间中的距离很近）

word2vec 本质上是一种**降维**操作——把词语从 one-hot encoder 形式的表示降维到 word2vec 形式的表示。（更低维度的向量的表示）



#### 实施过程

https://www.cnblogs.com/zhangyang520/p/10969975.html



#### skip-gram

跳元模型假设**一个词**可以用来在文本序列中**生成其周围的单词**

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxksw8th39j31iy0icq4d.jpg" alt="截屏2021-12-21 01.14.36" style="zoom:33%;" />

**跳元模型考虑了在给定中心词的情况下生成周围上下文词的条件概率。**

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkswol66nj31n60bkn00.jpg" alt="截屏2021-12-21 01.15.04" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxksyecyjnj31n00eimzu.jpg" alt="截屏2021-12-21 01.16.42" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxksyxl4r6j31be0u0wjb.jpg" alt="截屏2021-12-21 01.17.08" style="zoom: 67%;" />

对词典中索引为 i 的词进行训练后，得到 vi（作为中心词）和 ui（作为上下文词）两个词向量。在自然语言处理应用中，**跳元模型的中心词向量通常用作词表示。**



Skip-gram处理步骤：

1. 确定窗口大小window，对每个词生成2*window个训练样本，(i, i-window)，(i, i-window+1)，...，(i, i+window-1)，(i, i+window)

2. 确定batch_size，注意batch_size的大小必须是2*window的整数倍，这确保每个batch包含了一个词汇对应的所有样本

3. 训练算法有两种：层次Softmax和Negative Sampling

4. 神经网络迭代训练一定次数，得到输入层到隐藏层的参数矩阵，矩阵中每一行的转置即是对应词的词向量









#### CBOW

连续词袋模型假设**中心词**是基于其在文本序列中的**周围上下文词生成的**

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkt163k6fj31260igdgt.jpg" alt="截屏2021-12-21 01.19.22" style="zoom:50%;" />

**连续词袋模型考虑了给定周围上下文词生成中心词条件概率**

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkt29is9wj31l60u00z6.jpg" alt="截屏2021-12-21 01.20.25" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkt2uilacj31ny0req7u.jpg" alt="截屏2021-12-21 01.20.56" style="zoom:50%;" />



CBOW的处理步骤：

1. 确定窗口大小window，对每个词生成2*window个训练样本，(i-window, i)，(i-window+1, i)，...，(i+window-1, i)，(i+window, i)

2. 确定batch_size，注意batch_size的大小必须是2*window的整数倍，这确保每个batch包含了一个词汇对应的所有样本

3. 训练算法有两种：层次Softmax和Negative Sampling

4. 神经网络迭代训练一定次数，得到输入层到隐藏层的参数矩阵，矩阵中每一行的转置即是对应词的词向量



### 两种优化方法：分层softmax、负采样

由于softmax操作的性质，上下文词可以是词表V中的任意项。

计算log条件概率时，包含与整个词表大小一样多的项的求和，而在一个词典上（通常有几十万或数百万个单词）求和的梯度的计算成本是巨大的。



为了解决这个问题，我们直观的想法就是**限制每次必须更新的输出向量的数量**。一种有效的手段就是采用**分层softmax**；另一种可行的方法是通过**负采样**。



#### hierarchical softmax

本质是把 N 分类问题变成 log(N)次二分类

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkteq0ab1j311w0lyjta.jpg" alt="截屏2021-12-21 01.32.23" style="zoom:50%;" />

用二叉树来表示词汇表中的所有单词。V个单词必须存储于二叉树的叶子单元。可以被证明一共有V-1个内单元。对于每个叶子节点，有一条唯一的路径可以从根节点到达该叶子节点；该路径被用来计算该叶子结点所代表的单词的概率。

一个单词作为输出词的概率被定义为：

从根节点出发到叶子结点的随机路径。在每个隐节点（包含根节点），我们需要分配往左走或往右走的概率。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxktgw6r8sj31d00b6gnz.jpg" alt="截屏2021-12-21 01.34.27" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxktj7mqutj31q20foadc.jpg" alt="截屏2021-12-21 01.36.39" style="zoom:40%;" />

训练模型的计算复杂度从 ![[公式]](https://www.zhihu.com/equation?tex=O%28V%29) 降至 ![[公式]](https://www.zhihu.com/equation?tex=O%28logV%29) ，这在效率上是一个巨大的提升。而且我们仍然有差不多同样的模型参数（原始模型： ![[公式]](https://www.zhihu.com/equation?tex=V) 个单词的输出向量，分层softmax： ![[公式]](https://www.zhihu.com/equation?tex=V-1) 个隐节点的输出向量)。



#### negative sampling

```
Negative Sampling是对于给定的词,并生成其负采样词集合的一种策略。
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



#### 衡量生成序列的好坏——BLEU

计算预测中所有 n-gram 的精度。

惩罚过短的预测 + 长匹配有高权重

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkas8d5ycj31100q00vs.jpg" alt="截屏2021-12-20 14.47.59" style="zoom:33%;" />





#### 总结

Seq2Seq是从一个句子生成另一个句子

编码器和解码器都是RNN

将编码器最后时间隐状态来初始化解码器隐状态来完成信息传递

常用BLEU来衡量生成序列的好坏





### ⚠️RNN

循环神经网络的输出取决于当下输入和前一时间的隐变量。

应用到语言模型中时，循环神经网络根据当前词预测下一时刻词。

通常使用困惑度来衡量语言模型的好坏。





**相比于n-gram，为什么要用RNN？**

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





### ⚠️GRU

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxlde8fxenj319e0psmzx.jpg" alt="截屏2021-12-21 12.58.36" style="zoom:50%;" />

```
GRU有两个门，一个是重置门，一个是更新门。
它们都是由上一时刻的隐状态和当前时刻的输入计算得到的。
当重置门接近于1的时候，t-1时刻的候选隐状态更接近于普通RNN计算的结果。 当重置门接近于0的时候，t-1时刻的候选隐状态就相当于X作为输入变量到多层感知机的结果。
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





### ⚠️LSTM

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxldvykp7dj319y0oi775.jpg" alt="截屏2021-12-21 13.19.42" style="zoom:50%;" />

```
LSTM有一个记忆单元还有三个门，分别是遗忘门，输入门和输出门。
对于候选的记忆单元，和三个门都是由上一时刻的隐状态和当前时刻的输入得到的。
对于记忆单元，输入门 控制多少来自于 候选记忆单元的新数据，遗忘门就是控制保留多少过去的记忆单元 Ct-1的内容。（引入这种设计是用来缓解梯度消失问题，以及更好地捕获序列中长距离依赖关系）
最后关于隐状态的计算，输出门就是控制对传递记忆信息的程度。
```

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxle2dx9i3j31cs08qmz0.jpg" alt="截屏2021-12-21 13.27.06" style="zoom:50%;" />



<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxle2qgycfj31d40b6q5o.jpg" alt="截屏2021-12-21 13.27.27" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxle36cs5nj31cq0a20ur.jpg" alt="截屏2021-12-21 13.27.52" style="zoom:50%;" />



小结：

- 长短期记忆网络有三种类型的门：输入门、遗忘门和输出门。
- 长短期记忆网络的隐藏层输出包括“隐状态”和“记忆元”。只有隐状态会传递到输出层，而记忆元完全属于内部信息。
- 长短期记忆网络可以缓解梯度消失和梯度爆炸。



#### 为什么有效

门的动态控制（既能捕获到长期的依赖关系，也能捕获到序列短期的依赖关系）

可以缓解梯度消失和梯度爆炸



#### LSTM和GRU的区别和使用场景

GRU的优点是其模型的简单性 ，因此更适用于构建较大的网络。它只有两个门控，从计算角度看，它的效率更高，它的可扩展性有利于构筑较大的模型；但是LSTM更加的强大和灵活，因为它具有三个门控。LSTM是经过历史检验的方法。



#### 双向LSTM

- 在双向循环神经网络中，**每个时间步的隐状态由当前时间步的前后数据同时决定**。
- 双向循环神经网络与概率图模型中的“前向-后向”算法具有相似性。
- 双向循环神经网络主要用于序列编码和给定双向上下文的观测估计。
- 由于**梯度链更长**，因此双向循环神经网络的训练代价非常高。



### ⚠️Transformer

https://zh-v2.d2l.ai/chapter_attention-mechanisms/transformer.html

```
介绍一下Transformer？
Transformer是一个纯使用注意力的编码-解码器。
它的编码、解码器都有n个transformer块。
每个transformer快都使用多头注意力、基于位置的前馈网络、层与层之间使用残差连接和层归一化。

在编码器中，由多个层叠加。每个层都有两个子层，一个是多头自注意力，一个是基于位置的前馈网络。
在计算多头自注意力时，query key value来自于上一层的输出，然后使用残差连接，再进行layerNorm

在解码器中，也是由多个层堆叠起来的。但是除了编码器中提到的两个子层，解码器还在两个子层之间插入了第三个子层，成为编码器-解码器注意力层。它的query来自于前一个解码器层的输出，key 和 value来自于整个编码器的输出。
然后在解码器的自注意力层中，它的query key和value来自于上一层的输出。
但是解码器中的每个位置，只能考虑该位置之前的所有位置。这种masked多头注意力确保预测仅依赖于已生成的词元。
```



Transformer是一个纯使用注意力的 编码-解码器

编码器和解码器都有n个 transformer 块（base n=6）

每个块里使用多头注意力，基于位置的前馈网络，和层归一化。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkdudis5aj30s20yewhe.jpg" alt="截屏2021-12-20 16.33.51" style="zoom:40%;" />

从宏观角度来看，transformer 的编码器是由多个相同的层叠加而成的，每个层都有两个子层。第一个子层是***多头自注意力***（multi-head self-attention）汇聚；第二个子层是***基于位置的前馈网络***（positionwise feed-forward network）。

具体来说，在计算编码器的自注意力时，查询、键和值都来自前一个编码器层的输出。受 [7.6节](https://zh-v2.d2l.ai/chapter_convolutional-modern/resnet.html#sec-resnet)中残差网络的启发，每个子层都采用了*残差连接*（residual connection）。在残差连接的加法计算之后，紧接着应用 LayerNorm 。因此，输入序列对应的每个位置，transformer编码器都将输出一个d维表示向量。

Transformer解码器也是由多个相同的层叠加而成的，并且层中使用了残差连接和LayerNorm。除了编码器中描述的两个子层之外，解码器还在这两个子层之间插入了第三个子层，称为***编码器－解码器注意力*（encoder-decoder attention）层**。在编码器－解码器注意力中，**query来自前一个解码器层的输出**，而**键和值来自整个编码器的输出**。在解码器自注意力中，查询、键和值都来自上一个解码器层的输出。但是，解码器中的<u>每个位置只能考虑该位置之前的所有位置</u>。这种*掩蔽*（masked）注意力保留了*自回归*（auto-regressive）属性，<u>确保预测仅依赖于已生成的输出词元</u>。



### Bert

https://zh-v2.d2l.ai/chapter_natural-language-processing-pretraining/bert.html

```
介绍一下bert？
bert是一个基于微调的预训练模型，它抽取了足够多的信息。这些抽取的特征可以复用，可以挪到别的地方去。在面对新的任务时，只需要在后面加一个简单的输出层。而不需要加什么RNN Transformer等网络。

bert的输入 是 单个文本或单个文本对。
当输入为单个文本的时候，前面加cls，然后是文本序列的标记、以及特殊分隔词元sep
当输入为单个文本对的时候，前面是cls，然后是第一个文本序列的标记，加一个分隔词元sep，第二个文本序列标记，以及一个特殊分隔词元sep

输入有三个embedding
1. token embedding: 要将各个词转换成固定维度的向量
2. segment embedding: 使用分隔词进行拼接，然后对第一个句子id为0，第二个id为1
3. postion embedding: 让BERT在各个位置上学习一个向量表示来将序列顺序的信息编码进来。

bert的输出就是 对每一个词元 返回一个抽取了上下文信息的特征向量。

特点：上下文敏感，可以用于不可知任务。

预训练任务1: 带掩码的语言模型
Transformer编码器是双向的，语言模型要求是单向的（不能考虑它之后的词提供的信息）
每次随机（15%）将一些词元替换成<mask>。每次去预测一下是什么。
因为微调中没有mask，所以就不要让模型一看见mask、就去预测。
这时候
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



输入输出？

BERT输入序列明确地表示单个文本和文本对。

当输入为单个文本时，BERT输入序列是特殊类别词元“\<cls>”、文本序列的标记、以及特殊分隔词元“\<sep>”的连结。

当输入为文本对时，BERT输入序列是“\<cls>”、第一个文本序列的标记、“\<sep>”、第二个文本序列标记、以及“\<sep>”的连结。



三个embedding： **Token Enbedding, Segment Embedding, Position embedding**

- 要将各个词转换成固定维度的向量。在BERT中，每个词会被转换成768维的向量表示。
- 加入额外的片段嵌入。 给第一个句子 id为0，第二个句子id是1
- 位置编码可以学习。



模型输入：通过查询字向量表将文本中的每个字转换为一维向量

模型输出：每个位置都是融合全文语义信息后的向量表示。（抽取了上下文信息的特征向量）



**特点**

1. 从上下文无关到上下文敏感
2. 从特定任务到不可知任务

ELMo对上下文进行双向编码，但使用特定于任务的架构；而GPT是任务无关的，但是从左到右编码上下文。

BERT（来自Transformers的双向编码器表示）结合了这两个方面的优点。它对上下文进行双向编码，并且对于大多数的NLP任务只需要最少的架构改变。通过使用预训练的Transformer编码器，BERT能够基于其双向上下文表示任何词元。



**架构：**

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxk938f9gbj31a00oygog.jpg" alt="截屏2021-12-20 13.49.20" style="zoom: 33%;" />

Bert maxposition：512

谷歌开源的预训练模型，那么这个词表的大小将会被限制在512



#### 预训练任务1: 带掩码的语言模型

- Transformer的编码器是双向的，标准语言模型要求单向（不能考虑它之后的词的信息）。
- 每次随机（15%）将一些词元换成\<mask>。每次去预测一下是什么
- 因为微调任务中没有\<mask> （不要让模型看见mask就去预测）
  - 80%概率下，将选中的词元变成\<mask>
  - 10%概率换成一个随机词元
  - 10%概率保持原有的词元

#### 预训练任务2： 下一句子预测

- 预测一个句子对中两个句子是不是相邻
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

识别一个词是不是命名实体，例如人命、机构、位置。

将非特殊词（除去了cls，sep）放进全连接层分类。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkepn9mb1j30ia0dyjrq.jpg" alt="截屏2021-12-20 17.03.54" style="zoom:50%;" />



例子：问题问答

给定一个问题，和描述文字，找出一个片段作为回答

对片段中的每个词元预测它是不是回答的开头或结束

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkeqi6robj30pi0f40tb.jpg" alt="截屏2021-12-20 17.04.44" style="zoom:50%;" />





#### 微调总结

即使下游任务各有不同，使用Bert微调时均只需要增加输出层。

但根据任务的不同，输入的表示，和使用的Bert特征也会不一样。



### ⚠️SVM

与分类平面距离最近的样本点称为支持向量，进而构成支持平面。

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



#### SVM的理论依据，如何推导？





### XGboost

XGBoost的特点，缺失值如何处理





### lightgbm





### gbdt







### LR



### 隐马尔可夫



### 条件随机场



### 朴素贝叶斯







## 大类

### Attention

（下面的随意是指 follow your heart，而不是随便）

心理学认为人通过随意线索（自主性）和不随意线索（不自主的）选择注意点。

参考人类注意力的方式。非自主性提示下，选择偏向于感官输入。比如卷积、全连接层、池化层都只考虑不自主性的线索。

**注意力机制则显式地考虑随意性（有自主）的线索。**



**注意力机制通过 注意力汇聚 将*查询*（自主性提示）和*键*（非自主性提示）结合在一起，实现对*值*（感官输入）的选择倾向**



- 随意线索被称为查询 query
- 每个输入是一个 key-value对。key表示不随意性的线索，value表示它的值。
- 通过*注意力汇聚*（attention pooling）来有偏向性的选择某些输入。



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



#### 多头注意力

https://zh-v2.d2l.ai/chapter_attention-mechanisms/multihead-attention.html

当**给定相同的查询、键和值的集合**时， 我们希望模型可以**基于相同的注意力机制学习到不同的行为**， 然后将不同的行为作为知识组合起来， **捕获序列内各种范围的依赖关系** （例如，短距离依赖和长距离依赖关系）。 因此，允许注意力机制组合使用查询、键和值的不同 *子空间表示*（representation subspaces）可能是有益的。

（即，对同一key，value，query，希望抽取不同的信息； 多头注意力使用 h 个独立的注意力池化）



用独立学习得到的 h 组不同的 ***线性投影*（linear projections）来变换查询、键和值**。 然后，这 h 组**变换后的查询、键和值将并行地送到注意力汇聚**中。 最后，将这 h个注意力汇聚的**输出拼接在一起**， 并且**通过另一个可以学习的线性投影进行变换， 以产生最终输出**。 这种设计被称为*多头注意力*（multihead attention）

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkbjtn1r3j30ws0jmjsy.jpg" alt="截屏2021-12-20 15.14.31" style="zoom:50%;" />



<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkbln7ewmj31hk0ii0we.jpg" alt="截屏2021-12-20 15.16.13" style="zoom:50%;" />

基于这种设计，每个头都可能会关注输入的不同部分， 可以表示比简单加权平均值更复杂的函数。





#### 自注意力

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkcpbx3orj31by06y0tk.jpg" alt="截屏2021-12-20 15.54.24" style="zoom:50%;" />



**对于输入序列中的每一个词，都作为一个query，key，value。来对序列 抽取特征 得到输出序列。**

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkcr44qi9j31ak0mgju7.jpg" alt="image-20211220155610133" style="zoom: 33%;" />



#### 位置编码





### 图神经网络



### 知识图谱

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

在每个训练批次中，通过忽略一半的特征检测器（让一半的隐层节点值为0），这种方式可以减少特征检测器（隐层节点）间的相互作用.

Dropout说的简单一点就是：我们在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征.

一小批训练样本执行完这个过程后，在没有被删除的神经元上按照随机梯度下降法更新对应的参数（w，b）。



#### 实现思路：（在训练和测试过程的区别）

Dropout 在训练时采用，是为了减少神经元对部分上层神经元的依赖，类似将多个不同网络结构的模型集成起来，减少过拟合的风险。

而在测试时，应该用整个训练好的模型，因此不需要dropout。



#### Dropout 如何平衡训练和测试时的差异

训练时还要对输出向量除以（1-p）之后再传给输出层神经元，作为神经元失活的补偿，以使得在训练时和测试时每一层输入有大致相同的期望。



#### 为什么Dropout可以解决过拟合？

（1）取平均的作用： 先回到标准的模型即没有dropout，我们用相同的训练数据去训练5个不同的神经网络，一般会得到5个不同的结果，此时我们可以采用 “5个结果取均值”或者“多数取胜的投票策略”去决定最终结果。这种“综合起来取平均”的策略通常可以有效防止过拟合问题。因为不同的网络可能产生不同的过拟合，取平均则有可能让一些“相反的”拟合互相抵消。**dropout掉不同的隐藏神经元就类似在训练不同的网络，随机删掉一半隐藏神经元导致网络结构已经不同，整个dropout过程就相当于对很多个不同的神经网络取平均。**而不同的网络产生不同的过拟合，一些互为“反向”的拟合相互抵消就可以达到整体上减少过拟合。

（2）减少神经元之间复杂的共适应关系： **因为dropout程序导致两个神经元不一定每次都在一个dropout网络中出现。这样权值的更新不再依赖于有固定关系的隐含节点的共同作用，阻止了某些特征仅仅在其它特定特征下才有效果的情况** 。迫使网络去学习更加鲁棒的特征 ，这些特征在其它的神经元的随机子集中也存在。**换句话说假如我们的神经网络是在做出某种预测，它不应该对一些特定的线索片段太过敏感，即使丢失特定的线索，它也应该可以从众多其它线索中学习一些共同的特征。**从这个角度看dropout就有点像L1，L2正则，减少权重使得网络对丢失特定神经元连接的鲁棒性提高。

（3）Dropout类似于性别在生物进化中的角色：物种为了生存往往会倾向于适应这种环境，环境突变则会导致物种难以做出及时反应，性别的出现可以繁衍出适应新环境的变种，有效的阻止过拟合，即避免环境改变时物种可能面临的灭绝。



### LayerNorm和BatchNorm的区别

Batch Normalization 的处理对象是一批样本， 对这批样本的同一维度特征做归一化。它去除了不同特征之间的大小关系，但是保留了不同样本间的大小关系，所以在CV领域用的多。

Layer Normalization 的处理对象是单个样本。 是对这单个样本的所有维度特征做归一化。它去除了不同样本间的大小关系，但是保留了一个样本内不同特征之间的大小关系，所以在NLP领域用的多。（在NLP任务中，**序列的长度大小是不相等的**）





所以，LN不依赖于batch的大小和输入sequence的深度，因此可以用于batchsize为1和RNN中对边长的输入sequence的normalize操作。



举例：

将输入的图像shape记为[N, C, H, W]，这几个方法主要的区别就是在，

- batchNorm是在batch上，对NHW做归一化，对小batchsize效果不好；
- layerNorm在通道方向上，对CHW归一化，主要对RNN作用明显

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkvbnaxs4j30ja0b0dh4.jpg" alt="截屏2021-12-21 02.38.38" style="zoom:50%;" />









### 梯度消失和梯度爆炸？改进方法。

解决梯度爆炸：

- 可以通过梯度截断

  <img src="https://tva1.sinaimg.cn/large/008i3skNly1gxl7uu2h0jj30rm08kdgb.jpg" alt="截屏2021-12-21 09.52.13" style="zoom:33%;" />

- 通过添加正则项。

解决梯度消失：

- 将RNN改掉，使用LSTM等自循环和门控制机制。
- 优化激活函数，如将sigmold改为relu
- 使用batchnorm
- 使用残差结构



### 残差网络的本质是什么？

其表达式是ouput = layer(x) + x，其中layer(x)是某层网络，x是输入。设计残差网络的根本目的是为了避免梯度消失。



### 数据不平衡怎么处理





### 已经训练完毕的模型，如何在测试的时候，进一步提升其性能





### 聚类如果不清楚有多少类，有什么方法？





### 卷积是什么样的过程？





### 正则化有什么用，为什么有用，L1正则为什么能使参数稀疏，为什么能防止过拟合

L1正则化是指权值向量ww中各个元素的绝对值之和，通常表示为||w||1||w||1
L2正则化是指权值向量ww中各个元素的平方和然后再求平方根（可以看到Ridge回归的L2正则化项有平方符号），通常表示为||w||2||w||2



L1正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择
L2正则化可以防止模型过拟合（overfitting）；一定程度上，L1也可以防止过拟合
加上正则项L1，在贝叶斯的角度上，等同于对θ假设一个先验分布为拉普拉斯分布
从贝叶斯的角度，L2相当于给θ一个先验分布为高斯分布



### Bert为什么只使用了Transformer的Encoder

那么Decoder去哪了呢？显然是被BERT改造了。Transformer其实是个完整地seq-to-seq模型，可以解决诸如机器翻译、生成式QA这种输入输出为不定长句子的任务，在Transformer中，它使用Encoder做特征提取器，然后用Decoder做解析，输出我们想要的结果。

而对于BERT，它作为一个预训练模型，它使用固定的任务——language modeling来对整个模型的参数进行训练，这个language modeling的任务就是masked language model，所以它是一个用上下文去推测中心词[MASK]的任务，故和Encoder-Decoder架构无关，它的输入输出不是句子，其输入是这句话的上下文单词，输出是[MASK]的softmax后的结果，最终计算Negative Log Likelihood Loss，并在一次次迭代中以此更新参数。

**所以说，BERT的预训练过程，其实就是将Transformer的Decoder拿掉，仅使用Encoder做特征抽取器，再使用抽取得到的“特征”做Masked language modeling的任务，通过这个任务进行参数的修正。**

当然了，BERT不仅仅做了MLM任务，还有Next Sequence Prediction，这个由于后序验证对模型的效果提升不明显，所以没有赘述。

注意：我们常说，xxx使用Transformer作为特征抽取器，这其实在说用Transformer的Encoder(主要是Self-Attention和短路连接等模块)做特征抽取器，和Decoder啥关系也没有



## 概念/计算

### tf-idf

词频-逆向文件频率

TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

TF-IDF的主要思想是：如果某个单词在一篇文章中出现的频率TF高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxcuf26vsnj30eo03wq33.jpg" alt="截屏2021-12-14 04.02.49" style="zoom:40%;" /> <img src="https://tva1.sinaimg.cn/large/008i3skNly1gxcufk2srvj30h003ot8y.jpg" alt="截屏2021-12-14 04.03.16" style="zoom:40%;" />



### 精确率、召回率、准确率、F1

**精确率**表示的是**预测为正的样本中有多少是真正的正样本**。那么预测为正就有两种可能了，一种就是把正类预测为正类(TP)，另一种就是把负类预测为正类(FP)，也就是

![[公式]](https://www.zhihu.com/equation?tex=P++%3D+%5Cfrac%7BTP%7D%7BTP%2BFP%7D)

**召回率**表示的是**样本中的正例有多少被预测正确**了。那也有两种可能，一种是把原来的正类预测成正类(TP)，另一种就是把原来的正类预测为负类(FN)。

![[公式]](https://www.zhihu.com/equation?tex=R+%3D+%5Cfrac%7BTP%7D%7BTP%2BFN%7D)

准确率的定义是**预测正确的结果占总样本的百分比**

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%E5%87%86%E7%A1%AE%E7%8E%87+%3D+%5Cfrac%7BTP%2BTN%7D%7BTP%2BTN%2BFP%2BFN%7D+%5C%5C+%5Cend%7Bequation%7D)



F-score相当于precision和recall的调和平均



$$
F1 = \frac{2PR}{P+R}
$$


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





##### 损失函数

https://zh-v2.d2l.ai/chapter_linear-networks/softmax-regression.html

https://zhuanlan.zhihu.com/p/25723112

https://blog.csdn.net/GreatXiang888/article/details/99293507

损失函数——交叉熵损失

![截屏2021-12-21 00.38.20](https://tva1.sinaimg.cn/large/008i3skNly1gxkruhf2r4j31jy07s3z2.jpg)



softmax及其导数

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkrwf36zcj30te05gjrl.jpg" alt="截屏2021-12-21 00.40.12" style="zoom:50%;" />

推导还是看[这个](https://blog.csdn.net/GreatXiang888/article/details/99293507)

对于，（这里ai是yi对应的softmax）
$$
L o s s=-\sum_{i} y_{i} \ln a_{i}
$$
<img src="https://pic2.zhimg.com/80/v2-d3a4e22a107052ee998823b24b49db71_1440w.jpg" alt="img" style="zoom:75%;" /> <img src="https://pic3.zhimg.com/80/v2-5eafb4c0a835bc90248766ac0c123dfe_1440w.jpg" alt="img" style="zoom:75%;" />





#### sigmoid

https://www.jianshu.com/p/c78af484559b
$$
f(z)=\frac{1}{1+e^{-z}}
$$

它能够把输入的连续实值变换为0和1之间的输出，特别的，如果是非常大的负数，那么输出就是0；如果是非常大的正数，输出就是1.

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkv1qipwcj312y0dodgi.jpg" alt="截屏2021-12-21 02.29.04" style="zoom:50%;" />

**缺点**

1. 容易导致梯度消失。每传递一层梯度值都会减小为原来的0.25倍，如果神经网络隐藏层很多，那么梯度在穿过多层后悔变得接近0，即出现梯度消失现象。

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

- Dead ReLU Problem，指某些神经元可能永远不会被激活，导致相应的参数永远不能被更新。

  原因：learning rate太高导致在训练过程中参数更新太大

- 原点不可导。解决办法：坐标轴下降法、最小角回归法



#### Leaky ReLu

$$
\text { LeakyReLU }(x)=\max (\alpha x, x)
$$

$\alpha$通常取0.01

对ReLu的一个改进，可以改善relu中x<0部分的dead问题。



### 优化器

#### GD（标准梯度下降）

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkwfuifwjj321209ywg5.jpg" alt="截屏2021-12-21 03.17.14" style="zoom:50%;" />

沿着梯度的方向不断减小模型参数，从而最小化代价函数。



缺点：

- 训练速度慢：每走一步都要要计算调整下一步的方向，并且每次迭代都要遍历所有的样本。会使得训练过程及其缓慢，需要花费很长时间才能得到收敛解。
- 容易陷入局部最优解：由于是在有限视距内寻找下山的方向。当陷入平坦的洼地，会误以为到达了山地的最低点，从而不会继续往下走。所谓的局部最优解就是鞍点。落入鞍点，梯度为0，使得模型参数不在继续更新。





#### SGD（随机梯度下降）

随机梯度下降（SGD）可降低每次迭代时的计算代价。在随机梯度下降的每次迭代中，我们对数据样本随机均匀采样一个样本，计算梯度。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkwls4etlj30ao02sdfn.jpg" alt="截屏2021-12-21 03.22.56" style="zoom:50%;" />

优点：

- 计算梯度快

缺点：

- SGD在随机选择梯度的同时会引入噪声，使得权值更新的方向不一定正确。
- SGD也没能单独克服局部最优解的问题。





#### BGD（批量梯度下降）

批量梯度下降法比标准梯度下降法训练时间短，且每次下降的方向都很正确。



#### Momentum

使用动量(Momentum)的随机梯度下降法(SGD)，主要思想是引入一个积攒历史梯度信息动量来加速SGD。

动量主要解决SGD的两个问题：

一是随机梯度的方法（引入的噪声）；

二是Hessian矩阵病态问题（可以理解为SGD在收敛过程中和正确梯度相比来回摆动比较大的问题）。



#### Adam



#### AdaGrad







## 项目

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxcq1xm58ej30tk0vwacu.jpg" alt="image-20211214013147436" style="zoom:50%;" />

文本预处理：

为了对文本进行预处理，我们通常将文本拆分为词元，构建词表将词元字符串映射为数字索引，并将文本数据转换为词元索引以供模型操作。





## 有什么要问的吗

发展现状，推荐算法的发展现状，算法岗的发展现状，nlp+推荐算法的组合



## 暂未归类

