[TOC]



## 模型

### Word2vec

### 两种优化方法：分层softmax、负采样

#### 简介

https://zhuanlan.zhihu.com/p/26306795

Word2vec 本质上是一种**降维**操作——把词语从 one-hot encoder 形式的表示降维到 Word2vec 形式的表示。



当模型训练完后，最后得到的其实是**神经网络的权重**，比如现在输入一个 x 的 one-hot encoder: [1,0,0,…,0]，则在输入层到隐含层的权重里，只有对应 1 这个位置的权重被激活，这些权重的个数，跟隐含层节点数是一致的，从而这些权重组成一个向量 vx 来表示x，而因为每个词语的 one-hot encoder 里面 1 的位置是不同的，所以，这个向量 vx 就可以用来唯一表示 x。



------

对于这些模型，每个单词存在两类向量表达：输入向量![[公式]](https://www.zhihu.com/equation?tex=v_%7Bw%7D%5E%7B%7D)，输出向量![[公式]](https://www.zhihu.com/equation?tex=v_%7Bw%7D%5E%7B%27%7D)（这也是为什么word2vec的名称由来：1个单词对应2个向量表示）。学习得到输入向量比较简单；但要学习输出向量是很困难的。为了更新![[公式]](https://www.zhihu.com/equation?tex=v_%7Bw%7D%5E%7B%27%7D)，在每次训练中，我们必须遍历词汇表中的每个单词![[公式]](https://www.zhihu.com/equation?tex=w_%7Bj%7D%5E%7B%7D)，从而计算得到 ![[公式]](https://www.zhihu.com/equation?tex=u_%7Bj%7D%5E%7B%7D)，预测概率![[公式]](https://www.zhihu.com/equation?tex=y_%7Bj%7D%5E%7B%7D)（skip-gram为![[公式]](https://www.zhihu.com/equation?tex=y_%7Bc%2Cj%7D%5E%7B%7D)），它们的预测误差![[公式]](https://www.zhihu.com/equation?tex=e_%7Bj%7D%5E%7B%7D)，（skip-gram为![[公式]](https://www.zhihu.com/equation?tex=EI_%7Bj%7D%5E%7B%7D)），然后再用误测误差来更新输出向量![[公式]](https://www.zhihu.com/equation?tex=v_%7Bj%7D%5E%7B%27%7D)。

对每个训练过程做如此庞大的计算是非常昂贵的，使得它难以扩展到词汇表或者训练样本很大的任务中去。为了解决这个问题，我们直观的想法就是**限制每次必须更新的输出向量的数量**。一种有效的手段就是采用**分层softmax**；另一种可行的方法是通过**负采样**。



#### hierarchical softmax

本质是把 N 分类问题变成 log(N)次二分类

用二叉树来表示词汇表中的所有单词。V个单词必须存储于二叉树的叶子单元。可以被证明一共有V-1个内单元。对于每个叶子节点，有一条唯一的路径可以从根节点到达该叶子节点；该路径被用来计算该叶子结点所代表的单词的概率。

分层softmax模型**没有单词的输出向量**，取而代之的是， ![[公式]](https://www.zhihu.com/equation?tex=V-1) 中每个隐节点都有一个输出向量 ![[公式]](https://www.zhihu.com/equation?tex=v_%7Bn%28w%2Cj%29%7D%5E%7B%27%7D) 。一个单词作为输出词的概率被定义为：

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxd7egmc5zj3164062mxf.jpg" alt="截屏2021-12-14 11.31.58" style="zoom: 33%;" />

从根节点出发到叶子结点的随机路径。在每个隐节点（包含根节点），我们需要分配往左走或往右走的概率。它是由隐节点向量和隐藏层输出值（ ![[公式]](https://www.zhihu.com/equation?tex=h) ，也就是输入单词的向量表示)共同决定。

训练模型的计算复杂度从 ![[公式]](https://www.zhihu.com/equation?tex=O%28V%29) 降至 ![[公式]](https://www.zhihu.com/equation?tex=O%28logV%29) ，这在效率上是一个巨大的提升。而且我们仍然有差不多同样的模型参数（原始模型： ![[公式]](https://www.zhihu.com/equation?tex=V) 个单词的输出向量，分层softmax： ![[公式]](https://www.zhihu.com/equation?tex=V-1) 个隐节点的输出向量)。



#### negative sampling

本质是预测总体类别的一个子集

负采样的思想更加直观：为了解决数量太过庞大的输出向量的更新问题，我们就不更新全部向量，而只更新他们的一些样本。



显然正确的输出单词（也就是正样本）应该出现在我们的样本中，另外，我们需要采集几个单词作为负样本（因此该技术被称为“负采样”）。采样的过程需要指定总体的概率分布，我们可以任意选择一个分布。我们把这个分布叫做噪声分布，标记为 ![[公式]](https://www.zhihu.com/equation?tex=P_%7Bn%7D%28w%29) 。可以凭经验选择一个好的分布。

![[公式]](https://www.zhihu.com/equation?tex=P_%7Bn%7D%28w%29) 中采样得到的单词集合，也就是负样本。![[公式]](https://www.zhihu.com/equation?tex=t_%7Bj%7D)是单词![[公式]](https://www.zhihu.com/equation?tex=w_%7Bj%7D)的标签。t=1时， ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bj%7D)是正样本；t=0时，![[公式]](https://www.zhihu.com/equation?tex=w_%7Bj%7D)为负样本。

只需要将此公式作用于 ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bj%7D%5Cin%5Cleft%5C%7B+w_%7BO%7D+%5Cright%5C%7D%5Ccup+W_%7Bneg%7D) 而不用更新词汇表的所有单词。这也解释了为什们我们可以在一次迭代中节省巨大的计算量。



相关公式：

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxd7yfxhgzj314g07ijrj.jpg" alt="截屏2021-12-14 11.51.10" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxd7yvbhzbj313c0fk0tv.jpg" alt="截屏2021-12-14 11.51.35" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxd81lkuphj313o09g74v.jpg" alt="截屏2021-12-14 11.54.12" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxd82koogxj313a0bqaap.jpg" alt="截屏2021-12-14 11.55.10" style="zoom: 33%;" />

### Seq2Seq

#### 简介

Encoder-Decoder架构

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkandj3v9j31240eqdh7.jpg" alt="截屏2021-12-20 14.43.17" style="zoom:50%;" />

- 编码器是一个RNN，读取句子输入
  - 可以是双向
- 解码器使用另外一个RNN来输入
  - 必须是单向

循环神经网络编码器使用长度可变的序列作为输入， 将其转换为固定形状的隐状态。

为了连续生成输出序列的词元， 独立的循环神经网络解码器是基于 **输入序列的编码信息** 和 **输出序列已经看见的或者生成的词元** 来预测下一个词元。



**使用循环神经网络编码器最终的隐状态来初始化解码器的隐状态**



#### 训练

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





### ⚠️Transformer

https://zh-v2.d2l.ai/chapter_attention-mechanisms/transformer.html

Transformer是一个纯使用注意力的 编码-解码器

编码器和解码器都有n个 transformer 块

每个块里使用多头注意力，基于位置的前馈网络，和层归一化。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkdudis5aj30s20yewhe.jpg" alt="截屏2021-12-20 16.33.51" style="zoom:40%;" />

从宏观角度来看，transformer 的编码器是由多个相同的层叠加而成的，每个层都有两个子层。第一个子层是*多头自注意力*（multi-head self-attention）汇聚；第二个子层是*基于位置的前馈网络*（positionwise feed-forward network）。具体来说，在计算编码器的自注意力时，查询、键和值都来自前一个编码器层的输出。受 [7.6节](https://zh-v2.d2l.ai/chapter_convolutional-modern/resnet.html#sec-resnet)中残差网络的启发，每个子层都采用了*残差连接*（residual connection）。在残差连接的加法计算之后，紧接着应用 LayerNorm 。因此，输入序列对应的每个位置，transformer编码器都将输出一个d维表示向量。

Transformer解码器也是由多个相同的层叠加而成的，并且层中使用了残差连接和LayerNorm。除了编码器中描述的两个子层之外，解码器还在这两个子层之间插入了第三个子层，称为*编码器－解码器注意力*（encoder-decoder attention）层。在编码器－解码器注意力中，查询来自前一个解码器层的输出，而键和值来自整个编码器的输出。在解码器自注意力中，查询、键和值都来自上一个解码器层的输出。但是，解码器中的<u>每个位置只能考虑该位置之前的所有位置</u>。这种*掩蔽*（masked）注意力保留了*自回归*（auto-regressive）属性，<u>确保预测仅依赖于已生成的输出词元</u>。



### Bert

https://zh-v2.d2l.ai/chapter_natural-language-processing-pretraining/bert.html

#### 背景

NLP里的迁移学习

- 使用与训练好的模型来抽取词、句子的特征。例如word2vec或语言模型
- 不更新预训练好的模型
- 需要构建新的网络来抓取新任务需要的信息。例如，word2vec忽略了时序信息；语言模型只看了一个方向。

#### 简介

动机：

- 基于微调的NLP模型
- 预训练的模型抽取了足够多的信息
- 新的任务只需要增加一个简单的输出层

（比如图像分类任务，输入一个图像，经过特征抽取，最后加一个分类器就行了。底层的这些特征抽取的权重是可以复用的，可以挪到别的任务去。训练的时候只需要训练最后小的网络的权重就行。）

Bert就是，想把NLP的训练做的和CV差不多，预训练模型抽取了足够多的信息。足够抓住语义信息。新的任务只需要增加简单的输出层就行了。不需要再加什么RNN，Transformer等网络。

**只要把特征转换成语义label的空间就行了。**



输入输出？

BERT输入序列明确地表示单个文本和文本对。

当输入为单个文本时，BERT输入序列是特殊类别词元“\<cls>”、文本序列的标记、以及特殊分隔词元“\<sep>”的连结。

当输入为文本对时，BERT输入序列是“\<cls>”、第一个文本序列的标记、“\<sep>”、第二个文本序列标记、以及“\<sep>”的连结。



- 每个样本是一个句子对（两个句子拼起来放到encoder里）。用特定的分隔符分开\<sep>
- 加入额外的片段嵌入。 给第一个句子 id为0，第二个句子id是1
- 位置编码可以学习。



模型输入：通过查询字向量表将文本中的每个字转换为一维向量

模型输出：输入各字对应的融合全文语义信息后的向量表示。（抽取了上下文信息的特征向量）



#### 特点

1. 从上下文无关到上下文敏感
2. 从特定任务到不可知任务

ELMo对上下文进行双向编码，但使用特定于任务的架构；而GPT是任务无关的，但是从左到右编码上下文。

BERT（来自Transformers的双向编码器表示）结合了这两个方面的优点。它对上下文进行双向编码，并且对于大多数的NLP任务只需要最少的架构改变。通过使用预训练的Transformer编码器，BERT能够基于其双向上下文表示任何词元。



架构：

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



### SVM





### XGboost

XGBoost的特点，缺失值如何处理





### lightgbm





### gbdt







### LR



## 大类

### Attention

（下面的随意是指 follow your heart，而不是随便）

心理学认为人通过随意线索（自主性）和不随意线索（不自主的）选择注意点。

参考人类注意力的方式。非自主性提示下，选择偏向于感官输入。比如卷积、全连接层、池化层都只考虑不自主性的线索。

注意力机制则显式地考虑随意性（有自主）的线索。

- 随意线索被称为查询 query
- 每个输入是一个 key-value对。key表示不随意性的线索，value表示它的值。
- 通过*注意力汇聚*（attention pooling）来有偏向性的选择某些输入。

**注意力机制通过 注意力汇聚 将*查询*（自主性提示）和*键*（非自主性提示）结合在一起，实现对*值*（感官输入）的选择倾向**

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

当给定相同的查询、键和值的集合时， 我们希望模型可以**基于相同的注意力机制学习到不同的行为**， 然后将不同的行为作为知识组合起来， 捕获序列内各种范围的依赖关系 （例如，短距离依赖和长距离依赖关系）。 因此，允许注意力机制组合使用查询、键和值的不同 *子空间表示*（representation subspaces）可能是有益的。

（即，对同一key，value，query，希望抽取不同的信息； 多头注意力使用 h 个独立的注意力池化）



用独立学习得到的 h 组不同的 *线性投影*（linear projections）来变换查询、键和值。 然后，这 h 组变换后的查询、键和值将并行地送到注意力汇聚中。 最后，将这 h个注意力汇聚的输出拼接在一起， 并且通过另一个可以学习的线性投影进行变换， 以产生最终输出。 这种设计被称为*多头注意力*（multihead attention）

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkbjtn1r3j30ws0jmjsy.jpg" alt="截屏2021-12-20 15.14.31" style="zoom:50%;" />



<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkbln7ewmj31hk0ii0we.jpg" alt="截屏2021-12-20 15.16.13" style="zoom:50%;" />

基于这种设计，每个头都可能会关注输入的不同部分， 可以表示比简单加权平均值更复杂的函数。





#### 自注意力

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkcpbx3orj31by06y0tk.jpg" alt="截屏2021-12-20 15.54.24" style="zoom:50%;" />



**对于输入序列中的每一个词，都作为一个query，key，value。来对序列抽取特征得到输出序列。**

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxkcr44qi9j31ak0mgju7.jpg" alt="image-20211220155610133" style="zoom: 33%;" />



#### 位置编码





### 图神经网络



### 知识图谱





## 一般提问

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



BN、LN可以看作横向和纵向的区别。

经过归一化再输入激活函数，得到的值大部分会落入非线性函数的线性区，导数远离导数饱和区，避免了梯度消失，这样来加速训练收敛过程。

LayerNorm这类归一化技术，目的就是让每一层的分布稳定下来，让后面的层可以在前面层的基础上安心学习知识。

BatchNorm就是通过对batch size这个维度归一化来让分布稳定下来。LayerNorm则是通过对Hidden size这个维度归一。





### 数据不平衡怎么处理







### 已经训练完毕的模型，如何在测试的时候，进一步提升其性能





### 聚类如果不清楚有多少类，有什么方法？





### 卷积是什么样的过程？



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


$$
F1 = \frac{2PR}{P+R}
$$





## 项目

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxcq1xm58ej30tk0vwacu.jpg" alt="image-20211214013147436" style="zoom:50%;" />





## 暂未归类

