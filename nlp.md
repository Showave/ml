# ChatGPT系列论文精读——大模型经典论文GPT1、GPT2、GPT3
> *https://zhuanlan.zhihu.com/p/626494749*

# [Transformer 101系列] 初探LLM基座模型

> https://zhuanlan.zhihu.com/p/640784855

## NLP任务速览

在深入介绍LLM网络结构之前，我们先简单了解一下NLP (Natural Language Processing)都包含了哪些任务。主要包含3大类任务

- 自然语言理解任务(NLU, Natural Language Understanding)。特点是能看到完整上下文信息，然后做广义分类任务，典型任务如文本情感分析，词性标注，信息检索等。
- 有条件自然语言生成任务(conditioned-NLG, Natural Language Generation)。特点是seq2seq，典型任务例如机器翻译，自动摘要等。
- 无条件自然语言生成任务(unconditioned-NLG)。特点是开放性的句子生成，典型任务如问答系统(QA)、对话机器人(ChatBot)等。

## LLM演变

- Encoder-only框架(也叫**Auto-Encoder**)，典型代表如**BERT**等
- **Encoder-decoder**框架，典型代表如**T5和GLM**等
- Decoder-only框架(也叫**Auto-Regressive**)，典型代表如**GPT系列/LLaMa/PaLM**等

## LayerNorm

BERT当时使用的是Post-Norm的结构，同时期的GP1也是用该结构，后来的GPT2使用Pre-Norm。

- Post-Norm会削弱残差的作用，深度保持，但是收敛和调参困难
- Pre-Norm会将网络变成浅且宽的结构，收敛容易，但是精度会有一定损失

## GeLU激活函数
GeLU (Gaussian Error Linear Unit), 高斯误差线性单元,出发点是受到了RELU和dropout的启发

- RELU是激活小的时候乘以0
- dropout是随机乘以0
- GeLU就是概率性的乘以0 (但是跟dropout不同，用确定性的表达式给出)

## Tokenization
早期有word-base和character-base两种，但是

- word-base，单词种类太多，单词本会太大
- character-base，序列太长，单词本字母没有语义


于是出现了trade-off的方法，就是sub-word base，拆分成sub-word的原则是

- 把不频繁出现的单次拆成更加频繁出现的部分
- 不要把频繁的单词拆开成若干部分
  
Byte-Pair Encoding (BPE)是当前SOTA的LM模型常用的分词方法
从字母开始，统计频率，生成最初的单词表，并且把单词的结尾用分割开
统计两两字母，把出现频率最高的组成pair然后添加到词表中，同时更新原来单个字母的频率
依次类推，直到词表数量达到预设值
详细推导过程参见[here](https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0)

BERT使用的是Wordpiece，属于BPE的改进版，都是sub-word分词器.

此外BERT使用了长度为30k的单词本，每个token是长度为D的可学习向量

## Position Encoding

原始transformer(attention is all you need)里面用的是三角式位置编码

BERT使用的是可学习的位置编码，预设的位置个数是512，因此最大序列长度为512

## BERT训练 Pre-training & Fine-Tuning
在无监督Pre-training的时候用了两种任务:
- Masked LM任务，遮住部分词，让网络看到上下文预测这个词。
- Next Sentence Prediction任务，判断两个句子是否为紧挨着的两句话。

在Finetune阶段复用预训练的网络权重

- 分类的话可以用[CLS]的向量接softmax层做监督
- 更复杂的任务可以用对应单词输出的向量接softmax做监督

## BERT短板
BERT训练简单，但有以下两个短板

- 短板1：对连续的Mask Token处理的不好
- 短板2：没法直接用于做变长的文字生成的任务

## Encoder-Decoder
- decoder第一个MHA变成masked-MHA，使用的是前文casual的attention mask的方式，这样每个当前输出token只能看到过去生成的token
- decoder新增第二个MHA，并且K和V来自于encoder的输出，这样就实现了看到原始输入的全文

## GLM-130B
GLM(General Language Model)是清华提出的基座模型，属于Prefix LM方式。作者说出发点是

- 希望能同时在3种NLP任务上达到最优
- 不想引入原始encoder-decoder成倍的计算量代价

该论文出发点是改进BERT，想解决BERT的两个短板:

- 短板1：对连续的Mask Token处理的不好 → 干脆把连续的Mask Tokens合并成一个token [M]
- 短板2：没法直接用于做变长的文字生成的任务 → 对[M]位置进行任意长度的展开
  
于是得到解体思路为，先用双向网络encoder对题干(prompt)审题，然后通过decoder-only的方式把题干中 
位置做展开作答。最终的网络形式很像权值共享版本的encoder-decoder，这样计算量也降下来了。

- 从seq A里面采样出若干组连续的token，设置成Mask字符[M]
- 把seq A中所有[M]位置的token顺序打乱，并且添加前缀[S],形成seq B
- 把seq A和seq B连接起来，seq A内部attention是双向的。seq B是单向的，能看到seq A的信息，但是看不到seq B后面的信息
- **正确的标签**来自于原始文本里面的**下一个token**，注意每组的结尾要求输出[E]字符，代表当前组终止
- 位置编码采用的是2个层次编码
    - Position 1代表字符在原始文本中的位置下标
    - Position 2代表组内的相对偏移，对seq A而言默认是0
- 此外根据Mask token的数量多少可以自由设置单词(MASK)，句子(sMASK)，文档(gMASK)三种MASK方式

下游任务要finetune的时候:
- 如果是分类任务，那么添加模板句子，例如感情分类用It is really [M]，seq B对[M]位置做分类token预测
- 如果是生成任务，那么直接在seq A最后pad上[M]即可，seq B对[M]位置做续写，达到生成文本的目的

GLM-130B是比较晚出现的模型，用了比较新技术

<!-- - 使用了$\color{red}{Post-Deep-Norm}$的归一化方法 -->
- 使用了<font color=red>**Post-Deep-Norm**</font>的归一化方法
- 使用了**GeGLU**的激活函数
  
## GeGLU激活函数
GeGLU激活函数，由GeLU和GLU两部分组成。其中GLU(Gated Linear Unit)是双线性函数.

逐元素乘法。可见第一个是用sigmoid激活，第二个是线性，于是GeGLU就是把第一个sigmoid换成GeLU

## Decoder-only
奠基性工作GPT1/GPT2/GPT3中的transformer结构有啥变化。很遗憾其实变化很少，主要就是从Post-Norm转到Pre-Norm，最后加了一个LayerNorm输出.

GPT1/2/3更多探究的是如何更好的达到生成的效果。GPT2尝试用zero-shot解决问题，但发现实在太难了，于是GPT3开始转向用few-shot来解决问题

## LLaMA
出发点如下

- 只使用公开的数据集
- 用更多的数据训练更小网络，例如用1T的token训练7B的模型和13B模型，用1.4T的token训练33B和65B模型。这一点是参考了Chinchilla的结论。

网络结构也是decoder-only的方式，跟GPT3相比异同如下

- 使用了SentencePiece实现的PBE的编码方式
- 使用了PreNorm，这样收敛稳定一些。同时用RMSNorm，就是LayerNorm里面没有减均值项和beta项
- 使用SwiGLU，即swish激活+GeLU调制。由于SwiGLU引入了额外的参数矩阵，原始FFN需要做相应的砍小
- 用了苏剑林老师提出的RoPE旋转位置编码，核心思想是通过绝对位置编码的方式实现相对位置编码，理论推导见[link](https://zhuanlan.zhihu.com/p/359502624)

## RMSNorm
RMSNorm是本文要介绍的第3种norm，其中RMS(root mean square)是均方根的含义
[![\\ RMS(x) = \sqrt{{1 \over d }\sum_{i=1}^{d}x_i^2}](https://latex.codecogs.com/svg.latex?%5C%5C%20RMS(x)%20%3D%20%5Csqrt%7B%7B1%20%5Cover%20d%20%7D%5Csum_%7Bi%3D1%7D%5E%7Bd%7Dx_i%5E2%7D)](#_)

## 总结
本文主要介绍了LLM基座模型里常见的3种transformer架构，encoder-only，encoder-decoder和decoder-only。提及的模型组件包括

- Norm位置3种: Post-Norm，Pre-Norm和Sandwich-Norm
- Norm方法3种: LayerNorm, DeepNorm和RMSNorm
- 激活函数3种: GeLU, GeGLU和SwiGLU
- PE方法6种: Fixed Absolute, Learned Absolute, Fixed Relative, Learned Relative, RoPE, ALiBi
