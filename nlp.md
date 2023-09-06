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

# 通过个性化prompt结合语言建模与用户行为——基于LLM的零样本推荐系统

> *https://zhuanlan.zhihu.com/p/637856419*

## 基于推荐任务的预训练语言模型（P5）

“Pretrain, Personalized Prompt, and Predict Paradigm” (P5)，即：预训练、个性化Prompt以及预测范式。P5提出了一种统一的文本到文本（text-to-text）的推荐系统构建范式。他们的框架结合了五种类型的推荐任务：序列推荐、评分预测、推荐理由、摘要、直接推荐。

P5基于prompt的自然语言格式构造任务，并使用统一的seq-to-seq框架一起学习这些相关的推荐任务。使用自适应个性化prompt模板将特征喂给模型。

P5使用预训练T5模型的checkpoints作为基础模型。

共享任务框架使知识泛化成为可能，这使得P5可以通过零样本或者少样本的方式使用全新的下游prompt，并成为各种不可见下游推荐任务的基础模型。

## Fine-Tuning预训练语言模型（M6-Rec）

为了训练模型，他们增加了可以忽略不计的（1%）特定任务参数，而没有对原始M6 Transformer模型进行任何修改。下图展示了M6的架构，该结构将句子输入到双向和自回归区域，并计算自回归区域输出的自回归损失。

##  Prompt-Tuning预训练语言模型

1、首先，使用基于规则的系统来创建个性化prompt，如下所示。

A user watched One Flew Over the Cuckoo’s Nest, James and the Giant Peach, and My Fair Lady. Now the user wants to watch [MASK].

2、接着，使用GPT-2通过多token推理来估计下一个商品的概率分布；使用交叉熵损失最大化商品概率；使用MRR@K和Recall@K矩阵对top K个待预估商品进行评估。

## 零样本下一商品推荐（NIR）

提出了一种在零样本设置下执行下一商品推荐的prompt策略。

### 三步骤GPT-3 prompting

这个步骤使用GPT-3和三个prompt。

1）在第一个子任务（用户偏好子任务）中，基于用户历史观看记录中的商品，设计用户偏好prompt，并请求GPT-3以总结用户的偏好。

2）在第二个子任务中，作者设计了一个结合用户偏好prompt、prompt答案以及触发指令的prompt，请求GPT-3，按照偏好降序，选择出代表性的电影。

3）在第三个子任务中，作者整合了代表性电影prompt、它的答案以及一个问题，用于构建推荐prompt，引导GPT-3从候选集中推荐10部与代表性电影最相似的电影。

## Chat-REC

Chat-GPT增强推荐系统（Chat-REC）使用LLM来增强他们的对话推荐系统。他们首先使用用户profile（如年龄、性别、地点、兴趣等）、历史互动（如点击、购买或评分的商品）以及历史对话记录（可选）创建prompt。然后请求LLM总结用户偏好。接下来，使用传统的推荐系统来生成一系列候选商品；最后，使用LLM缩小候选集合，以生成最终的推荐集合。

# GPT4Rec解读：用于个性化推荐和用户兴趣解释的生成式框架——GPT在推荐系统中的应用

现在的推荐模型通常会用ID来指代商品item，并使用判别式模型。这种模式存在以下不足：

1）未能充分利用items的内容信息和NLP模型的语言建模能力；

2）无法解释用户兴趣，以提升相关性和多样性；

3）实际使用时存在局限，比如新商品的冷启动。

### 简单步骤：

1、首先根据用户历史记录中的item标题，声称假设的“搜索queries”；

2、通过搜索这些queries，检索需要推荐的item。

## GPT4Rec的结构

1、给定用户的商品交互序列，GPT4Rec通过提示（Prompt）对商品标题进行格式化，并使用生成语言模型（GPT-2）学习语言空间中的商品和用户embedding。

2、模型会生成表示用户兴趣的多个query（查询），这些query将被提供给搜索引擎（BM25），以检索待推荐的商品。

##  训练策略
采取两步骤训练方式，分别优化语言模型和搜索引擎。给定每个用户的商品交互序列，**根据prompt取前T-1个商品及其标题，然后将其与最后一个商品 的标题拼接起来组成训练语料库，对GPT-2模型进行微调**。其根本思想是，最细粒度和准确的搜索query就是目标商品本身。这在推荐系统的应用中很常见，也很好理解，根据用户前T-1个商品的行为，预测其对第T个商品的行为。即，已知用户在购买了前T-1个商品后，是否可以预估用户对第T个商品的购买决策。

# ChatGPT系列论文精读——大模型经典论文GPT1、GPT2、GPT3

## GPT1 论文：《Improving Language Understanding by Generative Pre-Training》

### 2.1 无监督预训练

给定一个无标注样本库的token序列集合，语言模型的目标就是最大化似然值。也就是通过前面的tokens，预测下一个token。其中，k是滑动窗口的大小，P是条件概率。模型参数使用SGD进行优化。GPT-1使用了12层Transformer decoder结构。
每个token会通过transformer block被编码，最后再经过一个线性层+softmax，得到下一个token的预测分布。

### 2.2 有监督微调

得到无监督的预训练模型后，将得到的参数值直接应用于有监督任务中。对于一个有标签的数据集C，每个实例有m个输入tokens,将这些tokens输入到预训练模型中，得到transformer block的输出向量,再经过一个全连接+softmax，得到预测结果y.

有监督学习的目标即最大化上述概率

作者发现，把预训练目标作为辅助目标加入下游任务loss中，将会提高有监督模型的泛化性能，并加速收敛

### 2.3 特定任务的输入变换

对于文本分类任务而言，可以直接对预训练模型进行fine-tune。由于我们的预训练模型是在连续的文本序列上训练的，对于某些具有结构化输入（比如有序的句子对，或是文档、问题、答案的三元组等）的task，需要我们对输入进行一些修改。文章中，将结构化输入转换成预训练模型可以处理的有序序列。

### 3.2 无监督预训练

12层的transformer decoder，12个注意力heads ，768维

位置编码3072维

Adam优化器，最大学习率为2.5e-4

token序列长度是512，100个epochs

激活函数使用GELU

正则化手段：残差网络、dropout，drop比例是0.1

### 3.3 有监督微调

dropout比例0.1

学习率6.25e-5，batchsize是32，训练epoch为3

## GPT2 论文：《Language Models are Unsupervised Multitask Learners》

### 核心思想

训练一个通用的预训练模型，使下游任务无需手动生成或标记训练数据集，更无需更改预训练模型的参数或结构。

简单理解就是，在已知输入的条件下，预估输出的条件概率分布：p(output | input)。对于下游不同task的有监督任务来说，它可以建模成：p(output | input, task)。比如：

- 翻译训练样本可以写成：(translate to french, english text, french text)
- 阅读理解训练样本可以写成：(answer the question, document, question, answer)

这其实就是Prompting方法。作者认为，**监督目标与无监督目标相同，因此无监督目标的全局最小值也是有监督目标的全局最小值**。换言之，**任意监督任务均是无监督语言模型的子集**，只要语言模型的容量足够大，训练集足够丰富，仅仅依赖语言模型的学习，便可以同时完成其他有监督任务的学习。作者称之为“无监督多目标学习”。

### 2.2 数据集

训练数据：为了实现多任务无监督的zero-shot，我们希望数据集越大越丰富越多样越好。作者调研了如下几个数据集：

1、类似于Common Crawl的网页爬虫，多样性好且文本趋近无限，虽然比当前语言模型的数据集大很多数量级，但存在严重的数据质量问题。

2、自建了一个强调文档质量的WebText数据集。具体的，媒体平台Reddit爬到了所有被用户分享出站的链接，并限制每个链接的内容至少获得三个Karma（类似点赞收藏等正反馈）。作者认为，被用户分享的链接，往往是人们感兴趣的、有用的或者有趣的内容。

3、生成的WebText数据集是上述4500万链接的文本子集。对其进行后处理：1）剔除2017年12月之前创建的链接；2）去重；3）启发式清理；4）移除了所有维基百科文档，考虑到它是其他数据集的通用数据源，避免与下游的测试评估数据重叠。最终生成800万个文档，40GB的文本。

### 2.3 模型

模型这部分跟GPT-1相比区别不大，在GPT-1的基础上做了几个改进。

字典大小为50257；

滑动窗口的token大小从512 -> 1024；

batchsize的大小是512；

Layer normalization移动到了每个block的输入端，类似于预激活残差网络；在最终的self-attention之后增加了一个LN层；

对残差层的初始化值进行1/sqrt(N)缩放，N是残差层的个数。

### 总结

GPT-2的最大贡献就是验证了**通过海量数据和大量参数训练出来的语言模型，可以迁移到下游其他任务，无需额外训练和fine-tune**。通过Prompting方法调动了zero-shot学习的能力。但大量实验表明，GPT-2的无监督学习能力还有很大的提升空间，GPT-2在很多不同的task中，有些表现良好，但在某些领域仍然表现欠佳，比如summarization、QA等。作者提出，**模型容量仍然是无监督模型在问答领域表现不佳的主要原因**。随着模型参数量的增加，GPT-2的效果也会稳步提升，这就有了未来GPT-3的大力出奇迹。

## GPT3 论文：《Language Models are Few-Shot Learners》

GPT-3引入了In-Context learning的概念，在很多NLP数据集上都取得了非常好的效果，包括翻译、问答、完形填空，以及一些需要实时推理和领域适应的任务（如整理单词、在句子中应用新单词、计算三位数的算术等复杂任务）。

### 背景
在BERT诞生后，虽然预训练+微调的范式取得了很多惊人效果，但是Fine-tuning（微调）仍然是存在诸多局限的：

1、fine-tune是针对特定数据集和特定任务的，若希望达到较强的性能，则需要基于成千上万甚至数十万的数据集上进行微调。

2、为了收集足够信息，预训练模型的参数量往往非常大，然后在较小的子任务上进行微调。在这种范式下，模型的泛化性能会比较差，它只是拟合了特定的训练分布。fine-tune在某些小任务上表现较好，可能只是一种过拟合的表现；这些下游任务的表现可能被放大了。

3、人类在学习了大量知识之后（对应于模型的预训练），通常不需要大量的监督数据集来学习新任务（对应于下游微调），只需要少量的提示和指令（对应于prompting）即可

下图很好地解释了这三种方法与传统Fine-tuning方法的区别。Fine-tuning通过训练针对下游特定任务的有监督数据集合，对预训练模型进行梯度下降和权重更新。而其他三种方法是不涉及任何梯度和权重更新的。

### 方案

#### 3.1 few-shot，one-shot，zero-shot learning

Few-Shot指的是在推理过程中，给模型一些任务演示,Few-shot的最主要优点就是大大减少了对特定任务数据的需求，降低了从大且窄的微调数据集中学习到一个过窄分布的可能性。缺点就是，这个方法仍然远远落后于目前最先进的fine-tune模型。

One-Shot在推理过程中，只会给模型一个演示，其余同Few-Shot相同。

Zero-Shot在推理过程中，不会给模型任何演示，只会提供对任务的自然语言描述。这个方法非常便利、鲁棒性强、避免了预训练-微调的虚假相关性.

#### 3.2 模型结构

GPT-3沿用了GPT-2的结构，训练过程也与GPT-2类似。主要区别在于扩展了GPT-2的模型大小、数据集大小及多样性。在初始化、归一化以及tokenization等方面做了一些改进，同时借鉴了Sparse Transformer的优化点。

## GPT1-3 总结
GPT系列从1到3，使用的底层结构均依托于transformer的解码器，在模型结构方面没有太大的创新和改进，主要靠不断增加模型容量和超强的算力。GPT-2的主要卖点是zero-shot，GPT-3的主要卖点是few-shot-learning（in-context learning）。

尽管和GPT-2相比，GPT-3的性能非常出色，在质量和参数量方面都有了明显的改进，但它仍然存在一些局限性：

1、GPT-3在文本合成和几个NLP任务上，仍然存在显著的弱点。比如文本合成任务，尽管总体质量很高，但GPT-3会在较长的段落中失去连贯性、自相矛盾，偶尔还会生成不合逻辑的句子和段落。GPT系列均基于自回归语言模型结构（单向），这个结构的采样和计算都相对简单。GPT-3相关实验均不包含任何双向结构或其他训练目标（如去噪）。这个结构限制了GPT-3的能力。作者推测，大型双向模型在微调方面比GPT-3更强。**文章提出，基于GPT-3的规模、或者尝试使用few-shot/zero-shot learning训练一个双向语言模型，是未来一个非常有前景的方向**，有助于实现“两全其美”。

2、GPT-3和其他类似LM的模型（无论是自回归模型还是双向模型）一样，它最**终可能会落入预训练目标的极限**。简言之，**大型语言模型缺乏与真实世界的交互，因此缺乏有关世界的上下文信息**。自监督训练可能会达到一个极限，因此有必要使用不同的方法进行增强。作者提出，未来有前景的方向可能包含：**从人类那里学习目标函数、通过强化学习进行微调、增加不同的模态（比如图像）等。前两者就是未来instruct-GPT的工作了。**

3、语言模型普遍存在的一个局限性是，**在预训练阶段，样本的利用效率低**。尽管GPT-3在测试阶段，朝着更加接近人类的测试时间迈进了一步（one-shot或zero-shot），但它在预训练阶段看到的文本仍然比人类一生中看到的文本还要多得多。

4、GPT-3的规模巨大，无论目标函数或是算法如何，GPT-3的训练昂贵，且不方便进行推理。GPT-3这样的大型模型往往都包含了非常多的技能，但大多数都不是特定任务所需要的。**作者提出，解决上述问题的一个未来方向可能是对大型模型进行蒸馏，以期达到特定任务的可管理规模。**业界已有针对蒸馏的大量探索，但尚未在千亿参数的规模下进行过尝试。

5、GPT-3与大多数深度学习系统有一些共同的局限性——**不可解释性。很难保证GPT-3生成的一些文章不包含敏感词汇和话题**——比如宗教偏见、种族歧视以及性别偏见等。

# query-doc蒸馏优化

> *https://cloud.tencent.com/developer/article/1951422?from=article.detail.1643116*

经典的蒸馏框架Distilled BiLSTM ，我们将蒸馏点选在最后的score层，将模型正负样本对(query，postitive doc) ( query，negative doc) 的两个输出值(postitive score，negative score)拼接作为logits，用于计算蒸馏的soft loss，hardloss即为student模型和ground truth计算的的hinge loss。最终的distill loss由softloss和hardloss两部分加权获得(见下一节Loss计算)。

训练过程中，固定finetune好的teacher（4L BERT）参数，只利用distill loss对student模型进行梯度优化。

## Loss计算

模型蒸馏的损失函数通常由soft loss和 hard loss两部分组成，soft loss使用MSE计算student logits和teacher logits的距离，使得student模型可以学到大模型teacher的知识；hard loss为ground truth和student模型输出的hinge loss，从而对teacher的错误知识做一定的纠偏。

## CNN模型尝试

在模型蒸馏的student模型选择上，除了将层数少/参数少的Transformer结构之外，我们还尝试了将其蒸馏到CNN模型结构上。在这个匹配任务中，我们选择了非常轻量的textCNN模型作为student。

最初的设计中TextCNN直接复用Teacher模型的Embedding层，通过拼接不同大小的卷积核提取出来的隐层特征值来计算匹配得分。为进一步提升蒸馏模型性能，我们还尝试了将Teacher模型的BERT Embedding替换为腾讯开源的Word2Vec Embedding，同时采用QQSeg分词器进行分词.

### 参考
1.Hinton@NIPS2014：Distilling the Knowledge in a Neural Network

2.Distilling Task-Specific Knowledge from BERT into Simple Neural Networks

3.ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS

4.Transformer to CNN: Label-scarce distillation for efficient text classification 