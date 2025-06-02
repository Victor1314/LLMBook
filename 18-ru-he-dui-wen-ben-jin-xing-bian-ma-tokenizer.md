# 如何对文本进行编码？Tokenizer



数据处理的第一步，就是使用数字对文本进行编码。 

Tokenizer，分词器，将文本划分成一个个最小单元 —词元(token),  用于模型训练。

以BERT、GPT等预训练语言模型为例，tokenizer的作用流程如下：

1. 输入文本："I love AI."
2. 分词器处理后：["I", "love", "AI", "."]
3. 再转成ID：[101, 2001, 727, 102]（假设的ID）
4. 输入到模型中。

词表(Vocabulary)是分词器的重要组件，将token转化成token_id。在训练前的语料处理，模型输出的解码阶段，都会用到词表。 词表重要参数就是词表大小，词表越大，能生成的语言越多。

常见的分词器有BPE、wordPiece等。 BPE在大模型广泛使用，而wordPiece则是在Bert中使用。两者都属于

子词分词器。



**什么是子词？**

子词是介于词和字符之间单元。 假如使用词作为单位来划分文本的话，颗粒度太大，词表无法兼容所有的词汇。 容易出现陌生词，也就是OOV问题(out-of-vocabulary) 。 假如使用字符来进行分词，颗粒度太小，token没有实际的含义。

子词兼有两者的优点。

比如： 常见英文单词的前缀和后缀，re，ful，ly。



**BPE(byte pair encoding)算法的原理 .**

它是如何训练，构建词表的呢？

先将要训练的文本集合，按照字符来进行拆分。拆分之后，将相邻字符组合，构成子词，统计子词的频率。

假设文本为，“Ｉ　love you”, 那么统计 I , Io,ov, yo, ou 的频率。

将频率高的子词，确定进行合并。 

再进行下一轮统计。将子词和相邻的字符或者子词合并，合并频率最高的。

直到达到词表的大小。



**wordPiece分词的原理。**

wordPiece是子词分词算法，在Bert等语言模型之中广泛使用。wordPiece和BPE的区别在与在进行子词合并的时候，考虑了语料的概率。也就是说，在分词之后，要确常见子词的概率。而BPE中仅根据出现频率来进行考虑。出现频率高的，不一定是常见子词。

假设我们有如下简单语料：

```
复制unhappiness
unhappy
unhappily
unhappiest
```

**1. BPE 的处理方式**

BPE 会统计所有连续字符对的出现频率，比如：

- “un”、“ha”、“pp”、“in”、“es”、“ly”、“est”等等

假设“pp”在语料中出现频率很高（比如在“happiness”、“happy”、“happily”、“happiest”里都出现了），BPE 可能会优先把“pp”合并成一个子词。

但有时候，某些字符对虽然频率高，却并不是有实际意义的子词（比如“pp”本身在英文中没有独立意义）。

**2. WordPiece 的处理方式**

WordPiece 不仅考虑“pp”的出现频率，还会计算如果把“pp”合并成一个子词，是否能显著提升整个语料的概率（即更好地表示原始语料中的单词）。

假设“un”、“happy”、“ness”、“ly”、“est”这些子词在英语中很常见，WordPiece 可能会更倾向于合并这些有实际意义的子词，而不是仅仅频率高但没意义的“pp”。

比如，WordPiece 可能会优先得到如下子词：

- “un”
- “happy”
- “ness”
- “ly”
- “est”

这样，“unhappiness”会被分成“un + happy + ness”，而不是“un + ha + pp + in + ess”。

| 单词        | BPE 分词结果            | WordPiece 分词结果 |
| ----------- | ----------------------- | ------------------ |
| unhappiness | un + ha + pp + in + ess | un + happy + ness  |
| unhappily   | un + ha + pp + ily      | un + happy + ly    |
| unhappy     | un + ha + pp + y        | un + happy         |



