# 为何Transfomer架构能够胜出？



先看看transformer之前的RNN和CNN. 



RNN和LSTM是什么 ? 

1. RNN（循环神经网络，Recurrent Neural Network）

**基本原理**

- RNN是一类用于处理序列数据的神经网络。
- 不同于传统的前馈神经网络，RNN在每个时间步都会接收当前输入和上一个时间步的“隐藏状态”作为输入，实现信息的“记忆”与传递。

**优点**

- 能处理变长的序列数据（如文本、语音、时间序列等）。
- 结构简单，参数共享，适合序列建模。

**缺点**

- **梯度消失/爆炸**：长序列训练时，梯度会迅速变小或变大，导致模型难以捕捉长距离依赖关系。
- 训练效率较低，不能并行。

LSTM（长短期记忆网络，Long Short-Term Memory）

**基本原理**

- LSTM是RNN的改进版，专门为了解决RNN的“长距离依赖”问题（即梯度消失/爆炸）。
- 通过引入“门控机制”，控制信息的“记忆”与“遗忘”。LSTM单元包含三个门：输入门、遗忘门、输出门。

| 特点       | RNN               | LSTM                     |
| ---------- | ----------------- | ------------------------ |
| 结构       | 简单              | 复杂（有门控）           |
| 长距离依赖 | 容易丢失          | 能较好捕捉               |
| 参数量     | 少                | 多                       |
| 训练难度   | 梯度消失/爆炸严重 | 缓解梯度消失/爆炸        |
| 计算速度   | 慢（不能并行）    | 慢（不能并行）           |
| 应用       | 简单序列建模      | 复杂序列建模，长依赖场景 |



transformer的优势在哪里？  

1. 并行计算能力强
RNN/LSTM：序列数据必须按时间步依次处理，不能并行（即第t步的输出依赖于第t-1步的输出），导致训练和推理速度慢。
Transformer：基于自注意力机制，所有位置的输入可以同时处理，实现完全并行，大幅提升训练效率。
2. 捕捉长距离依赖能力强
RNN/LSTM：虽然LSTM通过门控机制缓解了梯度消失问题，但长距离依赖仍然难以捕捉，信息传递路径长，容易丢失上下文信息。
Transformer：自注意力机制可以直接建立任意两个位置之间的联系，无论距离多远，捕捉长距离依赖效果更好。
3. 建模灵活
RNN/LSTM：只能顺序建模，难以处理非顺序结构的数据。
Transformer：通过自注意力机制，可以灵活地建模序列中任意位置之间的关系，更适合复杂结构的数据。
4. 扩展性强
RNN/LSTM：堆叠层数受限，层数多了易出现梯度消失或爆炸。
Transformer：结构简单，易于堆叠更深的网络层，提升模型容量和表达能力。
5. 更适合大规模数据和预训练
Transformer结构非常适合大规模数据的分布式训练，也是BERT、GPT等预训练模型的基础结构。
总结
RNN/LSTM：顺序处理、依赖前后关系、捕捉长距离依赖弱、不易并行。
Transformer：全局自注意力、并行处理、捕捉长距离依赖强、易扩展和预训练。



