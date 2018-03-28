# Text Classify

之前参加了一个比赛，比赛内容是和文本分类有关，实现了常见的rnn,cnn,cnn-rnn等模型，将代码整理以下，方便大家参考。

## 模型训练步骤

### 运行环境
tensorflow == 1.3
python == 2.7

### 预处理数据

- 数据分词
由于数据量较大，预先将数据分词，减少后来训练调参时候的分词时间开销。

- 训练词向量
训练词向量以便后续模型使用，同时保存词向量矩阵。

执行命令：`python preprocess.py`

### 模型训练模式

有三种训练方式：single, multi, kfold, 默认为single
single表示只会训练一次模型；
multi会多次训练模型，取多次训练结果的平均，每次训练数据选取随机；
kfold采取k折交叉验证训练模型，取多次训练结果的平均。

执行命令：`python train.py --mode==single`
