# pytorch rnn/lstm 古诗词生成
## 数据集处理
### 将古诗读取为list，并按字数排序
### 字频统计并从高到低排序生成words_vector,word_id
### 将古诗list数字化编码，生成poetrys_vector
### 封装为pytorch数据集
## 网络模型
### 词向量层+lstm/rnn+全连接层
## 生成古诗
### 载入模型，根据输入前缀，生成hidden，output作为下次input 
## 第一首诗 epoch one
### ['花', '落', '春', '风', '起', '，', '江', '南', '月', '照', '秋', '。', '不', '知', '无', '处', '处', '，', '不', '见', '一', '枝', '中', '。']