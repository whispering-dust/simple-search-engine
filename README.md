# MinSearch

本文是记录跟随llm-zoomcamp的开源课程[Build Your Own Search Engine](https://github.com/alexeygrigorev/build-your-own-search-engine)实现一个简易搜索引擎的过程，该项目主要为后续课程内容的基础。项目包含三个notebook：

- text-base-impl-process：记录了基于文本搜索的实现过程
- Text-base-search-engine：是封装后的简易文本搜索引擎

## 前置准备

### 数据集

课程相关的问答数据集：

- [DE Zoomcamp](https://docs.google.com/document/d/19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit)
- [ML Zoomcamp](https://docs.google.com/document/d/1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit)
- [MLOps Zoomcamp](https://docs.google.com/document/d/12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit)

### 环境配置

jupyter

scikit-learn

pandas

transformers

tqdm

pytorch

## 具体实现

搜索类型常分为：

- 文本搜索
- 语义搜索（向量搜索）

### 文本搜索

#### 基础

- 信息检索：基于用户查询从大型数据集中获取相关信息的过程。
- 向量空间：一种数学表示，将文本转换为向量（空间中的点），以便进行定量比较。
- 单词袋：一个简单的文本表示模型，将每个文档视为单词的集合，忽略语法和语序，但保持多样性。
- TF-IDF（术语频率逆文档频率）：用于评估一个单词对集合或语料库中的文档的重要性的统计度量。它随着单词在文档中出现的次数而增加，但被该单词在语料库中的频率所抵消。

#### 向量空间

- 将文档转化为向量
- 在搜索过程中，可以将数据文档映射到向量空间中，对于一个文档矩阵，有：
  - 行：documents
  - 列：word/tokens
- 通过矩阵获得一个单词表
  - 记录的单词将会失去其在原文中的顺序
  - 使用稀疏矩阵表示

### 语义搜索（嵌入和向量搜索）

文本搜索是利用询问语句中的关键词，在文档中找到与关键词匹配度较高的内容实现的。但是如果出现同义词，基于文本的搜索并不能很好地给出正确答案，因此需要引入语义搜索。

#### 嵌入

- 转换为数字：嵌入技术将不同的单词、句子和文档转换为密集向量（数字数组）。 
- 捕捉相似性：它们确保相似的项目具有相似的数值向量，从而在特征上展示它们的接近性。 
- 降维：嵌入技术将复杂的特征简化为向量。 
- 在机器学习中的应用：这些数值向量被用于推荐、文本分析和模式识别等机器学习模型任务。

#### SVD

SVD，全称为奇异值分解（Singular Value Decomposition），是线性代数中的一种矩阵分解方法。奇异值分解是将词袋表示转化为嵌入的最简单方法

SVD 在许多应用中非常有用，例如：

- **数据降维**：通过保留最大奇异值，可以实现数据的降维处理。
- **信号处理**：用于去除噪声和信号分离。
- **推荐系统**：用于矩阵填充和协同过滤。
- **图像压缩**：通过只保留最大的奇异值和相应的向量，可以大幅度减少图像数据的存储量。

#### NMF

非负矩阵分解（Non-Negative Matrix Factorization, NMF）是一种矩阵分解技术，主要用于降维、特征提取和信号分离等领域。

SVD的计算会产生负数值，很难解释；而相比之下，NMF采用非负输入并且不会产生负数结果，对此我们可以将通过NMF嵌入的每一列（特征）解释为不同的主题/关注点，以及本文在多大程度上是关于这一概念的。

具体应用场景有：

- **文本挖掘**：在文本挖掘中，NMF 可以用于主题提取，通过分解文档-词语矩阵来发现潜在的主题。
- **图像处理**：在图像处理中，NMF 可以用于特征提取和图像分解，将复杂的图像表示为基本特征的组合。
- **推荐系统**：在推荐系统中，NMF 可以用于用户-物品矩阵分解，帮助生成个性化推荐。

#### BERT

前两种方法的问题在于，它们没有考虑词序。他们只是将所有单词分开处理（这就是为什么它被称为“单词袋”），而BERT和 Transformer模型则考虑到了这点。

BERT，全称为双向编码器表示的转换器（Bidirectional Encoder Representations from Transformers），是由谷歌在2018年发布的一种自然语言处理（NLP）模型，是NLP领域的重要突破。

BERT的特点如下：

- **双向性**：与传统的单向语言模型不同，BERT 使用双向Transformer架构，即它在训练时同时考虑了上下文中的前后词。这使得BERT在理解句子含义和词语之间的关系上更为精准。
- **Transformer架构**：BERT基于Transformer模型，该模型采用自注意力机制，能够高效地处理长距离依赖关系。Transformer由编码器和解码器组成，而BERT只使用了编码器部分。
- **预训练和微调**：BERT使用了大规模文本语料库进行预训练，包括两个任务：
  - **Masked Language Model (MLM)**：在预训练过程中，随机遮蔽一些词，并让模型预测这些词，从而学到双向上下文信息。
  - **Next Sentence Prediction (NSP)**：训练模型预测两个句子是否是连续的，这有助于理解句子之间的关系。 预训练完成后，BERT可以通过在特定任务上的微调来应用于各种NLP任务。