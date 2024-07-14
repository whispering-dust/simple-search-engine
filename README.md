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

### 语义搜索