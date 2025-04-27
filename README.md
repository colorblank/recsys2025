# 通用行为建模数据挑战赛

## 介绍

### 为什么需要一个通用行为建模挑战?

该挑战旨在推广一种统一的行为建模方法。许多现代企业依赖机器学习和预测分析来改进商业决策。这些组织中常见的预测任务包括推荐、倾向预测、流失预测、用户生命周期价值预测等。用于这些预测任务的核心信息是用户过去行为的日志，例如他们购买了哪些商品、将哪些商品加入购物车、访问了哪些页面等。我们建议将这些任务视为一个整体，而不是单独的问题，提出一种统一的建模方法。

为此，我们引入了通用行为档案的概念——即编码每个个体过去交互关键方面的用户表示。这些档案设计为可跨多个预测任务普遍适用，例如流失预测和产品推荐。通过开发能够捕捉用户行为基本模式的表示，我们使模型能够在不同应用中有效地泛化。

### 挑战概述

本次挑战的目标是基于提供的数据开发通用行为档案，这些数据包括各种类型的事件，如购买、加入购物车、从购物车移除、页面访问和搜索查询。这些用户表示将根据其在各种预测任务中的泛化能力进行评估。挑战参与者的任务是提交用户表示，这些表示将作为简单神经网络架构的输入。基于提交的表示，模型将在多个任务上进行训练，包括一些向参与者公开的任务（称为"公开任务"），以及一些将在比赛结束后揭晓的隐藏任务。最终的性能得分将是所有任务结果的综合。我们会在提交后自动迭代模型训练和评估。参与者的唯一任务是提交通用的用户表示。

- 参与者需要提供用户表示——即通用行为档案
- 下游任务的训练由组织者进行，但竞赛流程是公开的，并在本仓库中展示
- 每个下游任务的模型是单独训练的，但使用相同的嵌入（用户表示）
- 性能将基于所有下游任务进行评估

### 公开任务
- **流失预测:** 二元分类，1表示用户将流失，0表示用户不会流失。流失任务在具有至少一次`product_buy`事件的历史活跃用户子集上执行（数据可供参与者使用）
任务名称: `churn`
- **类别倾向:** 多标签分类，分为100个可能的标签之一。这些标签代表最常购买的100个产品类别。
任务名称: `propensity_category`
- **产品倾向:** 多标签分类，分为100个可能的标签之一。这些标签代表训练目标数据中最常购买的100个产品。
任务名称: `propensity_sku`

### 隐藏任务
除了公开任务外，本次挑战还包括隐藏任务，这些任务在比赛期间保持保密。这些任务的目的是确保提交的通用行为档案能够泛化，而不是针对特定已知目标进行微调。与公开任务类似，隐藏任务基于提交的表示来预测用户行为，但它们引入了参与者没有明确优化的新场景。

比赛结束后，隐藏任务将与相应的代码一起公布，允许参与者复现结果。

## 数据集

我们发布了一个包含真实用户交互日志的匿名数据集。
此外，我们提供了可以与`product_buy`、`add_to_cart`和`remove_from_cart`事件类型连接的产品属性。
每个数据源都存储在一个单独的文件中。

**注意**
所有记录的交互都可以用于创建通用行为档案；然而，参与者只需要为1,000,000个用户的子集提交行为档案，这些档案将用于模型训练和评估。


|          |    product_buy    |    add_to_cart    |    remove_from_cart    |      page_visit  |      search_query  |
|:---------|:-----------------:|:-----------------:|:----------------------:|:----------------:|:------------------:|
|   Events |     1,682,296     |     5,235,882     |     1,697,891          |     150,713,186  |     9,571,258      |

### 数据集描述

#### 列

**product_properties**:
- **sku (int64):** 商品的数字ID。
- **category (int64):** 商品类别的数字ID。
- **price (int64):** 商品价格区间的数字ID（参见[列编码](https://github.com/Synerise/recsys2025#column-encoding)部分）。
- **name (object):** 表示商品名称量化嵌入的数字ID向量（参见[列编码](https://github.com/Synerise/recsys2025#column-encoding)部分）。

**product_buy**:
- **client_id (int64):** 客户（用户）的数字ID。
- **timestamp (object):** 事件日期，格式为YYYY-MM-DD HH:mm:ss。
- **sku (int64):** 商品的数字ID。

**add_to_cart**:
- **client_id (int64):** 客户（用户）的数字ID。
- **timestamp (object):** 事件日期，格式为YYYY-MM-DD HH:mm:ss。
- **sku (int64):** 商品的数字ID。

**remove_from_cart**:
- **client_id (int64):** 客户（用户）的数字ID。
- **timestamp (object):** 事件日期，格式为YYYY-MM-DD HH:mm:ss。
- **sku (int64):** 商品的数字ID。

**page_visit**:
- **client_id (int64):** 客户的数字ID。
- **timestamp (object):** 事件日期，格式为YYYY-MM-DD HH:mm:ss。
- **url (int64):** 访问URL的数字ID。关于特定页面上展示的内容（例如哪个商品）的明确信息未提供。

**search_query**:
- **client_id (int64):** 客户的数字ID。
- **timestamp (object):** 事件日期，格式为YYYY-MM-DD HH:mm:ss。
- **query (object):** 表示搜索查询词量化嵌入的数字ID向量（参见[列编码](https://github.com/Synerise/recsys2025#column-encoding)部分）。

#### 列编码

**文本列 ('name', 'query')**:
为了匿名化数据，我们首先使用适当的大语言模型（LLM）对文本进行嵌入。然后，我们使用高质量的嵌入量化方法对嵌入进行量化。最终的量化嵌入长度为16个数字（桶），每个桶有256个可能的值：{0, …, 255}。

**数字列 ('price')**:
这些列最初是浮点数，被分割为100个基于分位数的桶。

## 数据格式

本节描述竞赛数据的格式。
我们提供了一个包含事件文件和两个子目录的数据目录：`input`和`target`。

好的，这是您提供的英文文档的中文翻译：

-----

**注意**

为了运行此存储库中的训练和基线代码，保持数据目录结构完整非常重要。

### 1\. 事件和属性文件

用于生成用户表示的事件数据被分为五个 Parquet 文件。每个文件对应于数据集中不同类型的用户交互（见[数据集描述](https://github.com/Synerise/recsys2025#dataset-description)部分）：

  - **product\_buy.parquet**
  - **add\_to\_cart.parquet**
  - **remove\_from\_cart.parquet**
  - **page\_visit.parquet**
  - **search\_query.parquet**

商品属性存储在：

  - **product\_properties.parquet**

### 2\. `input` 目录

此目录存储一个 NumPy 文件，其中包含 1,000,000 个需要生成通用行为特征的 `client_id` 子集：

  - **relevant\_clients.npy**

参与者需要使用事件文件为 `relevant_clients.npy` 中列出的客户创建通用行为特征。这些客户通过事件数据中的 `client_id` 列进行标识。

生成的特征必须遵循**竞赛提交格式**部分中概述的格式，并将作为跨所有指定任务（包括流失预测、商品偏好、类别偏好和额外的隐藏任务）训练模型的输入。下游任务的代码是固定的，并提供给参与者（见[模型训练](https://github.com/Synerise/recsys2025#model-training)部分）。

### 3\. `target` 目录

此目录存储偏好任务的标签。对于每个偏好任务，目标类别名称存储在 NumPy 文件中：

  - **propensity\_category.npy**: 包含模型需要提供预测的 100 个类别的子集
  - **popularity\_propensity\_category.npy**: 包含 `propensity_category.npy` 文件中类别的流行度得分。这些得分用于计算新颖性指标。有关详细信息，请参阅[评估](https://github.com/Synerise/recsys2025#evaluation)部分
  - **propensity\_sku.npy**: 包含模型需要提供预测的 100 个商品的子集
  - **popularity\_propensity\_sku.npy**: 包含 `propensity_sku.npy` 文件中商品的流行度得分。这些得分用于计算新颖性指标。有关详细信息，请参阅[评估](https://github.com/Synerise/recsys2025#evaluation)部分
  - **active\_clients.npy**: 包含至少有一个 `product_buy` 历史事件的相关客户子集（参与者可以获得的数据）。活跃客户用于计算流失目标。有关详细信息，请参阅[公开任务](https://github.com/Synerise/recsys2025#open-tasks)部分

这些文件专门用于创建偏好任务的真实标签。每个任务的目标（真实值）由 `universal_behavioral_modeling_challenge.training_pipeline.target_calculators` 中的 `TargetCalculator` 类自动计算。

**注意**

为了在此存储库中运行内部实验，事件数据应拆分为 `input` 和 `target` 两个部分，分别存储在 `input` 和 `target` 目录中。此设置模仿了官方评估流程；但是，官方的训练和验证目标数据不会提供给参赛者。要创建数据拆分，请参阅[数据拆分](https://github.com/Synerise/recsys2025#data-splitting)部分。

## 竞赛提交格式

**参与者需要准备通用行为特征——用户表示，它将作为具有固定、简单架构的神经网络第一层的输入。** 对于每次提交，模型将由组织者进行训练和评估，评估结果将显示在排行榜上。但是，我们提供了训练流程和模型架构供参与者在内部测试中使用。

每个竞赛提交包含两个文件：`client_ids.npy` 和 `embeddings.npy`

#### `client_ids.npy`

  - 存储已创建通用行为特征的客户 ID 的文件
  - `client_ids` 必须存储在一维 NumPy ndarray 中，数据类型为 `dtype=int64`
  - 该文件应包含来自 `relevant_clients.npy` 文件的客户 ID，并且 ID 的顺序必须与 embeddings 文件中 embeddings 的顺序相匹配

#### `embeddings.npy`

  - 存储通用行为特征作为**稠密用户嵌入矩阵**的文件
  - 每个 embedding 对应于 `client_ids.npy` 中具有相同索引的客户 ID
  - 稠密 embeddings 必须存储在二维 NumPy ndarray 中，其中：
      - 第一维对应于用户数量，并与 `client_ids` 数组的维度相匹配
      - 第二维表示 embedding 的大小
      - embeddings 数组的数据类型 `dtype` 必须为 `float16`
  - embedding 的大小不能超过 `max_embedding_dim = 2048`

**至关重要的是，`client_ids` 文件中 ID 的顺序必须与 embeddings 文件中 embeddings 的顺序相匹配，以确保正确的对齐和数据完整性。**

### **重要提示！嵌入向量的最大长度为 2048。**

## 竞赛提交验证器

参赛者必须确保其提交的文件符合此结构，才能成功参与竞赛。可以使用提供的验证脚本验证条目格式：

`universal_behavioral_modeling_challenge/validator/run.py`

### **参数**

  - `--data-dir`: 包含组织者提供的数据的目录，包括 `input` 子目录中的 `relevant_clients.npy` 文件（在**数据格式**部分中描述）
  - `--embeddings-dir`: 存储 `client_ids` 和 `embeddings` 文件的目录

### **运行验证器**

```bash
python -m validator.run --data-dir <data_dir> --embeddings-dir <your_embeddings_dir>
```

## 模型训练

模型训练由竞赛组织者进行。多个具有相同架构的独立模型针对下游任务（流失、偏好和隐藏任务）进行训练，并在排行榜上显示组合得分。对于每个任务和所有竞赛提交，训练过程都是固定的。我们的目标是评估所创建的通用行为特征的表达能力，而不是模型架构本身。

### 模型架构

  - 模型架构由三个倒置瓶颈块组成，改编自 *Scaling MLPs: A Tale of Inductive Bias* ([https://arxiv.org/pdf/2306.13575.pdf](https://arxiv.org/pdf/2306.13575.pdf))，带有层归一化和残差连接；参见 `universal_model_challenge.training_pipeline.model` 中的 `UniversalModel` 类
  - 网络第一层的输入是竞赛提交中提供的 embeddings
  - 输出是特定于任务的，由 `universal_behavioral_modeling_challenge.training_pipeline.target_calculators` 中的 `TargetCalculator` 类计算
  - 模型超参数是固定的，并且对于每个任务和每个竞赛提交都是相同的：
      - BATCH\_SIZE = 128
      - HIDDEN\_SIZE\_THIN = 2048
      - HIDDEN\_SIZE\_WIDE = 4096
      - LEARNING\_RATE = 0.001
      - MAX\_EMBEDDING\_DIM = 2048
      - MAX\_EPOCH = 3

**注意**

对于模型评估，我们考虑 3 个 epoch 中的最佳得分。

## 评估

用于衡量模型性能的主要指标是 AUROC。我们使用 `torchmetrics` 库实现的 AUROC。此外，类别偏好和商品偏好模型的性能还基于预测的新颖性和多样性进行评估。在这种情况下，任务得分计算为所有指标的加权和：

```
0.8 × AUROC + 0.1 × 新颖性 + 0.1 × 多样性
```

### 多样性

要计算单个预测的多样性，我们首先对预测应用元素级的 sigmoid 函数，然后对结果进行 l1 归一化。预测的多样性是此分布的熵。

最终的多样性得分计算为模型预测的平均多样性。

### 新颖性

单个预测的流行度是预测中前 `k` 个推荐目标的流行度的加权和。这经过归一化处理，使得流行度得分为 `1` 对应于以下情况：

> 模型的前 `k` 个预测是 `k` 个最受欢迎的商品，并且模型对预测所有这些商品都非常有把握。

然后将流行度得分计算为模型预测的平均流行度。最后，我们将预测的新颖性计算为 `1 - 流行度`。

由于数据的稀疏性，到目前为止计算出的流行度得分接近于 0，因此相应的原始新颖性得分非常接近于 1。为了使该指标对接近 1 的微小变化更敏感，我们将原始流行度得分提高到 100 次方。

### 最终排行榜得分

对于每个任务，都会根据各自的任务得分创建一个排行榜。最终得分用于评估用户表示的整体质量及其泛化能力，通过使用 Borda 计数法聚合来自所有按任务划分的排行榜的排名来确定。在这种方法中，模型在任务排行榜中的排名 `k` 会转换为分数，其中在 `N` 个参与者中排名第 `k` 的模型获得 `N - k` 分。最终排名基于跨所有任务累积的总分，确保在多个任务中表现始终良好的模型获得更高的总分。

## 竞赛提交

1.  组织者提供如[数据格式](https://github.com/Synerise/recsys2025#data-format)部分所述的输入事件数据集。
2.  参赛者需要根据提供的数据创建用户嵌入（通用行为特征）。
3.  创建的嵌入按照**竞赛提交格式**提交。
4.  组织者使用提交的嵌入在多个**下游任务**中训练模型。
5.  模型在预留的数据子集上进行验证。
6.  验证结果将显示在排行榜上。

## 竞赛代码

我们提供了一个框架，参与者可以使用该框架测试他们的解决方案。竞赛中也使用相同的代码来训练下游任务的模型。只有隐藏任务的目标不包含在提供的代码中。

### 要求

要求在 `requirements.txt` 文件中提供。

### 数据拆分

运行竞赛代码进行内部测试需要将原始事件数据拆分为三个不同的时间窗口：输入事件、计算训练目标事件和计算验证目标事件。

第一组事件是训练输入数据，用于创建用户表示。这些表示作为训练下游模型的输入。有关使用用户表示方法的基线解决方案，请参阅此存储库中的 `baseline` 模块。

训练目标没有在数据表中显式包含，而是根据训练目标时间窗口中的事件动态计算的。它包括输入数据中最后一个时间戳之后的 14 天。模型经过训练以基于输入用户表示预测目标中的事件。

**目标示例：**

要创建偏好目标，我们会检查用户是否在给定的目标时间窗口内在给定的类别中进行了任何购买。偏好目标类别在单独的文件中提供：`propensity_category.npy` 和 `propensity_sku.npy`。对于流失目标，我们会检查用户是否在提供的目标数据样本中进行了任何购买。

训练目标集之后的接下来的 14 天用于计算验证目标并衡量模型性能。

**重要提示！此数据拆分过程仅用于内部测试。在官方竞赛设置中，用户的表示（即竞赛提交）应基于所有提供的事件创建。官方的训练和验证目标对参赛者是隐藏的。**

### 拆分数据脚本

我们提供了一个脚本来按照描述的过程拆分数据：

`data_utils/split_data.py`

**参数**

  - `--challenge-data-dir`: 竞赛数据目录，应包含事件文件、商品属性文件和两个子目录——input（包含 `relevant_clients.npy`）和 target（包含 `propensity_category.npy` 和 `propensity_sku.npy`）。

**输出**

输入事件将作为 Parquet 文件保存在 `--challenge-data-dir` 的 input 子目录中。训练和验证目标事件将作为 Parquet 文件保存在 `--challenge-data-dir` 的 target 子目录中。

**运行**

运行脚本：

```bash
python -m data_utils.split_data --challenge-data-dir <your_challenge_data_dir>
```

### 模型训练脚本

**参数**

  - `--data-dir`: 存储竞赛目标和输入数据的目录。
  - `--embeddings-dir`: 存储用作模型输入嵌入的通用行为特征的目录。嵌入应以**竞赛提交格式**部分描述的格式存储。
  - `--tasks`: 要评估模型的任务列表，可能的值包括：`churn`、`propensity_category`、`propensity_sku`。
  - `--log-name`: 实验的名称，用于日志记录。
  - `--num-workers`: 数据加载的子进程数。
  - `--accelerator`: 要使用的加速器类型。该参数直接传递给 `pl.LightningModule`。可能的值包括：`gpu`、`cpu`。更多选项请[参见此处](https://lightning.ai/docs/pytorch/stable/common/trainer.html#accelerator)。
  - `--devices`: 用于训练的设备列表。请注意，当 `accelerator="gpu"` 时使用 `auto` 有时会产生不希望的行为，并可能导致训练时间变慢。
  - `--neptune-api-token` (可选): Neptune 日志记录器的 API 令牌。如果未指定，结果将离线记录。
  - `--neptune-project` (可选): Neptune 项目的名称，格式为 `<workspace>/<project>`，用于将实验结果记录到该项目。如果未指定，结果将离线记录。
  - `--disable-relevant-clients-check` (可选): 此标志禁用验证检查，该检查确保提交的 `client_ids.npy` 文件与 `relevant_clients.npy` 的内容匹配。它允许在与相关客户不同的客户集上运行训练。

**注意**

对于正式提交，将启用相关客户验证检查，并且必须**仅为所有相关客户**提供嵌入。但是，对于内部实验，应使用 `--disable-relevant-clients-check` 标志，因为在使用 `data_utils.split_data` 脚本后，并非所有相关客户都保留在输入数据中。

**运行脚本**

离线日志记录：

```bash
python -m training_pipeline.train --data-dir <your_splitted_challenge_data_dir> --embeddings-dir <your-embeddings-dir> --tasks churn propensity_category propensity_sku --log-name <my_experiment> --accelerator gpu --devices 0 --disable-relevant-clients-check
```

记录到 Neptune 工作区：

```bash
python -m training_pipeline.train --data-dir <your_splitted_challenge_data_dir> --embeddings-dir <your-embeddings-dir> --tasks churn propensity_category propensity_sku --log-name <my_experiment> --accelerator gpu --devices 0 --neptune-api-token <your-api-token> --neptune-project <your-worskspace>/<your-project> --disable-relevant-clients-check
```

## 基线

在 `baseline` 目录中，我们提供了包含示例竞赛提交的脚本和一个额外的 README\_AGGREGATED\_FEATURES.md 文件，其中包含更详细的说明。

-----

*如有任何问题，您可以通过电子邮件联系组织者：[email address removed]。*