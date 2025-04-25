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

**Note**  
For the purpose of running training and baseline code from this repository, it is important to keep the data directory structure intact.

### 1. Event and properties files
The event data, which should be used to generate user representations, is divided into five Parquet files. Each file corresponds to a different type of user interaction available in the dataset (see section [Dataset Description](https://github.com/Synerise/recsys2025#dataset-description)):

- **product_buy.parquet**
- **add_to_cart.parquet**
- **remove_from_cart.parquet**
- **page_visit.parquet**
- **search_query.parquet**

Product properties are stored in:

- **product_properties.parquet**

### 2. `input` directory
This directory stores a NumPy file containing a subset of 1,000,000 `client_id`s for which Universal Behavioral Profiles should be generated:

- **relevant_clients.npy**

Using the event files, participants are required to create Universal Behavioral Profiles for the clients listed in `relevant_clients.npy`. These clients are identified by the `client_id` column in the event data.

The generated profiles must follow the format outlined in the **Competition Entry Format** section and will serve as input for training models across all specified tasks, including churn prediction, product propensity, category propensity, and additional hidden tasks. The code for the downstream tasks is fixed and provided to participants (see [Model Training](https://github.com/Synerise/recsys2025#model-training) section).

### 3. `target` directory
This directory stores the labels for propensity tasks. For each propensity task, target category names are stored in NumPy files:

- **propensity_category.npy**: Contains a subset of 100 categories for which the model is asked to provide predictions
- **popularity_propensity_category.npy**: Contains popularity scores for categories from the `propensity_category.npy` file. Scores are used to compute the Novelty measure. For details, see the [Evaluation](https://github.com/Synerise/recsys2025#evaluation) section
- **propensity_sku.npy**: Contains a subset of 100 products for which the model is asked to provide predictions
- **popularity_propensity_sku.npy**: Contains popularity scores for products from the `propensity_sku.npy` file. These scores are used to compute the Novelty measure. For details, see the [Evaluation](https://github.com/Synerise/recsys2025#evaluation) section
- **active_clients.npy**: Contains a subset of relevant clients with at least one `product_buy` event in history (data available for the participants). Active clients are used to compute churn target. For details, see the [Open Tasks](https://github.com/Synerise/recsys2025#open-tasks) section

These files are specifically used to create the ground truth labels for the propensity tasks. The target (ground truth) for each task is automatically computed by the `TargetCalculator` class in `universal_behavioral_modeling_challenge.training_pipeline.target_calculators`.

**Note**  
To run internal experiments with this repository, the event data should be split into `input` and `target` chunks, which are stored in the `input` and `target` directories, respectively. This setup imitates the official evaluation pipeline; however, the official train and validation target data are not provided to competitors. To create the data split, see the [Data Splitting](https://github.com/Synerise/recsys2025#data-splitting) section.

## Competition Entry Format

**Participants are asked to prepare Universal Behavioral Profiles — user representations that will serve as input to the first layer of a neural network with a fixed, simple architecture.** For each submission, the models will be trained and evaluated by the organizers, and the evaluation outcome will be displayed on the leaderboard. However, we make the training pipeline and model architecture available for participants to use in internal testing.

Each competition entry consists of two files: `client_ids.npy` and `embeddings.npy`

#### `client_ids.npy`
   - A file that stores the IDs of clients for whom Universal Behavioral Profiles were created
   - `client_ids` must be stored in a one-dimensional NumPy ndarray with `dtype=int64`
   - The file should contain client IDs from the `relevant_clients.npy` file, and the order of IDs must match the order of embeddings in the embeddings file

#### `embeddings.npy`
   - A file that stores Universal Behavioral Profiles as a **dense user embeddings matrix**
   - Each embedding corresponds to the client ID from `client_ids.npy` with the same index
   - Dense embeddings must be stored in a two-dimensional NumPy ndarray, where:
     - The first dimension corresponds to the number of users and matches the dimension of the `client_ids` array
     - The second dimension represents the embedding size
     - The `dtype` of the embeddings array must be `float16`
   - The embedding size cannot exceed `max_embedding_dim = 2048`

**It is crucial that the order of IDs in the `client_ids` file matches the order of embeddings in the embeddings file to ensure proper alignment and data integrity.**

### **IMPORTANT! The maximum length of the embedding vectors is 2048.**

## Competition Entry Validator

Competitors must ensure that their submitted files adhere to this structure for successful participation in the competition. The entry format can be validated using the provided validation script:

`universal_behavioral_modeling_challenge/validator/run.py`

### **Arguments**
   - `--data-dir`: The directory containing the data provided by the organizer, including `relevant_clients.npy` file in `input` subdirectory (described in **Data Format** section)
   - `--embeddings-dir`: The directory where the `client_ids` and `embeddings` files are stored

### **Running the Validator**

```bash
python -m validator.run --data-dir <data_dir> --embeddings-dir <your_embeddings_dir>
```

## Model training
Model training is conducted by challenge organizers. Multiple indepedent models with identical architecutre are trained for downstream tasks (churn, propensity, and hidden tasks) and the combined score is presented on the leaderboard. The training process is fixed for every task and all competition entries. Our objective is to evaluate the expressive power of created Universal Behaviorl Profiles, not the model architecture itself.

### Model architecture
- the model architecture consists of three Inverted Bottleneck blocks, adapted from *Scaling MLPs: A Tale of Inductive Bias* (https://arxiv.org/pdf/2306.13575.pdf), with layer normalization and residual connections; see `UniversalModel` class in `universal_behavioral_modeling_challenge.training_pipeline.model`
- the input to the first layer of the network are embeddings provided in the competition entry
- the output is task-specific and computed by the `TargetCalculator` class in `universal_behavioral_modeling_challenge.training_pipeline.target_calculators`
- model hyperparameters are fixed and the same for each task and each competition entry:
   - BATCH_SIZE = 128
   - HIDDEN_SIZE_THIN = 2048
   - HIDDEN_SIZE_WIDE = 4096
   - LEARNING_RATE = 0.001
   - MAX_EMBEDDING_DIM = 2048
   - MAX_EPOCH = 3

**Note**  
For model evaluation, we consider the best score out of 3 epochs.

## Evaluation

The primary metric used to measure model performance is AUROC. We use `torchmetrics` implementation of AUROC. Additionally, the performance of Category Propensity and Product Propensity models is evaluated based on the novelty and diversity of the predictions. In these cases, the task score is calculated as a weighted sum of all metrics:
```
0.8 × AUROC + 0.1 × Novelty + 0.1 × Diversity
```
### Diversity

To compute the diversity of single prediction, we first apply element-wise sigmoid to the predictions, and l1 normalize the result. The diversity of the prediction is the entropy of this distribution.

The final diversity score is computed as the average diversity of the model's predictions.

### Novelty

The popularity of a single prediction is the weighted sum of the popularities of the top `k` recommended targets in the prediction. This is normalized so that a popularity score of `1` corresponds to the following scenario:
> The model's top `k` predictions are the `k` most popular items, and the model is absolutely certain about predicting all of these items.

The popularity score is then computed as the average popularity of the model's predictions. Finally, we compute the novelty of the predictions as `1 - popularity`.
 
Due to the sparsity of the data, the popularity scores, as computed so far are close to 0, and thus the corresponding raw novelty scores are really close to 1. To make the measure more sensitive to small changes near 1, we raise the raw popularity score to the 100th power.

### Final leaderboard score

For each task, a leaderboard is created based on the respective task scores. The final score, which evaluates the overall quality of user representations and their ability to generalize, is determined by aggregating ranks from all per-task leaderboards using the Borda count method. In this approach, each model's rank in a task leaderboard is converted into points, where a model ranked `k`-th among `N` participants receives `N - k` points. The final ranking is based on the total points accumulated across all tasks, ensuring that models performing well consistently across multiple tasks achieve a higher overall score.

## Competition submission

1. Organizers provide the input set of event data as described in the [Data Format](https://github.com/Synerise/recsys2025#data-format) section
2. Competitors are asked to create user embeddings (Universal Behavioral Profiles) based on the provided data
3. Created embeddings are submitted following the **Competition Entry Format**
4. Organizers are using submitted embeddings to train models in multiple **Downstream Tasks**
5. Models are validated on the left-out subset of data
6. Validation results are presented on the leaderboards


## Competition code
We provide a framework that participants can use to test their solutions. The same code is used in the competition to train models for downstream tasks. Only targets for hidden tasks are not included in the provided code.

### Requirements
Requirements are provided in the `requirements.txt` file.

### Data Splitting
Running the competition code for internal tests requires splitting raw event data into three distinct time windows: input events, events to compute train target, and events to compute validation target.
The first set of events is training input data that are used to create users' representations. These representations serve as an input to train downstream models. For baseline solutions with user representation methods see `baseline` module in this repository.
The training target is not included in data tables explicitly, but is computed on the fly based on events in the training target time window. It consists of 14 days after the last timestamp in the input data. The model is trained to predict events in the target based on input user representations.

**Target example:**
To create a propensity target, we check if the user made any purchase in a given category within the provided target time window. Propensity target categories are provided in separate files: `propensity_category.npy` and `propensity_sku.npy`. In the case of a churn target, we check if the user made any purchase in the provided target data sample.

The next 14 days after the training target set are used to compute the validation target and measure model performance.

**IMPORTANT! This data-splitting procedure is meant for internal testing. In the official competition settings, users' representations — which are the competition entry — should be created based on ALL provided events. The official training and validation targets are hidden from the competitors.**

### Split data script
We provide a script to split data according to the described procedure:
`data_utils/split_data.py`

**Arguments**

- `--challenge-data-dir`: Competition data directory which should consist of event files, product properties file and two subdirectories — input (with `relevant_clients.npy`) and target (with `propensity_category.npy` and `propensity_sku.npy`).

**Output**
Input events are saved as Parquet files in the input subdirectory in `--challenge-data-dir`. Train and validation target events are saved as Parquet files in the target subdirectory in `--challenge-data-dir`

**Running**
Run the script:
```bash
python -m data_utils.split_data --challenge-data-dir <your_challenge_data_dir>
```

### Model training script
**Arguments**
- `--data-dir`: Directory where competition target and input data are stored.
- `--embeddings-dir`: Directory where Universal Behavioral Profiles, which are used as model input embeddings are stored. Embeddings should be stored in the format described in the **Competition Entry Format** section.
- `--tasks`: List of tasks to evaluate the model on, possible values are: `churn`, `propensity_category`, `propensity_sku`.
- `--log-name`: Name for the experiment, used for logging.
- `--num-workers`: Number of subprocesses for data loading.
- `--accelerator`: Type of accelerator to use. Argument is directly passed to `pl.LightningModule`. Possible values include: `gpu`, `cpu`. For more options, [see here](https://lightning.ai/docs/pytorch/stable/common/trainer.html#accelerator) .
- `--devices`: List of devices to use for training. Note that using `auto` when `accelerator="gpu"` sometimes produces undesired behavior, and may result in slower training time.
- `--neptune-api-token` (optional): API token for Neptune logger. If not specified, the results are logged offline.
- `--neptune-project` (optional): Name of Neptune project in the format `<workspace>/<project>` to log the results of the experiment to. If not specified, the results are logged offline.
- `--disable-relevant-clients-check` (optional): This flag disables the validation check that ensures the `client_ids.npy` file from the submission matches the contents of `relevant_clients.npy`. It allows training to be run on a different set of clients than the relevant clients.  

**Note**
For the official submission, the relevant clients validation check will be enabled, and embeddings must be provided **for all and only the relevant clients**. However, the --disable-relevant-clients-check flag should be used for internal experiments, as not all relevant clients remain in the input data after using the `data_utils.split_data` script.

**Running scripts**

Offline logging:
```bash
python -m training_pipeline.train --data-dir <your_splitted_challenge_data_dir> --embeddings-dir <your-embeddings-dir> --tasks churn propensity_category propensity_sku --log-name <my_experiment> --accelerator gpu --devices 0 --disable-relevant-clients-check
```
Logging into a Neptune workspace:
```bash
python -m training_pipeline.train --data-dir <your_splitted_challenge_data_dir> --embeddings-dir <your-embeddings-dir> --tasks churn propensity_category propensity_sku --log-name <my_experiment> --accelerator gpu --devices 0 --neptune-api-token <your-api-token> --neptune-project <your-worskspace>/<your-project> --disable-relevant-clients-check
```

## Baselines
In the `baseline` directory, we provide scripts with an example competition entry and an additional README_AGGREGATED_FEATURES.md file with more detailed instructions.  

---
*In case of any problems, you can contact the organizers by email via the address [recsyschallenge\@synerise.com](mailto:recsyschallenge\@synerise.com).*