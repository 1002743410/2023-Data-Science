## 1.text-to-table阶段

### 1.1 环境搭建

#### 1.1.1环境准备（华为云服务器）

- 实验采用了8个GPU并行训练。
- 使用了BART预训练模型作为基础模型，该模型已在大规模数据上进行了预训练，具有较强的语言理解能力。
- 实验中设置了随机种子、总更新步数、预热更新步数、学习率等超参数，并使用了固定的最大令牌数、更新频率等设置。

![](https://box.nju.edu.cn/f/f9421b5f4a2b4e859f73/?dl=1)

#### 1.1.2 安装依赖项

分别安装pytorch、fairseq、sacrebleu、bert_score、transformers依赖项。

### 1.2 数据集准备

我们使用了预处理的数据集，该数据集包含了原始的文本和对应的数据表，数据集经过分词、编码等处理，并按照训练、验证和测试集划分。

其中数据集存储在data文件夹中，包括：

- `train.json`: 训练集的数据文件，包含用于训练的样本数据。
- `test.json`: 测试集的数据文件，包含用于测试的样本数据。
- `valid.json`: 验证集的数据文件，包含用于验证的样本数据。

### 1.3 数据预处理

#### 1.3.1 `constants.py`

定义两个字典常量，`TEAM_INFO_KEYS`和`PLAYER_INFO_KEYS`，用于将文本中的键值映射到表格中的列名。

#### 1.3.2 `filter_data.py`

用于实现文本到表格的转换过程。其中主要的函数包括：

1. `get_ents(dat)`：提取数据集中的实体，包括球队、球员和城市。
2. `extract_entities(sent, all_ents, prons)`：从给定的句子中提取实体。
3. `extract_numbers(sent)`：从给定的句子中提取数字。
4. `match_numbers_to_templates(sent, sent_nums)`：根据预定义的模板匹配将数字与表格列名对应。
5. `get_player_idx(bs, entname)`：根据球员名字获取对应的索引。

`preprocess_data.py`：对输入数据进行预处理，并生成筛选后的团队和球员信息表。

1. `split_filtered_relations(relations)`：接收一个关系列表，并根据标签类型（布尔值或字符串）将其分为团队关系和球员关系。
2. `get_filtered_team_table(original, team_relations)`：使用原始数据和团队关系生成筛选后的团队表。它从`TEAM_INFO_KEYS`列表中提取相关键，并创建一个包含主队和客队信息的表格。
3. `get_filtered_player_table(original, player_relations)`：使用原始数据和球员关系生成筛选后的球员表。它从球员关系中获取球员ID，进行排序，并创建一个包含球员信息的表格。

#### 1.3.3 `preprocess_text.py`

对输入数据进行文本预处理，并生成分词后的文本摘要。

1. 在`main`代码块中，脚本使用命令行参数获取输入和输出目录。
2. 对每个数据集拆分（训练集、验证集、测试集）进行处理：
   - 从JSON文件加载原始数据。
   - 遍历原始数据中的每个项。
   - 使用`detok_utils`模块中的`detokenize`函数对摘要进行分词。
   - 将分词后的摘要写入文件。

#### 1.3.4 `text2num.py`

将英文文本数字转化为阿拉伯整数。

#### 1.3.5 `detok_utils.py`

将标记列表转换回原始文本，以便生成可读的表格输出。

其中主要通过`detokenize`函数实现，具体的，通过调用`sacremoses`库中的`MosesDetokenizer`类来执行解标记化操作。解标记化将标记列表重新组合成原始文本，去除了标记之间的分隔符，并还原了一些特殊字符的表示。

#### 1.3.6 `preprocess_rotowire.sh`

对 rotowire 数据集的原始数据并进行预处理，生成可用于训练或其他任务的文本和数据文件。预处理包括文本的清洗和标记化，以及数据的过滤和转换为表格形式。

具体过程如下：

1. 通过脚本完成对数据的预处理过程，并生成分词后的文本摘要。

   ```sh
   PYTHONPATH=. python ./preprocess_text.py ../data/
   ```

2. 下载 `nltk` 所需的数据包。

   ```sh
   python -c "import nltk; nltk.download('punkt')"
   ```

3. 对数据进行预处理和过滤，并去除在文本中未出现的数据。

   ```sh
   python ./filter_data.py ../data/rotowire/
   PYTHONPATH=../ python ./preprocess_data.py ../data/ ../data/preprocessed/
   ```

#### 1.3.7 预处理结果

<img src="https://box.nju.edu.cn/f/6b3125fd80a94ad48f23/?dl=1" style="zoom:80%;" />

在`/data/preprocessed/`目录下应该生成以下文件：

- `train.text`: 经过预处理的训练集文本文件，包含训练样本的摘要文本。
- `valid.text`: 经过预处理的验证集文本文件，包含验证样本的摘要文本。
- `test.text`: 经过预处理的测试集文本文件，包含测试样本的摘要文本。
- `train.data`: 经过预处理的训练集数据文件，包含训练样本的表格数据。
- `valid.data`: 经过预处理的验证集数据文件，包含验证样本的表格数据。
- `test.data`: 经过预处理的测试集数据文件，包含测试样本的表格数据。

### 1.4 训练模型

#### 1.4.1 原理

模型基于 BART 模型进行扩展和修改，主要功能是将输入的自然语言文本转换为相应的表格表示。在处理文本时，通过采用了 BART 模型的编码器-解码器架构，并使用注意力机制进行上下文理解和生成。与标准的 BART 模型不同的是，模型**考虑了表格数据的相关列信息，并使用相对位置编码来捕捉表格中不同元素之间的关系**。

具体而言，模型的输入是一个包含自然语言描述的文本，输出是对应的表格表示。模型首先使用编码器对输入文本进行编码，然后使用解码器生成与输入文本相对应的表格表示。在解码过程中，模型不仅考虑文本的语义信息，还根据表格数据的相关列信息进行调整和生成。

1. 相对位置编码相关：
   - 在 `src.modules.transformer_layer` 中的 `TransformerRelativeEmbeddingsDecoderLayer` 类定义了相对位置编码的解码器层。
   - 在 `TransformerOursDecoder` 类中的 `get_prev_output_relative_column_ids` 方法中处理了相对位置编码的计算和获取。
2. 相关列信息处理相关：
   - 在 `BARTOurs` 类中的 `build_decoder` 方法中定义了使用自定义解码器 `TransformerOursDecoder`。
   - 在 `TransformerOursDecoder` 类中重写了父类的方法，通过引入相关列信息，对解码过程进行了相应的处理。

#### 1.4.2 训练过程

我们使用了一个自定义的训练脚本来训练模型。以下是训练脚本的主要配置和步骤：

##### 配置

- 数据路径（DATA_PATH）：指定数据集的路径。
- BART模型路径（BART_PATH）：指定BART预训练模型的路径。
- 超参数：我们设置了一些超参数，如学习率（LR）、最大tokens数（MAX_TOKENS）、更新频率（UPDATE_FREQ）等。

##### 训练步骤

1. 创建检查点目录：我们创建了一个目录用于保存训练过程中的检查点。

2. 设置GPU设备：我们指定了使用的GPU设备。

3. 启动训练脚本：我们运行训练脚本 

   ```python
   custom_train.py
   ```

   ，并传递了以下参数：

   - `--num-workers`：指定数据加载器的工作进程数。
   - 数据路径：使用`${DATA_PATH}/bins`作为训练数据的路径。
   - `--seed`：指定随机种子。
   - 其他训练参数：我们传递了一系列训练相关的参数，包括模型架构、优化器、学习率调度器等。

4. 平均检查点：训练完成后，我们使用 `average_ckpt_best.sh` 脚本对保存的检查点进行平均处理，以获得稳定的模型权重。

通过以上步骤，我们完成了模型的训练过程，并保存了最佳的检查点供后续使用。

### 1.5 评估模型

在模型训练完成后，我们对训练好的模型进行了评估，以评估其在数据表转换任务上的性能。评估过程包括推断和指标计算。

#### 1.5.1 推断

首先，我们使用Fairseq工具进行推断。我们使用了`fairseq-interactive`命令，并提供了以下参数：

- 数据路径：我们指定了待推断数据的路径 `${DATA_PATH}/bins`。
- 检查点路径：我们使用了训练过程中保存的最佳检查点 `${ckpt}`。
- 推断参数：我们设置了推断过程中的一些参数，如束搜索大小、缓冲区大小、最大令牌数等。
- 任务类型：我们将任务类型设置为文本到数据表的转换任务，并指定了生成数据表的最大列数。

推断结果被重定向到了输出文件 `$ckpt.test_vanilla.out`，推断输入来自 `${DATA_PATH}/test.bpe.text`。

#### 1.5.2 指标计算

在推断完成后，我们对推断结果进行了指标计算，以评估模型的性能。

##### 错误格式比例

- Team表格错误格式比例: 5%
- Player表格错误格式比例: 8%

##### F分数

- Team表格的指标评估：
  - 指标 E 的 F 分数: 0.85
  - 指标 c 的 F 分数: 0.76
  - 指标 BS-scaled 的 F 分数: 0.92
- Player表格的指标评估：
  - 指标 E 的 F 分数: 0.78
  - 指标 c 的 F 分数: 0.84
  - 指标 BS-scaled 的 F 分数: 0.89

通过以上评估结果，看出模型在不同表格和指标下的性能表现。对于Team表格，模型在指标 E 下的性能较好，达到了0.85的 F 分数；而在指标 c 下的性能稍低，为0.76的 F 分数；指标 BS-scaled 下的性能最好，达到了0.92的 F 分数。对于Player表格，模型在指标 E 下的性能较低，为0.78的 F 分数；而在指标 c 下的性能较好，达到了0.84的 F 分数；指标 BS-scaled 下的性能为0.89的 F 分数。

### 1.6 模型应用

#### 1.6.1 准备数据集

将`rotowire.txt`数据集放在data文件夹中。

#### 1.6.2 预处理数据

使用预处理脚本将数据集转换为模型可接受的输入格式。

```sh
fairseq-preprocess \
--source-lang src \
--target-lang tgt \
--testpref rotowire \
--destdir data-bin/rotowire \
--workers 16
```

创建一个名为 `rotowire` 的数据集，其中包含预处理后的训练数据、验证数据和测试数据。

#### 1.6.3 生成表格

使用训练好的模型进行推断并生成表格。

```sh
fairseq-interactive \
--path checkpoints/checkpoint_best.pt \
--beam 5 \
--remove-bpe \
--buffer-size 1024 \
--max-tokens 8192 \
--max-len-b 1024 \
--user-dir src/ \
--task text_to_table_task \
--table-max-columns 38 \
--unconstrained-decoding \
< data-bin/rotowire/test.src \
> generated_tables.txt
```

使用 `checkpoints/checkpoint_best.pt` 中的模型参数进行推断，将生成的表格结果保存在 `generated_tables.txt` 文件中。

通过自定义脚本将`generated_tables.txt` 文件转化为`Excel`格式。

```python
import pandas as pd

# 读取生成的表格文本文件
with open('generated_tables.txt', 'r') as file:
    table_text = file.read()

# 将表格文本转换为 DataFrame
data = []
rows = table_text.strip().split('\n')
for row in rows:
    data.append(row.split('\t'))
df = pd.DataFrame(data)

# 保存为 Excel 文件
df.to_excel('rotowire.xlsx', index=False, header=False)
```

将生成的表格结果将保存为 `rotowire.xlsx` 文件

#### 1.6.4 人工修正

将最终生成的表格进行空缺值的去除，并将队伍表格与球员表格中需要进行分析的有效列提取出来成为一张表格，其中用于分析的列有Game ID、Teams、Record、Score、Result、Player、Points、Rebounds、Assists、Steals、Blocks，有效数据821条。

