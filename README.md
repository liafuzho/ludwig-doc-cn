# [Ludwig](https://ludwig-ai.github.io/ludwig-docs/) 中文使用手册

<font color="silver">翻译自：</font><br>
[https://ludwig-ai.github.io/ludwig-docs/user_guide/](https://ludwig-ai.github.io/ludwig-docs/user_guide/)<br>

## 目录
* [命令行](#命令行)
  * [train](#train)
  * [predict](#predict)
  * [evaluate](#evaluate)
  * [experiment](#experiment)
  * [hyperopt](#hyperopt)
  * [serve](#serve)
  * [visualize](#visualize)
  * [collect_summary](#collectsummary)
  * [collect_weights](#collectweights)
  * [collect_activations](#collectactivations)
  * [export_savedmodel](#exportsavedmodel)
  * [export_neuropod](#exportneuropod)
  * [preprocess](#preprocess)
  * [synthesize_dataset](#synthesizedataset)
* [数据预处理](#数据预处理)
  * [数据集格式](#数据集格式)
* [数据后处理](#数据后处理)
* [配置](#配置)
  * [输入特征](#输入特征)
  * [合成器](#合成器)
  * [输出特征](#输出特征)
  * [训练](#训练)
  * [预处理](#预处理)
  * [Binary 特征](#Binary_特征)
  * [Numerical 特征](#Numerical_特征)
  * [Category 特征](#Category_特征)
  * [Set 特征](#Set_特征)
  * [Bag 特征](#Bag_特征)
  * [Sequence 特征](#Sequence_特征)
  * [Text 特征](#Text_特征)
  * [Time Series 特征](#Time_Series_特征)
  * [Audio 特征](#Audio_特征)
  * [Image 特征](#Image_特征)
  * [Date 特征](#Date_特征)
  * [H3 特征](#H3_特征)
  * [Vector 特征](#Vector_特征)
  * [组合器](#组合器)
* [分布式训练](#分布式训练)
* [超参数优化](#超参数优化)
  * [超参数](#超参数)
  * [采样器](#采样器)
  * [执行器](#执行器)
  * [完整的超参数优化示例](#完整的超参数优化示例)
* [集成](#集成)
* [编程接口【API】](#编程接口【API】)
  * [训练一个模型](#训练一个模型)
  * [加载一个预先训练好的模型](#加载一个预先训练好的模型)
  * [预测](#预测)
* [可视化](#可视化)
  * [学习曲线](#学习曲线)
  * [混淆矩阵](#混淆矩阵)
  * [性能比较](#性能比较)
  * [比较分类器的预测](#比较分类器的预测)
  * [可信阈值](#可信阈值)
  * [二进制阈值与度量](#二进制阈值与度量)
  * [ROC 曲线](#ROC_曲线)
  * [校准图](#校准图)
  * [类别频率与 F1 评分](#类别频率与_F1_评分)
  * [超参数优化的可视化](#超参数优化的可视化)
* [中文教程](https://github.com/liafuzho/ludwig-tutor-cn)

## 命令行<a id='命令行'></a>
Ludwig 提供了几个命令行入口点

* `train`：训练模型
* `predict`：使用训练好的模型进行预测
* `evaluate` ：评估一个训练好的模型性能
* `experiment`：进行完整的实验，训练模型并对其进行评估
* `serve`：为训练好的模型提供一个 http 服务
* `visualize`：实验结果可视化
* `hyperopt`：执行超参数优化
* `collect_summary`：打印权重和激活层的名称，以便与其他 collect 命令一起使用
* `collect_weights`：收集包含训练好的模型权值张量
* `collect_activations`：使用训练好的模型为每个数据点收集张量
* `export_savedmodel`：导出 Ludwig 模型为 SavedModel 格式
* `export_neuropod`：导出 Ludwig 模型为 Neuropod 格式
* `preprocess`：预处理数据并将其保存为 HDF5 和 JSON 格式
* `synthesize_dataset`：为测试目的创建合成数据

它们将在下面详细描述。

### train<a id='train'></a>
这个命令可以让您从您的数据中训练出一个模型，您可以在 Ludwig 的主目录中这样调用它：

```shell
ludwig train [options]
```
或

```shell
python -m ludwig.train [options]
```

以下是可用的参数：

```shell
usage: ludwig train [options]

This script trains a model

optional arguments:
  -h, --help            show this help message and exit
  --output_directory OUTPUT_DIRECTORY
                        directory that contains the results
  --experiment_name EXPERIMENT_NAME
                        experiment name
  --model_name MODEL_NAME
                        name for the model
  --dataset DATASET     input data file path. If it has a split column, it
                        will be used for splitting (0: train, 1: validation,
                        2: test), otherwise the dataset will be randomly split
  --training_set TRAINING_SET
                        input train data file path
  --validation_set VALIDATION_SET
                        input validation data file path
  --test_set TEST_SET   input test data file path
  --training_set_metadata TRAINING_SET_METADATA
                        input metadata JSON file path. An intermediate
                        preprocessed  containing the mappings of the input
                        file created the first time a file is used, in the
                        same directory with the same name and a .json
                        extension
  --data_format {auto,csv,excel,feather,fwf,hdf5,htmltables,json,jsonl,parquet,pickle,sas,spss,stata,tsv}
                        format of the input data
  -sspi, --skip_save_processed_input
                        skips saving intermediate HDF5 and JSON files
  -c CONFIG, --config CONFIG
                        config
  -cf CONFIG_FILE, --config_file CONFIG_FILE
                        YAML file describing the model. Ignores --config
  -mlp MODEL_LOAD_PATH, --model_load_path MODEL_LOAD_PATH
                        path of a pretrained model to load as initialization
  -mrp MODEL_RESUME_PATH, --model_resume_path MODEL_RESUME_PATH
                        path of the model directory to resume training of
  -sstd, --skip_save_training_description
                        disables saving the description JSON file
  -ssts, --skip_save_training_statistics
                        disables saving training statistics JSON file
  -ssm, --skip_save_model
                        disables saving weights each time the model improves.
                        By default Ludwig saves weights after each epoch the
                        validation metric imrpvoes, but if the model is really
                        big that can be time consuming. If you do not want to
                        keep the weights and just find out what performance
                        can a model get with a set of hyperparameters, use
                        this parameter to skip it
  -ssp, --skip_save_progress
                        disables saving weights after each epoch. By default
                        ludwig saves weights after each epoch for enabling
                        resuming of training, but if the model is really big
                        that can be time consuming and will save twice as much
                        space, use this parameter to skip it
  -ssl, --skip_save_log
                        disables saving TensorBoard logs. By default Ludwig
                        saves logs for the TensorBoard, but if it is not
                        needed turning it off can slightly increase the
                        overall speed
  -rs RANDOM_SEED, --random_seed RANDOM_SEED
                        a random seed that is going to be used anywhere there
                        is a call to a random number generator: data
                        splitting, parameter initialization and training set
                        shuffling
  -g GPUS [GPUS ...], --gpus GPUS [GPUS ...]
                        list of gpus to use
  -gml GPU_MEMORY_LIMIT, --gpu_memory_limit GPU_MEMORY_LIMIT
                        maximum memory in MB to allocate per GPU device
  -dpt, --disable_parallel_threads
                        disable TensorFlow from using multithreading for
                        reproducibility
  -uh, --use_horovod    uses horovod for distributed training
  -dbg, --debug         enables debugging mode
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

当 Ludwig 训练一个模型时，它创建两个中间文件，一个 HDF5 和一个 JSON。HDF5 文件包含映射到 numpy ndarray 的数据，而 JSON 文件包含从张量中的值到原始标签的映射。

例如，对于具有 3 个可能值的分类，HDF5 文件将包含从 0 到 3 的整数 (0表示 `<UNK>` 类别) ，而 JSON 文件将包含一个所有分类记号 (`[<UNK>, label_1, label_2, label_3]`) 的 `idx2str` 列表、一个 `str2idx` 字典 (`{"<UNK>": 0, "label_1": 1, "label_2": 2, "label_3": 3}`) 和一个 `str2freq` 字典 (`{"<UNK>": 0, "label_1": 93, "label_2": 55, "label_3": 24}`)。

拥有这些中间文件的原因有两方面：一方面，如果要再次训练模型，Ludwig 将尝试加载它们，而不是重新计算所有张量，这将节省大量时间；另一方面，如果要使用模型进行预测，数据必须以训练期间映射的完全相同的方式映射到张量，因此需要在 `predict` 命令中加载 JSON 元数据文件。其工作原理是：第一次提供 UTF-8编码的数据集 (`--dataset`) 时，创建 HDF5 和 JSON 文件，从第二次开始 Ludwig 将加载它们，而不是数据集，即使您指定了数据集(它以相同的方式为文件名指定相同的目录，但有不同的扩展名) ，最后您可以直接指定 HDF5 和 JSON 文件。

由于从原始数据到张量的映射取决于您在配置中指定的列类型，因此如果您更改类型(例如从 `sequence` 到 `text`) ，还必须重做预处理，这是通过删除 HDF5 和 JSON 文件来实现的。或者您指定参数 `--skip_save_processed_input` 来跳过保存 HDF5 和 JSON 文件。

训练集、验证集和测试集之间的分离可以通过几种方式实现。这允许一些可能的输入数据场景:

  * 提供了一个UTF-8编码的数据集文件 (`-dataset`)。在这种情况下，如果数据集包含一个值为 `0`(训练)、`1`(验证)和 `2`(测试)的拆分列，将使用这个拆分。如果您想忽略拆分列并执行随机拆分，请在配置中使用 `force_split ` 参数。如果没有拆分列，则执行随机的 `70-20-10` 拆分。您可以在配置预处理部分设置百分比并指定是否需要分层抽样。
  * 您可以提供单独的 UTF-8 编码的训练、验证和测试集 (`--training_set`, `--validation_set`, `--test_set`)。
  * 单个数据集文件中指定的 HDF5 和 JSON 文件标志也适用于多个文件的情况，唯一的区别是您只需要指定一个 JSON文件 (`--train_set_metadata_json`)。

验证集是可选的，但是如果缺少训练，训练将持续到训练周期结束，而当有验证集时，默认行为是在验证措施在一定周期没有改善之后提前停止。测试集也是可选的。

其他可选参数是 `--output_directory`，`--experiment_name` 和 `--model_name`。默认情况下，输出目录为 `./results`。如果指定了模型名和实验名，那么该目录将包含一个名为[ experiment _ name ] _ [ model _ name ] _ 0的目录。如果再次使用相同的实验和模型名称的组合，名称后面的整数将被增加。如果没有指定这两个目录，那么这个目录将命名为 `run_0`。该目录将包含

  * `description.json` 一份包含训练过程描述的文件，包含可以再现训练过程的所有信息
  * `training_statistics.json` 一份包含每个周期所有测量和损失记录的文件
  * `model ` 一个包含模型超参数、权重、检查点和日志的目录(用于 TensorBoard)

配置可以作为字符串 (`--config`) 或 YAML 文件 (`--config_file`) 提供。有关如何编写配置的详细信息在[配置](#配置)部分中提供。

在训练期间 Ludwig 为模型保存了两组权重，一组是在验证度量获得最佳性能的周期权重，另一组是最后周期的权重。保留第二组的原因是为了在训练过程被中断的情况下能够恢复训练。

要使用最新的权重和到目前为止的整个进展历史恢复训练，必须指定 `--model_resume_path` 参数。您可以使用参数 `--skip_save_progress` 避免保存最新的权重和到目前为止的全部进度，但之后将无法恢复它。另一个可用的选项是加载以前训练过的模型，作为新训练过程的初始化。在这种情况下，Ludwig 将开始一个新的训练过程，不知道前一个模型的任何进展，不知道训练统计数据，也不知道到目前为止模型已经训练了多少个周期。它不是恢复训练，而是使用先前训练过的模型和相同的配置初始化训练，它是通过 `--model_load_path argument` 参数来完成的。

您可以使用 `--random_seed` 参数指定 python 环境、 python random 包、 numpy 和 TensorFlow 使用的随机种子。这对于可重复性很有用。请注意，由于在 TensorFlow GPU 执行中的异步性，当在 GPU 上进行训练时，结果可能是不可重复的。

您可以使用 `--gpus` 参数来管理机器上的 GPUs，它接受一个与 `CUDA_VISIBLE_DEVICES ` 环境变量格式相同的字符串，即一个用逗号分隔的整数列表。您还可以指定最初分配给 TensorFlow 的 GPU 内存量，并设置 `--gpu_memory_limit `。默认情况下，所有内存都已分配。如果少于所有的内存被分配，TensorFlow 将需要更多的 GPU 内存，它将尝试增加这个数量。

如果参数 `--use_horovod` 设置为 `true`，将使用 Horovod 进行分布式处理。

最后，`--logging_level` 参数允许您设置在培训期间希望看到的日志数量，`--debug` 参数启用 TensorFlow 的 `tfdbg`。这样做时要小心，因为它将有助于捕获错误，特别是 `infs` 和 `NaNs`，但它将消耗更多的内存。

示例：

```shell
ludwig train --dataset reuters-allcats.csv --config "{input_features: [{name: text, type: text, encoder: parallel_cnn, level: word}], output_features: [{name: class, type: category}]}"
```

### predict<a id='predict'></a>
这个命令可以让您使用一个以前训练过的模型来预测新的数据，您可以在 Ludwig 的主目录中这样调用它：

```shell
ludwig predict [options]
```

或：

```shell
python -m ludwig.predict [options]
```

以下是可用参数：

```shell
usage: ludwig predict [options]

This script loads a pretrained model and uses it to predict

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     input data file path
  --data_format {auto,csv,excel,feather,fwf,hdf5,htmltables,json,jsonl,parquet,pickle,sas,spss,stata,tsv}
                        format of the input data
  -s {training,validation,test,full}, --split {training,validation,test,full}
                        the split to test the model on
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        directory that contains the results
  -ssuo, --skip_save_unprocessed_output
                        skips saving intermediate NPY output files
  -sstp, --skip_save_predictions
                        skips saving predictions CSV files
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        size of batches
  -g GPUS, --gpus GPUS  list of gpu to use
  -gml GPU_MEMORY_LIMIT, --gpu_memory_limit GPU_MEMORY_LIMIT
                        maximum memory in MB to allocate per GPU device
  -dpt, --disable_parallel_threads
                        disable TensorFlow from using multithreading for
                        reproducibility
  -uh, --use_horovod    uses horovod for distributed training
  -dbg, --debug         enables debugging mode
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

[train](#train) 一节中解释的 UTF-8 编码的数据集文件和 HDF5/JSON 文件之间的区别在这里也同样适用。在这两种情况下，都需要在训练期间获得 JSON 元数据文件，以便将新数据映射到张量。如果新数据包含拆分列，则可以使用 `--split` 参数指定要使用哪个拆分来计算预测。默认情况下，所有的拆分都将被使用。

需要加载一个模型，您可以使用 `--model_path` 参数指定它的路径。如果您以前训练过一个模型，并且得到了结果。例如 `./results/experiment_run_0`，您必须指定 `./results/experiment_run_0/model` 来使用它进行预测。

您可以使用参数 `--output-directory` 指定输出目录，默认情况下它是 `./result_0` 如果存在具有相同名称的目录，则使用递增的数字。

该目录将包含一个预测 CSV 文件和每个输出特征的概率 CSV 文件，以及包含原始张量的原始 NPY 文件。您可以使用参数 `--skip_save_unprocessed_output` 指定不保存原始 NPY 输出文件。

可以使用参数 `--batch_size` 指定加速预测的特定批量大小。

最后 `--logging_level`, `--debug ` `--gpus `, `--gpu_memory_limit` 和 `--disable_parallel_threads` 相关参数的行为与 [train](#train) 命令一节中描述的行为完全相同。

示例：

```shell
ludwig predict --dataset reuters-allcats.csv --model_path results/experiment_run_0/model/
```

### evaluate<a id='evaluate'></a>
此命令允许您使用以前训练过的模型来预测新数据，并评估预测与实际情况相比的性能。您可以在 Ludwig 的主目录中这样调用它：

```shell
ludwig evaluate [options]
```

或：

```shell
python -m ludwig.evaluate [options]
```

以下是可用的参数：

```shell
usage: ludwig evaluate [options]

This script loads a pretrained model and evaluates its performance by
comparingits predictions with ground truth.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     input data file path
  --data_format {auto,csv,excel,feather,fwf,hdf5,htmltables,json,jsonl,parquet,pickle,sas,spss,stata,tsv}
                        format of the input data
  -s {training,validation,test,full}, --split {training,validation,test,full}
                        the split to test the model on
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        directory that contains the results
  -ssuo, --skip_save_unprocessed_output
                        skips saving intermediate NPY output files
  -sses, --skip_save_eval_stats
                        skips saving intermediate JSON eval statistics
  -scp, --skip_collect_predictions
                        skips collecting predictions
  -scos, --skip_collect_overall_stats
                        skips collecting overall stats
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        size of batches
  -g GPUS, --gpus GPUS  list of gpu to use
  -gml GPU_MEMORY_LIMIT, --gpu_memory_limit GPU_MEMORY_LIMIT
                        maximum memory in MB to allocate per GPU device
  -dpt, --disable_parallel_threads
                        disable TensorFlow from using multithreading for
                        reproducibility
  -uh, --use_horovod    uses horovod for distributed training
  -dbg, --debug         enables debugging mode
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

所有的参数及其行为与 [predict](#predict) 是相同的。惟一的区别是 `evaluate` 要求数据集还包含具有相同名称的输出列。这是必要的，因为 `evaluate` 将模型产生的预测与实际情况进行比较，并将所有这些统计数据保存在结果目录中的 `test_statistics.json` 文件。

请注意，为了计算模型性能数据，数据必须包含每个输出列的正确值。如果您收到一个关于数据中缺少输出列的错误，这意味着数据不包含每个输出列的正确值。

示例：
```shell
ludwig evaluate --dataset reuters-allcats.csv --model_path results/experiment_run_0/model/
```

### experiment<a id='experiment'></a>
这个命令将训练和评估结合成一个简单的命令。您可以通过指定 `--k_fold` 参数来请求 k-fold 交叉验证。您可以在 Ludwig 的主目录中这样调用它：

```shell
ludwig experiment [options]
```

或：

```shell
python -m ludwig.experiment [options]
```

以下是可用的参数：

```shell
usage: ludwig experiment [options]

This script trains and evaluates a model

optional arguments:
  -h, --help            show this help message and exit
  --output_directory OUTPUT_DIRECTORY
                        directory that contains the results
  --experiment_name EXPERIMENT_NAME
                        experiment name
  --model_name MODEL_NAME
                        name for the model
  --dataset DATASET     input data file path. If it has a split column, it
                        will be used for splitting (0: train, 1: validation,
                        2: test), otherwise the dataset will be randomly split
  --training_set TRAINING_SET
                        input train data file path
  --validation_set VALIDATION_SET
                        input validation data file path
  --test_set TEST_SET   input test data file path
  --training_set_metadata TRAINING_SET_METADATA
                        input metadata JSON file path. An intermediate
                        preprocessed  containing the mappings of the input
                        file created the first time a file is used, in the
                        same directory with the same name and a .json
                        extension
  --data_format {auto,csv,excel,feather,fwf,hdf5,htmltables,json,jsonl,parquet,pickle,sas,spss,stata,tsv}
                        format of the input data
  -es {training,validation,test,full}, --eval_split {training,validation,test,full}
                        the split to evaluate the model on
  -sspi, --skip_save_processed_input
                        skips saving intermediate HDF5 and JSON files
  -ssuo, --skip_save_unprocessed_output
                        skips saving intermediate NPY output files
  -kf K_FOLD, --k_fold K_FOLD
                        number of folds for a k-fold cross validation run
  -skfsi, --skip_save_k_fold_split_indices
                        disables saving indices generated to split training
                        data set for the k-fold cross validation run, but if
                        it is not needed turning it off can slightly increase
                        the overall speed
  -c CONFIG, --config CONFIG
                        config
  -cf CONFIG_FILE, --config_file CONFIG_FILE
                        YAML file describing the model. Ignores
                        --model_hyperparameters
  -mlp MODEL_LOAD_PATH, --model_load_path MODEL_LOAD_PATH
                        path of a pretrained model to load as initialization
  -mrp MODEL_RESUME_PATH, --model_resume_path MODEL_RESUME_PATH
                        path of the model directory to resume training of
  -sstd, --skip_save_training_description
                        disables saving the description JSON file
  -ssts, --skip_save_training_statistics
                        disables saving training statistics JSON file
  -sstp, --skip_save_predictions
                        skips saving test predictions CSV files
  -sstes, --skip_save_eval_stats
                        skips saving eval statistics JSON file
  -ssm, --skip_save_model
                        disables saving model weights and hyperparameters each
                        time the model improves. By default Ludwig saves model
                        weights after each epoch the validation metric
                        imprvoes, but if the model is really big that can be
                        time consuming if you do not want to keep the weights
                        and just find out what performance a model can get
                        with a set of hyperparameters, use this parameter to
                        skip it,but the model will not be loadable later on
  -ssp, --skip_save_progress
                        disables saving progress each epoch. By default Ludwig
                        saves weights and stats after each epoch for enabling
                        resuming of training, but if the model is really big
                        that can be time consuming and will uses twice as much
                        space, use this parameter to skip it, but training
                        cannot be resumed later on
  -ssl, --skip_save_log
                        disables saving TensorBoard logs. By default Ludwig
                        saves logs for the TensorBoard, but if it is not
                        needed turning it off can slightly increase the
                        overall speed
  -rs RANDOM_SEED, --random_seed RANDOM_SEED
                        a random seed that is going to be used anywhere there
                        is a call to a random number generator: data
                        splitting, parameter initialization and training set
                        shuffling
  -g GPUS [GPUS ...], --gpus GPUS [GPUS ...]
                        list of GPUs to use
  -gml GPU_MEMORY_LIMIT, --gpu_memory_limit GPU_MEMORY_LIMIT
                        maximum memory in MB to allocate per GPU device
  -dpt, --disable_parallel_threads
                        disable TensorFlow from using multithreading for
                        reproducibility
  -uh, --use_horovod    uses horovod for distributed training
  -dbg, --debug         enables debugging mode
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

这些参数结合了来自 [train](#train) 和 [test](#test) 的参数，请参考这些章节以获得更深入的解释。输出目录将包含两个命令产生的输出。

示例：
```shell
ludwig experiment --dataset reuters-allcats.csv --config "{input_features: [{name: text, type: text, encoder: parallel_cnn, level: word}], output_features: [{name: class, type: category}]}"
```

### hyperopt<a id='hyperopt'></a>
这个命令可以让您使用给定的采样器和参数执行超参数搜索。您可以在 Ludwig 的主目录中这样调用它：

```shell
ludwig hyperopt [options]
```

或：

```shell
python -m ludwig.hyperopt [options]
```

以下是可用的参数：

```shell
usage: ludwig hyperopt [options]

This script searches for optimal Hyperparameters

optional arguments:
  -h, --help            show this help message and exit
  -sshs, --skip_save_hyperopt_statistics
                        skips saving hyperopt statistics file
  --output_directory OUTPUT_DIRECTORY
                        directory that contains the results
  --experiment_name EXPERIMENT_NAME
                        experiment name
  --model_name MODEL_NAME
                        name for the model
  --dataset DATASET     input data file path. If it has a split column, it
                        will be used for splitting (0: train, 1: validation,
                        2: test), otherwise the dataset will be randomly split
  --training_set TRAINING_SET
                        input train data file path
  --validation_set VALIDATION_SET
                        input validation data file path
  --test_set TEST_SET   input test data file path
  --training_set_metadata TRAINING_SET_METADATA
                        input metadata JSON file path. An intermediate
                        preprocessed file containing the mappings of the input
                        file created the first time a file is used, in the
                        same directory with the same name and a .json
                        extension
  --data_format {auto,csv,excel,feather,fwf,hdf5,htmltables,json,jsonl,parquet,pickle,sas,spss,stata,tsv}
                        format of the input data
  -sspi, --skip_save_processed_input
                        skips saving intermediate HDF5 and JSON files
  -c CONFIG, --config CONFIG
                        config
  -cf CONFIG_FILE, --config_file CONFIG_FILE
                        YAML file describing the model. Ignores
                        --model_hyperparameters
  -mlp MODEL_LOAD_PATH, --model_load_path MODEL_LOAD_PATH
                        path of a pretrained model to load as initialization
  -mrp MODEL_RESUME_PATH, --model_resume_path MODEL_RESUME_PATH
                        path of the model directory to resume training of
  -sstd, --skip_save_training_description
                        disables saving the description JSON file
  -ssts, --skip_save_training_statistics
                        disables saving training statistics JSON file
  -ssm, --skip_save_model
                        disables saving weights each time the model improves.
                        By default Ludwig saves weights after each epoch the
                        validation metric imrpvoes, but if the model is really
                        big that can be time consuming. If you do not want to
                        keep the weights and just find out what performance
                        can a model get with a set of hyperparameters, use
                        this parameter to skip it
  -ssp, --skip_save_progress
                        disables saving weights after each epoch. By default
                        ludwig saves weights after each epoch for enabling
                        resuming of training, but if the model is really big
                        that can be time consuming and will save twice as much
                        space, use this parameter to skip it
  -ssl, --skip_save_log
                        disables saving TensorBoard logs. By default Ludwig
                        saves logs for the TensorBoard, but if it is not
                        needed turning it off can slightly increase the
                        overall speed
  -rs RANDOM_SEED, --random_seed RANDOM_SEED
                        a random seed that is going to be used anywhere there
                        is a call to a random number generator: data
                        splitting, parameter initialization and training set
                        shuffling
  -g GPUS [GPUS ...], --gpus GPUS [GPUS ...]
                        list of gpus to use
  -gml GPU_MEMORY_LIMIT, --gpu_memory_limit GPU_MEMORY_LIMIT
                        maximum memory in MB to allocate per GPU device
  -uh, --use_horovod    uses horovod for distributed training
  -dbg, --debug         enables debugging mode
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

这些参数结合了来自 [train](#train) 和 [test](#test) 的参数，因此请参考这些章节以获得更深入的解释。输出目录将包含一个 `hyperopt_statistics.json` 文件，它总结了所获得的结果。

为了执行超参数优化，需要在配置中提供 `hyperopt` 配置。在 `hyperopt` 配置中，您将能够定义要优化的指标、参数、使用什么采样器来优化它们以及如何执行优化。关于 `hyperopt` 配置的详细信息请参见 [超参数优化](#超参数优化) 章节中的详细描述。

### serve<a id='serve'></a>
这个命令允许您加载一个预先训练好的模型并在 http 服务器上提供它。您可以在 Ludwig 的主目录中这样调用它：

```shell
ludwig serve [options]
```

或：

```shell
python -m ludwig.serve [options]
```

以下是可用的参数：

```shell
usage: ludwig serve [options]

This script serves a pretrained model

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
  -p PORT, --port PORT  port for server (default: 8000)
  -H HOST, --host HOST  host for server (default: 0.0.0.0)
```

最重要的参数是 `--model_path` 在这里您必须指定要加载的模型路径。

运行之后，可以在 `/predict` 节点上发出 POST 请求，以对提交的表单数据进行推断。

#### CURL 示例：

##### 文件

```shell
curl http://0.0.0.0:8000/predict -X POST -F 'image_path=@path_to_image/example.png'
```

##### 文字

```shell
curl http://0.0.0.0:8000/predict -X POST -F 'english_text=words to be translated'
```

##### 文字和文件

```shell
curl http://0.0.0.0:8000/predict -X POST -F 'text=mixed together with' -F 'image=@path_to_image/example.png'
```

##### 批量预测
您还可以对 `/batch_predict` 节点发出 POST 请求，以便一次对多个样本进行推理。

请求必须作为表单数据提交，其中一个字段是 `dataset`： 要预测数据的 JSON 编码字符串形式。

JSON 数据集的字符串应该是 Pandas "split" 格式，以减少负载大小。这种格式将数据集分为三个部分：

  1. columns：`List[str]`
  2. index (optional)：`List[Union[str, int]]`
  3. data：`List[List[object]]`

可以使用其他表单字段提供文件资源，比如数据集中引用的图像。

批量预测的例子：

```shell
curl http://0.0.0.0:8000/batch_predict -X POST -F 'dataset={"columns": ["a", "b"], "data": [[1, 2], [3, 4]]}'
```

### visualize<a id='visualize'></a>
您可以通过这个命令将训练和预测统计数据可视化，同时比较不同模型的性能和预测。您可以在 Ludwig 的主目录中这样调用它：

```shell
ludwig visualize [options]
```

以下是可用的参数：

```shell
usage: ludwig visualize [options]

This script analyzes results and shows some nice plots.

optional arguments:
  -h, --help            show this help message and exit
  -g GROUND_TRUTH, --ground_truth GROUND_TRUTH
                        ground truth file
  -gm GROUND_TRUTH_METADATA, --ground_truth_metadata GROUND_TRUTH_METADATA
                        input metadata JSON file
  -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        directory where to save plots.If not specified, plots
                        will be displayed in a window
  -ff {pdf,png}, --file_format {pdf,png}
                        file format of output plots
  -v {binary_threshold_vs_metric,calibration_1_vs_all,calibration_multiclass,compare_classifiers_multiclass_multimetric,compare_classifiers_performance_changing_k,compare_classifiers_performance_from_pred,compare_classifiers_performance_from_prob,compare_classifiers_performance_subset,compare_classifiers_predictions,compare_classifiers_predictions_distribution,compare_performance,confidence_thresholding,confidence_thresholding_2thresholds_2d,confidence_thresholding_2thresholds_3d,confidence_thresholding_data_vs_acc,confidence_thresholding_data_vs_acc_subset,confidence_thresholding_data_vs_acc_subset_per_class,confusion_matrix,frequency_vs_f1,hyperopt_hiplot,hyperopt_report,learning_curves,roc_curves,roc_curves_from_test_statistics}, --visualization {binary_threshold_vs_metric,calibration_1_vs_all,calibration_multiclass,compare_classifiers_multiclass_multimetric,compare_classifiers_performance_changing_k,compare_classifiers_performance_from_pred,compare_classifiers_performance_from_prob,compare_classifiers_performance_subset,compare_classifiers_predictions,compare_classifiers_predictions_distribution,compare_performance,confidence_thresholding,confidence_thresholding_2thresholds_2d,confidence_thresholding_2thresholds_3d,confidence_thresholding_data_vs_acc,confidence_thresholding_data_vs_acc_subset,confidence_thresholding_data_vs_acc_subset_per_class,confusion_matrix,frequency_vs_f1,hyperopt_hiplot,hyperopt_report,learning_curves,roc_curves,roc_curves_from_test_statistics}
                        type of visualization
  -f OUTPUT_FEATURE_NAME, --output_feature_name OUTPUT_FEATURE_NAME
                        name of the output feature to visualize
  -gts GROUND_TRUTH_SPLIT, --ground_truth_split GROUND_TRUTH_SPLIT
                        ground truth split - 0:train, 1:validation, 2:test
                        split
  -tf THRESHOLD_OUTPUT_FEATURE_NAMES [THRESHOLD_OUTPUT_FEATURE_NAMES ...], --threshold_output_feature_names THRESHOLD_OUTPUT_FEATURE_NAMES [THRESHOLD_OUTPUT_FEATURE_NAMES ...]
                        names of output features for 2d threshold
  -pred PREDICTIONS [PREDICTIONS ...], --predictions PREDICTIONS [PREDICTIONS ...]
                        predictions files
  -prob PROBABILITIES [PROBABILITIES ...], --probabilities PROBABILITIES [PROBABILITIES ...]
                        probabilities files
  -trs TRAINING_STATISTICS [TRAINING_STATISTICS ...], --training_statistics TRAINING_STATISTICS [TRAINING_STATISTICS ...]
                        training stats files
  -tes TEST_STATISTICS [TEST_STATISTICS ...], --test_statistics TEST_STATISTICS [TEST_STATISTICS ...]
                        test stats files
  -hs HYPEROPT_STATS_PATH, --hyperopt_stats_path HYPEROPT_STATS_PATH
                        hyperopt stats file
  -mn MODEL_NAMES [MODEL_NAMES ...], --model_names MODEL_NAMES [MODEL_NAMES ...]
                        names of the models to use as labels
  -tn TOP_N_CLASSES [TOP_N_CLASSES ...], --top_n_classes TOP_N_CLASSES [TOP_N_CLASSES ...]
                        number of classes to plot
  -k TOP_K, --top_k TOP_K
                        number of elements in the ranklist to consider
  -ll LABELS_LIMIT, --labels_limit LABELS_LIMIT
                        maximum numbers of labels. If labels in dataset are
                        higher than this number, "rare" label
  -ss {ground_truth,predictions}, --subset {ground_truth,predictions}
                        type of subset filtering
  -n, --normalize       normalize rows in confusion matrix
  -m METRICS [METRICS ...], --metrics METRICS [METRICS ...]
                        metrics to dispay in threshold_vs_metric
  -pl POSITIVE_LABEL, --positive_label POSITIVE_LABEL
                        label of the positive class for the roc curve
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

正如 `--visualization` 参数所表明的那样，有大量的可视化效果可供使用。它们中的每一个都需要该命令参数的不同子集，因此它们将在 [可视化](#可视化) 章节中逐一描述。

### collect_summary<a id='collectsummary'></a>
这个命令加载一个预先训练好的模型，并打印权重和激活层的名称，以便与 `collect_weights ` 或 `collect_activations ` 一起使用。您可以在 Ludwig 的主目录中这样调用它：

```shell
ludwig collect_summary [options]
```

或：

```shell
python -m ludwig.collect names [options]
```

以下是可用的参数：

```shell
usage: ludwig collect_summary [options]

This script loads a pretrained model and print names of weights and layer activations.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

### collect_weights<a id='collectweights'></a>
这个命令允许您加载一个预先训练好的模型，并使用特定的名称收集张量，以便以 NPY 格式保存它们。这对于可视化学习权重(例如收集嵌入矩阵)和一些事后分析可能是有用的。您可以在 Ludwig 的主目录中这样调用它：

```shell
ludwig collect_weights [options]
```

或：

```shell
python -m ludwig.collect weights [options]
```

以下是可用的参数：

```shell
usage: ludwig collect_weights [options]

This script loads a pretrained model and uses it collect weights.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -t TENSORS [TENSORS ...], --tensors TENSORS [TENSORS ...]
                        tensors to collect
  -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        directory that contains the results
  -dbg, --debug         enables debugging mode
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

三个最重要的参数—— `--model_path` 您必须指定要加载的模型路径, `--tensors` 这允许您在 TensorFlow 图中指定一个张量名称列表，其中包含您想要收集的权重, 最后, `--output_directory` 允许您指定 NPY 文件的保存位置(每个张量名称对应一个)。

为了找出包含您想要收集的权重的张量名称，最好的方法是用 TensorBoard 检查模型图。

或者使用 `collect_summary ` 命令。

### collect_activations<a id='collectactivations'></a>
这个命令允许您加载预先训练好的模型和输入数据，并收集具有特定名称的张量中包含的激活值，以便以 NPY 格式保存它们。这对于可视化激活(例如收集最后一层的激活作为输入数据点的嵌入表示)和一些事后分析可能很有用。您可以在 Ludwig 的主目录中这样调用它：

```shell
ludwig collect_activations [options]
```

或：

```shell
python -m ludwig.collect activations [options]
```

以下是可用的参数：

```shell
usage: ludwig collect_activations [options]

This script loads a pretrained model and uses it collect tensors for each
datapoint in the dataset.

optional arguments:
  -h, --help            show this help message and exit
  --dataset  DATASET    filepath for input dataset
  --data_format DATA_FORMAT  format of the dataset.  Valid values are auto,
                        csv, excel, feature, fwf, hdf5, html, tables, json,
                        json, jsonl, parquet, pickle, sas, spss, stata, tsv
  -s {training,validation,test,full}, --split {training,validation,test,full}
                        the split to test the model on
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -lyr LAYER [LAYER ..], --layers LAYER [LAYER ..]
                        layers to collect
  -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        directory that contains the results
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        size of batches
  -g GPUS, --gpus GPUS  list of gpu to use
  -gml GPU_MEMORY, --gpu_memory_limit GPU_MEMORY
                        maximum memory in MB of gpu memory to allocate per
                        GPU device
  -dpt, --disable_parallel_threads
                        disable Tensorflow from using multithreading
                        for reproducibility
  -uh, --use_horovod    uses horovod for distributed training
  -dbg, --debug         enables debugging mode
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

与数据相关的参数和与运行时相关的参数(GPUs, 批量大小等) 与在 [predict](#predict) 中使用相同，您可以参考该章节。”收集“的特定参数 `--model_path`，`--tensors` 和 `--output_directory` 与 [collect_weights](#collectweights) 对应的参数使用方法相同，您可以参考该章节以获得解释。

为了找出包含您想要收集的权重的张量名称，最好的方法是用 TensorBoard 检查模型图。

```shell
tensorboard --logdir /path/to/model/log
```

### export_savedmodel<a id='exportsavedmodel'></a>
将预先训练好的模型导出为 Tensorflow SavedModel 格式。

```shell
ludwig export_savedmodel [options]
```

或：

```shell
python -m ludwig.export savedmodel [options]
```

以下是可用的参数：

```shell
usage: ludwig export_savedmodel [options]

This script loads a pretrained model and uses it collect weights.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -od OUTPUT_PATH, --output_path OUTPUT_PATH
                        path where to save the export model  
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

### export_neuropod<a id='exportneuropod'></a>
一个 Ludwig 模型可以导出为一个 [Neuropod](https://github.com/uber/neuropod)，这种机制允许在一种不可知方式的框架内运行*<small>(详情参见 Neuropod, 这里说的很笼统——译者注)</small>*。

为了将 Ludwig 模型导出为 Neuropod，首先确保 `neuropod` 包和相应的后端一起安装在您的环境中(仅使用 Python 3.7+)，然后运行以下命令：

```shell
ludwig export_neuropod [options]
```

或：

```shell
python -m ludwig.export neuropod [options]
```

以下是可用的参数：

```shell
usage: ludwig export_neuropod [options]

This script loads a pretrained model and uses it collect weights.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        model to load
  -mn MODEL_NAME, --model_name MODEL_NAME
                        model name
  -od OUTPUT_PATH, --output_path OUTPUT_PATH
                        path where to save the export model  
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```


这个功能已经用 `neuropod==0.2.0` 测试过了。

### preprocess<a id='preprocess'></a>
预处理数据并将其保存为 HDF5和 JSON 格式。预处理后的文件可用于执行培训、预测和评估。这样做的好处是，作为已经预处理过的数据，如果需要针对同一数据训练多个模型，预处理文件就会充当缓存，从而避免多次执行预处理。

```shell
ludwig preprocess [options]
```

或：

```shell
python -m ludwig.preprocess [options]
```
以下是可用的参数：

```shell
usage: ludwig preprocess [options]

This script preprocess a dataset

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     input data file path. If it has a split column, it
                        will be used for splitting (0: train, 1: validation,
                        2: test), otherwise the dataset will be randomly split
  --training_set TRAINING_SET
                        input train data file path
  --validation_set VALIDATION_SET
                        input validation data file path
  --test_set TEST_SET   input test data file path
  --training_set_metadata TRAINING_SET_METADATA
                        input metadata JSON file path. An intermediate
                        preprocessed  containing the mappings of the input
                        file created the first time a file is used, in the
                        same directory with the same name and a .json
                        extension
  --data_format {auto,csv,excel,feather,fwf,hdf5,htmltables,json,jsonl,parquet,pickle,sas,spss,stata,tsv}
                        format of the input data
  -pc PREPROCESSING_CONFIG, --preprocessing_config PREPROCESSING_CONFIG
                        preproceesing config. Uses the same format of config,
                        but ignores encoder specific parameters, decoder
                        specific paramters, combiner and training parameters
  -pcf PREPROCESSING_CONFIG_FILE, --preprocessing_config_file PREPROCESSING_CONFIG_FILE
                        YAML file describing the preprocessing. Ignores
                        --preprocessing_config.Uses the same format of config,
                        but ignores encoder specific parameters, decoder
                        specific paramters, combiner and training parameters
  -rs RANDOM_SEED, --random_seed RANDOM_SEED
                        a random seed that is going to be used anywhere there
                        is a call to a random number generator: data
                        splitting, parameter initialization and training set
                        shuffling
  -dbg, --debug         enables debugging mode
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

### synthesize_dataset<a id='synthesizedataset'></a>
根据 YAML 格式提供的特征列参数创建用于测试目的的合成数据。

```shell
ludwig synthesize_dataset [options]
```

或：

```shell
python -m ludwig.data.dataset_synthesizer [options]
```

以下是可用的参数：

```shell
usage: ludwig synthesize_dataset [options]

This script generates a synthetic dataset.

optional arguments:
  -h, --help            show this help message and exit
  -od OUTPUT_PATH, --output_path OUTPUT_PATH
                        output CSV file path
  -d DATASET_SIZE, --dataset_size DATASET_SIZE
                        size of the dataset
  -f FEATURES, --features FEATURES
                        list of features to generate in YAML format. Provide a
                        list containing one dictionary for each feature, each
                        dictionary must include a name, a type and can include
                        some generation parameters depending on the type

Process finished with exit code 0
```

特征列文件应该为每个特征包含一个条目字典，以及它的名称和类型，还有可选的超参数。

```yaml
-
  name: first_feature
  type: first_feature_type
-
  name: second_feature
  type: second_feature_type
...
```

可用的参数取决于特征类型。

#### binary
  * `prob` (float，默认值: `0.5`)：生成 `true` 概率。
  * `cycle` (boolean，默认值: `false`)：循环遍历值而不是抽样。

#### numerical
  * `min` (float，默认值: `0`)：要生成的值其范围的最小值。
  * `max` (float，默认值: `1`)：要生成的值其范围的最大值。

#### category
  * `vocab_size` (int，默认值: `10`)：样本词汇的大小。
  * `cycle` (boolean，默认值: `false`)：循环遍历值而不是抽样。

#### sequence
  * `vocab_size` (int，默认值: `10`)：样本词汇的大小。
  * `max_len` (int，默认值: `10`)：生成序列的最大长度。
  * `min_len` (int，默认值: `null`)：如果为null，则所有序列的大小都为 `max_len`。如果提供了一个值，长度将在 `min_len` 和 `max_len` 之间随机确定。

#### set
  * `vocab_size` (int，默认值: `10`)：样本词汇的大小。
  * `max_len` (int，默认值: `10`)：生成集的最大长度。

#### bag
  * `vocab_size` (int，默认值: `10`)：样本词汇的大小。
  * `max_len` (int，默认值: `10`)：生成集的最大长度。

#### text
  * `vocab_size` (int，默认值: `10`)：样本词汇的大小。
  * `max_len` (int，默认值: `10`)：生成序列的最大长度，长度将在 `max_len - 20%` 和 `max_len` 之间随机采样。

#### timeseries
  * `max_len` (int，默认值: `10`)：生成序列的最大长度。
  * `min` (float，默认值: `0`)：要生成的值其范围的最小值。
  * `max` (float，默认值: `1`)：要生成的值其范围的最大值。

#### audio
  * `destination_folder` (str)：存放生成的音频文件的文件夹。
  * `preprocessing: {audio_file_length_limit_in_s}` (int，默认值: `1`)：生成的音频长度，以秒为单位。

#### image
  * `destination_folder` (str)：存放生成的图像文件的文件夹。
  * `preprocessing: {height}` (int，默认值: `28`)：生成图像的高度(以像素为单位)。
  * `preprocessing: {width}` (int，默认值: `28`)：生成图像的宽度(以像素为单位)。
  * `preprocessing: {num_channels}` (int，默认值: `1`)：生成图像的通道数。有效值为 `1`，`3`，`4`。

#### date
没有参数。

#### h3
没有参数。

#### vector
  * `vector_size` (int，默认值: `10`)：要生成的向量大小。

## 数据预处理<a id='数据预处理'></a>
Ludwig 可以从 14 种*<small>(数了下，只有13种——译者注)</small>*文件格式读取 UTF-8 编码的数据，支持的格式有：

  * Comma Separated Values (`csv`)
  * Excel Workbooks (`excel`)
  * Feather (`feather`)
  * Fixed Width Format (`fwf`)
  * Hierarchical Data Format 5 (`hdf5`)
  * Hypertext Markup Language (`html`) **注意**: 仅限于文件中的单个表。
  * JavaScript Object Notation (`json and jsonl`)
  * Parquet (`parquet`)
  * Pickled Pandas DataFrame (`pickle`)
  * SAS data sets in XPORT or SAS7BDAT format (`sas`)
  * SPSS file (`spss`)
  * Stata file (`stata`)
  * Tab Separated Values (`tsv`)

Ludwig 数据预处理将支持数据集中的原始数据映射到包含张量的 HDF5 文件，当有需要时也包含从字符串映射到张量的 JSON 文件。如果提供UTF-8编码的数据作为输入，并且HDF5和JSON文件都保存在与输入数据集相同的目录中，则执行此映射，除非使用参数 `--skip_save_processed_input` (在 `train` 和 `experiment` 命令中都使用)。保存这些文件的原因是为了提供缓存和避免再次执行预处理(因为，这取决于涉及的特征类型，它可能很耗时)，并提供所需的映射，以便能够将不可见的数据映射到张量中。

预处理过程是个性化的，可以适应数据格式的具体要求，但基本的假设是，UTF-8 编码的数据集对于每个数据点包含一行，对于每个特征(输入或输出)包含一列，并且您能够在 Ludwig 支持的列中确定该列的类型。原因是每个数据类型都以不同的方式映射到张量，并期望内容以特定的方式格式化。不同的数据类型可能有不同的分词器来格式化单元格的值。

例如，序列特征列的单元格的值默认由 `space` 分词器(tokenizer)管理，该分词器使用空格将值的内容分割成一个字符串列表。

| 分词前 | 分词后 | 
| :-------- | :----- |
| "token3 token4 token2"	| [token3, token4, token2] |
| "token3 token1"	 | [token3, token1] |

然后创建一个列表 idx2str 和两个字典 str2idx 和 str2freq，它们包含通过拆分列的所有行获得的所有列表中的所有标记(token)，并为每个列表分配一个整数 id (按频率顺序)。

```yaml
{
    "column_name": {
        "idx2str": [
            "<PAD>",
            "<UNK>",
            "token3",
            "token2",
            "token4",
            "token1"
        ],
        "str2idx": {
            "<PAD>": 0,
            "<UNK>": 1,
            "token3": 2,
            "token2": 3,
            "token4": 4,
            "token1": 5
        },
        "str2freq": {
            "<PAD>":  0,
            "<UNK>":  0,
            "token3": 2,
            "token2": 1,
            "token4": 1,
            "token1": 1
        }
    }
}
```

最后，创建一个 `n x l` numpy 矩阵，其中 `n` 是列中的行数，`l` 是由 `max_length` 参数设置的最长的最小分词列表,。所有比 `l` 短的序列都在右边填充(但是这种行为也可以通过一个参数来修改)。

| 分词后 | numpy 矩阵 | 
| :-------- | :----- |
| [token3, token4, token2]	| 2 4 3 |
| [token3, token1] | 2 5 0 |

最终的结果矩阵以数据集中原始列的名称作为键名保存在 HDF5 中，而从标记到整数 ID (及其逆映射)的映射则保存在 JSON 文件中。

每种数据类型都以不同的方式进行预处理，使用不同的参数和不同的分词器。关于如何为每种特征类型和每种特定特征设置这些参数的详细信息将在[配置——预处理](#预处理)章节中描述。

`Binary` 特征直接转换为长度为 `n` 的二进制值向量(其中 `n` 是数据集的大小) ，并通过数据集中的列名做为键名添加到 HDF5 中。JSON 元数据文件中没有关于它们的额外信息。

`Numerical` 特征直接转换为长度为 `n` 的浮点值向量(其中 `n` 是数据集的大小)， 并通过数据集中的列名做为键名添加到 HDF5 中。JSON 元数据文件中没有关于它们的额外信息。

`Category` 特征被转换为大小为 `n` 的整数值向量(其中 `n` 是数据集的大小) ，并通过数据集中的列名做为键名添加到 HDF5 中。将类别映射到整数的方法包括：首先收集数据集列中所有不同类别字符串的字典，然后按照频率对它们进行排序，然后将它们按频率由最频繁到最罕见的顺序递增的分配一个整数 ID (0 分配给一个 `<UNK>` 标记)。列名将被添加到 JSON 文件中，并带有一个关联字典，其中包含：

  1. integer 到 string 映射 (`idx2str`)
  2. string 到 id 的映射 (`str2idx`)
  3. string 到 frequency 的映射 (`str2freq`)
  4. 所有标记集的大小 (`vocab_size`)
  5. 额外的预处理信息 (额缺省情况下如何填充缺少的值以及使用什么标记来填充缺少的值)

`Set` 特征被转换为一个 `n x l`  (其中 `n` 是数据集的大小，`l` 是由 `max_size` 参数设置的最大的最小尺寸)大小的二进制(实际为 int8)值矩阵，用数据集中的列名做为一个键名添加到 HDF5 中。集合映射到整数的方式包括首先使用分词器将字符串映射到集合项的序列(默认按空格来分割完成)。然后收集数据集列中所有不同设置项字符串的字典，然后按频率对它们进行排序，然后将它们按频率由最频繁到最罕见的顺序递增的分配一个整数 ID  (0 分配给 `<PAD>` 用于填充，1 分配给 `<UNK>` 项)。列名将被添加到 JSON 文件中，并带有一个关联字典，其中包含：

  1. integer 到 string 映射 (`idx2str`)
  2. string 到 id 的映射 (`str2idx`)
  3. string 到 frequency 的映射 (`str2freq`)
  4. 所有集合的最大大小 (`max_set_size`)
  5. 额外的预处理信息 (额缺省情况下如何填充缺少的值以及使用什么标记来填充缺少的值)

`Bag` 特征以同样的方式处理设置特征，唯一的区别是矩阵有浮点值(频率)。

`Sequence` 特征被转换成大小为 `n x l` 的整数值矩阵(其中 `n` 是数据集的大小，`l` 是由 `sequence_length_limit` 参数设置的最长序列的最小长度) ，并通过数据集中的列名做为键名添加到 HDF5 中。集合映射到整数的方式包括首先使用分词器将字符串映射到词序列(默认按空格来分割完成)。然后收集数据集列中出现的所有不同词的字典，然后按频率对它们进行排序，然后将它们按频率由最频繁到最罕见的顺序递增的分配一个整数 ID (0 分配给 `<PAD>` 用于填充，1 分配给 `<UNK>` 项)。列名将被添加到 JSON 文件中，并带有一个关联字典，其中包含：

  1. integer 到 string 映射 (idx2str)
  2. string 到 id 的映射 (str2idx)
  3. string 到 frequency 的映射 (str2freq)
  4. 所有序列的最大长度 (sequence_length_limit)
  5. 额外的预处理信息 (额缺省情况下如何填充缺少的值以及使用什么标记来填充缺少的值)

`Text` 特征与序列特征的处理方式相同，但有一些差异。有两种不同的分词方法，一种是在每个字符处分开，另一种是使用基于 spaCy 的分词器(并删除 stopwords) ，HDF5文件中添加了两个不同的键，一个用于字符矩阵，另一个用于单词矩阵。在 JSON 文件中也发生了同样的事情，其中有字典用于将字符映射到整数(及其逆)以及单词映射到整数(及其逆)。在配置中，您可以指定使用哪个级别的表示：字符级别还是单词级别。

`Timeseries` 特征与序列特征的处理方式相同，唯一的区别是 HDF5文件中的矩阵没有整数值，而是浮点值。此外，JSON 文件中不需要任何映射。

`Image` 特征被转换成一个大小为 `n x h x w x c` (其中 `n` 是数据集的大小，`h x w` 是可以设置的图像的特定大小，`c` 是颜色通道的数量)的 int8 值张量 ，并通过数据集中的列名做为键名添加到 HDF5 中。将列名添加到 JSON 文件中，并带有一个关联的字典，其中包含有关调整大小的预处理信息。

### 数据集格式<a id='数据集格式'></a>
Ludwig 使用 Pandas 读取 UTF-8 编码的数据集文件，该文件支持 CSV、 Excel、 Feather、 fwf、 HDF5、 HTML (包含 `<table>`)、 JSON、 JSONL、 Parquet、 pickle (pickle Pandas DataFrame)、 SAS、 SPSS、 Stata 和 TSV 格式。Ludwig 试图通过后缀名自动的识别格式。

如果提供了 * SV*<small>(应该是 CSV——译者注)</small>* 文件，Ludwig 将尝试从数据中确定分隔符(通常为 `,`)。默认转义字符为 `\`。例如，如果 `,` 是列分隔符，并且您的一个数据列中有一个 `,` 则 Pandas 将无法正确加载数据。为了处理这种情况，我们希望用反斜杠转义列中的值(用 `\\` 替换数据中的 `,`)。

## 数据后处理<a id='数据后处理'></a>
从预处理中获得的 JSON 文件也用于后处理：Ludwig模型返回“输出预测”，并根据其数据类型将其映射回原始空间。`numerical ` 和 `timeseries` 按原样返回，而 `category`、`set`、`sequence` 和 `text` 特征输出整数，这些整数使用 JSON 文件中的 `idx2str` 映射回原始标记/名称。当您运行 `experiment ` 或 `predict`，您会发现其中一个 CSV 文件包含了其对应的每个“输出预测”。另外一个概率 CSV 文件包含了对应的预测概率。还有一个包含可能可供选择的概率 CSV 文件(例如，一个特征的所有可能分类的概率)。您还将找到未对应的 NPY 文件。如果不需要它们，可以使用 `--skip_save_unprocessed_output` 参数。

## 配置<a id='配置'></a>
配置是 Ludwig 的核心。它是一个包含建立和训练 Ludwig 模型所需要的所有信息的字典。它混合了易用性、灵活性，以及合理的默认设置和模型参数来精确控制。 配置参数可以以字符串(`--config`)或文件(`--config_file`)提供给 `experiment ` 和 `train` 命令。文件的字符串或内容将由 PyYAML 解析到内存中的字典中，因此解析器接受的任何样式的 YAML 都被认为是有效的，所以多行格式和单行格式都被接受。例如，一个字典列表可以这样写：

```yaml
mylist: [{name: item1, score: 2}, {name: item2, score: 1}, {name: item3, score: 4}]
```

或：

```yaml
mylist:
    -
        name: item1
        score: 2
    -
        name: item2
        score: 1
    -
        name: item3
        score: 4
```

配置文件的结构是一个有五个键的字典：

```json
input_features: []
combiner: {}
output_features: []
training: {}
preprocessing: {}
```

只有 `input_features` 和 `output_features` 是必需的，其他三个字段具有默认值，但您可以自由修改它们。

### 输入特征<a id='输入特征'></a>
`input_features` 列表包含一个字典，每个字典都包含两个必填字段 `name` 和 `type`。`name` 是特征名称，与输入文件中的数据集列名相同，`type` 是受支持的数据类型之一。不同的输入特征有不同的编码方式和参数来决定它的编码器。

您在输入特征中指定的所有其他参数都将作为构建编码器的函数参数，每个编码器可以有不同的参数。

例如，`sequence` 特征可以由 `stacked_cnn` 或者和 `rnn` 进行编码，但只有 `stacked_cnn` 接受 `num_filters` 参数，而只有 `rnn` 接受 `bidirectional` 参数。

在特定数据类型章节中将提供所有可用编码器的列表以及所有参数的描述。有些数据类型只有一种类型的编码器，因此不需要指定它。

编码器的作用是将输入映射到张量中，在数据类型没有时间/顺序方面的情况下通常是向量，在输入数据有时间/顺序方面的情况下是矩阵，或者在输入数据有空间或时空方面的情况下是高阶张量。

同一编码器的不同配置可以返回不同阶的张量，例如，一个序列编码器可以返回大小为 `h` 的向量，它要么是序列的最终向量，要么是序列长度的池化结果，也可以返回大小为 `l × h` 的矩阵，其中 `l` 是序列的长度，如果指定池化操作(`reduce_output`)为 `null`，则 h 是隐藏维数。为了简单起见，在大多数情况下，可以将输出想象为一个向量，但是可以指定一个 `reduce_output` 参数来更改默认行为。

Ludwig 提供的另一个功能是在不同编码器之间绑定权重的选项。例如，如果我的模型将两个句子作为输入，并返回其蕴含的概率，那么我可能希望用相同的编码器对两个句子进行编码。实现方法是在第二个特征上指定 `tied-weights` 参数，该参数是定义的第一个特征的名称。

```yaml
input_features:
    -
        name: sentence1
        type: text
    -
        name: sentence2
        type: text
        tied_weights: sentence1
```

如果指定一个没有定义的输入特征的名称，则会导致错误。此外，为了能够有捆绑的权重，两个输入特征的所有编码器参数必须相同。


### 合成器<a id='合成器'></a>
合成器是模型的一部分，它将不同输入特征的所有输出合成一个单一的表示，然后传递给输出。您可以在配置 `combiner` 部分指定使用哪一个。不同的合成器实现不同的合成逻辑，但默认的一个合成器只是连接输入特征编码器的所有输出，并可选择的通过完全连接层传递连接，最后一层的输出被转发到输出解码器。

```shell
+-----------+
|Input      |
|Feature 1  +-+
+-----------+ |            +---------+
+-----------+ | +------+   |Fully    |
|...        +--->Concat+--->Connected+->
+-----------+ | +------+   |Layers   |
+-----------+ |            +---------+
|Input      +-+
|Feature N  |
+-----------+
```

为了简单起见，您可以想象在大多数情况下输入和输出都是向量，但是有 `reduce_input` 和 `reduce_output` 参数可以指定去改变默认行为。

### 输出特征<a id='输出特征'></a>
`output_features` 列表具有与 `input_features` 列表相同的结构：它是一个包含 `name` 和 `type` 的字典列表。它们表示您希望模型预测的输出/目标。在大多数机器学习任务中，您只需要预测一个目标变量，但在 Ludwig 中您可以指定任意多的输出，它们将以多任务的方式进行优化，使用损失的加权总和作为总体损失进行优化。

输出功能没有编码器，而是有解码器，但是大多数只有一个解码器，所以您不必指定它。

解码器将合成器的输出作为输入，进一步处理，例如通过完全连接层，最后预测值和计算损失和一些度量(根据数据类型，应用不同的损失和度量)。

解码器有额外的参数，特别是 `loss` 允许您指定一个不同的损失来优化这个特定的解码器，例如，数值特征支持 `mean_squared_error` 和 `mean_absolute_error` 作为损失。有关可用解码器和损失的详细信息以及所有参数的描述将在特定数据类型章节中提供。

为了简单起见，在大多数情况下，您可以将来自合成器的输入想象成一个向量，但是可以指定 `reduce_input` 参数来改变默认行为。

#### 多任务学习
Ludwig 允许多个输出特征被指定，每个输出特征可以被看作是一个正在学习的模型任务去执行，因此 Ludwig 天生支持多任务学习。当多个输出特征被指定时，优化的损失是每个输出特征损失的加权和。默认情况下，每个损失权重为 `1`，但是可以通过在每个输出特征定义的 `loss` 部分中为 `weight` 参数指定一个值来改变它。

例如，给定一个 `category` 特征 `A` 和 `numerical` 特征 `B`，为了优化损失 `loss_total = 1.5 * loss_A + 0.8 + loss_B`，配置的 `output_feature` 部分应该是:

```yaml
output_features:
    -
        name: A
        type: category
        loss:
          weight: 1.5
    -
        name: A
        type: numerical
        loss:
          weight: 0.8
```

#### 输出特征依赖
另外一个特点是 Ludwig 提供 `output_features` 之间的依赖概念。在编写特定特征字典时，可以将输出特征的列表指定为依赖项。在建立模型时，Ludwig 检查是否存在循环依赖。如果您这样做，在这些输出特征的预测到原始输入的解码器之前，Ludwig 将连接所有的最终表示。原因在于，如果不同的输出特征具有因果依赖关系，那么了解对其中一个特征的预测有助于对另一个特征的预测。

例如，如果两个输出特征分别是一个粗粒度分类和一个细粒度分类，并且它们彼此位于一个层次结构中，那么知道对粗粒度的预测限制了对细粒度类别进行预测的可能类别。在这种情况下，可以使用以下配置结构:

```yaml
output_features:
    -
        name: coarse_class
        type: category
        num_fc_layers: 2
        fc_size: 64
    -
        name: fine_class
        type: category
        dependencies:
            - coarse_class
        num_fc_layers: 1
        fc_size: 64
```

假设来自合成器的输入具有隐藏维度 `h` 128，有两个完全连接层，它们在 `coarse_class` 解码器的末端返回一个隐藏大小为 64 的向量(该向量将用于最终层，然后投影到输出 `coarse_class` 空间中)。在 `fine_class` 解码器中，将 `coarse_class` 的64维向量与合成器输出向量拼接，生成隐藏大小为192的向量，通过一个完全连接层，最后将64维输出用于最终层，然后投影到 `fine_class` 的输出类空间。

### 训练<a id='训练'></a>
训练部分的配置允许您指定训练过程的一些参数，例如，周期数或学习速率。

以下是现有的训练参数：

  * `batch_size` (默认值 `128`)：用于训练模型的批次大小。
  * `eval_batch_size` (默认值 `0`)：用于评估模型的批次大小。如果值为 `0`，则使用相同的 `batch_size` 值。如果有足够的内存，这有助于加速评估比训练大得多的批次大小，或者将 `sampled_softmax_cross_entropy` 为词汇量大的序列和分类特征用做损失时减小批次大小(需要对整个词汇表进行评估，因此可能需要小得多的批次大小才能适应内存中的激活张量)。
  * `epochs` (默认值 `100`)：训练过程将会经历多少个周期。
  * `early_stop` (默认值  `5`)：如果在一个验证集上的度量没有提升，仍将继续训练的周期数。
  * `optimizer` (默认值 `{type: adam, beta1: 0.9, beta2: 0.999, epsilon: 1e-08}`)：哪个优化器与相关参数一起使用。可用的优化器有 `sgd` (或 `stochastic_gradient_descent`, `gd`, `gradient_descent`, 它们全都相同), `adam`, `adadelta`, `adagrad`, `adamax`, `ftrl`, `nadam`, `rmsprop`。要了解它们的参数，请查看 [TensorFlow的优化器文档](https://www.tensorflow.org/api_docs/python/tf/train)。
  * `learning_rate` (默认值 `0.001`)：要使用的学习速率。
  * `decay` (默认值 `false`)：是否采用指数衰减学习速率。
  * `decay_rate` (默认值 `0.96`)：指数学习速率的衰减速率。
  * `decay_steps` (默认值 `10000`)：指数学习速率衰减的步数。
  * `staircase` (默认值 `false`)：以离散的时间间隔衰减学习速率。
  * `regularization_lambda` (默认值 `0`)：lambda 参数用于将 l2 正则化损失添加到总体损失中。
  * `reduce_learning_rate_on_plateau` (默认值 `0`)：如果有验证集，则达到验证度量的稳定水平时，降低学习速率的次数。
  * `reduce_learning_rate_on_plateau_patience` (默认值 `5`)：如果有一个验证集，那么在降低学习速率之前，在验证度量没有提升的情况下，仍将训练的周期数。
  * `reduce_learning_rate_on_plateau_rate` (默认值 `0.5`)：如果有一个验证集，学习速率的降低速率。
  * `increase_batch_size_on_plateau` (默认值 `0`)：如果有一个验证集，当达到一个验证度量稳定水平时，增加批次大小的次数。
  * `increase_batch_size_on_plateau_patience` (默认值 `5`)：如果有一个验证集，在提高学习速率之前，在验证度量没有提升的情况下，仍将训练的周期数。
  * `increase_batch_size_on_plateau_rate` (默认值 `2`)：如果有一个验证集，批次大小的增加速率。
  * `increase_batch_size_on_plateau_max` (默认值 `512`)：如果有一个验证集，批次大小的最大值。
  * `validation_field` (默认值 `combined`)：当有多个输出特征时，如果验证有提升，则使用哪一个进行计算。可以使用 `validation_measure` 参数设置用于确定是否有提升的度量值。不同的数据类型有不同的可用度量，有关更多详细信息，请参阅特定数据类型章节。组合表示使用所有特征的组合。例如，组合和损失作为度量的组合使用所有输出特征的组合损失的减少量来检查验证是否有所提升*<small>(存疑-译者注)</small>*，而组合和准确性考虑所有输出特征的预测正确的数据点数量(但是考虑到对于某些特征，例如，数值没有精度度量，因此只有在所有输出特征都有精度度量时才应使用精度)*<small>(存疑-译者注)</small>*。
  * `validation_metric` (默认值 `loss`)：用于确定是否有度量的提升。 对于 `validation_field` 中指定的输出特征，应考虑该度量。 不同的数据类型具有不同的可用度量，有关更多详细信息，请参阅特定数据类型章节。
  * `bucketing_field` (默认值 `null`)：当不为 `null` 时，在创建批次时，不是随机洗牌，而是使用指定输入特征的矩阵最后一个维度的长度对数据点进行 bucketing，然后对来自同一容器的随机洗牌的数据点进行采样。被修剪填充到批处理中最长的数据点。指定的特征应该是 `sequence` 或 `text`，编码它的编码器必须是 `rnn`。使用 bucketing 时，`rnn` 编码的速度提高了 1.5 倍，这取决于输入的长度分布。
  * `learning_rate_warmup_epochs` (默认值 `1`)：它是使用学习速率预热的训练周期或次数。它是按照 [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) 描述计算的。论文中作者建议 6 个周期的预热，该参数适用于大数据集和大批次的预热。

#### 优化器细节
可用的优化器包装了 TensorFlow 中可用的优化器。参数的详细信息请参考 [TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)。
  
优化器将使用的 `learning_rate` 参数来自训练部分。其他优化器特定的参数，显示了他们的默认设置，如下：

  * `sgd` (或 `stochastic_gradient_descent`, `gd`, `gradient_descent`)

    ```yaml
    'momentum': 0.0,
    'nesterov': false
    ```
    
  * `adam`

    ```yaml
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-08
    ```
  
  * `adadelta`

    ```yaml
    'rho': 0.95,
    'epsilon': 1e-08
    ```
  
  * `adagrad`

    ```yaml
    'initial_accumulator_value': 0.1,
    'epsilon': 1e-07
    ```
  
  * `adamax`

    ```yaml
    'beta_1': 0.9, 
    'beta_2': 0.999, 
    'epsilon': 1e-07
    ```
  
  * `ftrl`

    ```yaml
    'learning_rate_power': -0.5, 
    'initial_accumulator_value': 0.1,
    'l1_regularization_strength': 0.0, 
    'l2_regularization_strength': 0.0,
    ```
  
  * `nadam`

    ```yaml
    'beta_1': 0.9, 
    'beta_2': 0.999, 
    'epsilon': 1e-07
    ```
  
  * `rmsprop`

    ```yaml
    'decay': 0.9,
    'momentum': 0.0,
    'epsilon': 1e-10,
    'centered': false
    ```
    
### 预处理<a id='预处理'></a>
配置中的 `preprocessing` 部分使指定数据类型特定的参数来执行数据预处理成为可能。预处理字典包含每种数据类型的一个键，但是您必须只指定应用于您的案例的键，其他键将保留为默认值。此外，预处理字典包含与如何拆分数据相关的参数，这些参数不是特定于特征的。

  * `force_split` (默认值 `false`)：如果为 `true`，则忽略数据集文件中的 `split` 列，并且数据集被随机分割。如果为 `false`，则使用 `split` 列。
  * `split_probabilities` (默认值 `[0.7, 0.1, 0.2]`)：数据集数据分别在训练、验证和测试中的占比。这三个值加起来必须为一。
  * `stratify` (默认值 `null`)：如果为 `null`，拆分是随机的，否则您可以指定一个 `category` 特征的名称，以在此特征上进行拆分。

预处理字典示例(显示默认值)：

```yaml
preprocessing:
    force_split: false
    split_probabilities: [0.7, 0.1, 0.2]
    stratify: null
    category: {...}
    sequence: {...}
    text: {...}
    ...
```

关于每个数据类型接受的预处理参数的详细信息将在特定数据类型章节中提供。

必须指出，具有相同数据类型的不同特征可能需要不同的预处理。例如，一个文档分类模型可能有两个文本输入特征，一个用于文档的标题，另一个用于正文。

由于标题的长度比正文的长度短得多，因此应该为标题设置参数 `word_length_limit` 为 10 ，为正文设置参数 `2000`，但它们都使用值为 10000 的 `most_common_words` 参数。

方法是在 `input_feature` 字典中为 `title` 和 `body` 都添加一个包含所需参数和值的 `preprocessing` 键。配置如下所示：

```yaml
preprocessing:
    text:
        most_common_word: 10000
input_features:
    -
        name: title
        type: text
        preprocessing:
            word_length_limit: 20
    -
        name: body
        type: text
        preprocessing:
            word_length_limit: 2000
```

#### 分词器<a id='分词器'></a>
一些不同特征通过标记(例如 `sequence`、`text` 和 `set`)来执行原始数据预处理。下面是您可以为这些特征指定的标记选项:

  * `characters`：将输入字符串中的每个字符用一个单独的标记分割。
  * `space`：使用空格的正则表达式 `\s+` 进行分割。
  * `space_punct`：使用空格和标点的正则表达式 `\w+|[^\w\s]` 进行分割。
  * `underscore`：使用下划线 `_` 进行分割。
  * `comma`：使用逗号 `,` 进行分割。
  * `untokenized`：将整个字符串视为单个标记。
  * `stripped`：在删除字符串开头和结尾的空格后，将整个字符串视为单个标记。
  * `hf_tokenizer`：使用 "Hugging Face AutoTokenizer"，它使用 `pretrained_model_name_or_path` 参数来决定加载哪个分词器。
  * 特定语言的分词器：基于 `spaCy` 的语言分词器。

基于 `spaCy` 的分词器是使用强大的标记化和 NLP 预处理模型提供的函数库。有几种语言可用: 英语(代码 `en`) ，意大利语(代码 `it`) ，西班牙语(代码 `es`) ，德语(代码 `de`) ，法语(代码 `fr`) ，葡萄牙语(代码 `pt`) ，荷兰语(代码 `nl`) ，希腊语(代码 `el`) ，中文(代码 `zh`) ，丹麦语(代码 `da`) ，荷兰语(代码 `el`) ，日语(代码 `ja`) ，立陶宛语(代码 `lt`) ，挪威语(代码 `nb`) ，波兰语(代码 `pl`) ，罗马尼亚语(代码 `ro`)和 Multi (代码 `xx`) ，如果您有一个包含不同语言的数据集，这些语言将非常有用。每种语言都有不同的功能：

  * `tokenize`：使用 spaCy 分词器。
  * `tokenize_filter`：使用 spaCy 分词器，并过滤掉标点、数字、停用词(stopwords)、和长度小于3个字符的词。
  * `tokenize_remove_stopwords`：使用 spaCy 分词器，并过滤掉停用词。
  * `lemmatize`：使用 spaCy 的词形还原工具。
  * `lemmatize_filter`：使用 spaCy 的词形还原工具，并过滤掉标点、数字、停用词、和长度小于3个字符的词。
  * `lemmatize_remove_stopwords`：使用 spaCy 的词形还原工具，并过滤掉停用词。

为了使用这些选项，您必须下载 spaCy 模型：

```shell
python -m spacy download <language_code>
```

并提供 `<language>_<function>` 作为 `tokenizer`，如: `english_tokenizer`，`italian_lemmatize_filter`，`multi_tokenize_filter` 等。关于这些模型的更多细节可以在 [spaCy 文档](https://spacy.io/models) 中找到。

### Binary 特征<a id='Binary_特征'></a>
二进制特征直接转换为长度为 `n` 的二进制值向量(其中 `n` 是数据集的大小) ，并通过数据集中的列名做为键名添加到 HDF5 中。JSON 元数据文件中没有关于它们的额外信息。

可用于预处理的参数如下

  * `missing_value_strategy` (默认值 `fill_with_const`)：当二进制列中缺少一个值时，应该遵循什么策略。该值应该是 `fill_with_const` (用 `fill_value` 参数指定一个特定的值替换缺失的值)，`fill_with_mode` (用列中最常见的值替换缺失的值), `fill_with_mean` (用列中的均值替换缺失的值)，`backfill` (用下一个有效值替换缺失的值)。
  * `fill_value` (默认值 `0`)：在 `missing_value_strategy` 的参数值为 `fill_with_const` 情况下指定的特定值。

#### 二进制输入特征和编码器
二进制特征有两个编码器。一个编码器(`passthrough`)接受来自输入占位符的原始二进制值作为输出返回。输入的大小为`b`，输出的大小为 `b x 1`，其中 `b` 是批次大小。另一个编码器(`dense`)通过一个完全连接层传递原始二进制值。在这种情况下，大小为 `b` 的输入被转换为大小为 `b x h` 的输入。

输入特征列表中的二进制特征示例：

```yaml
name: binary_column_name
type: binary
encoder: passthrough
```

二进制输入特征参数如下

  * `encoder` (默认值 `passthrough`) 对二进制特征进行编码的有效选择：`passthrough`——二进制特征按原样传递，`dense`——二进制特征通过一个完全连接层输入。

`passthrough` 编码器没有额外的参数。


#### Dense 编码器参数
对于 `dense` 编码器，这些是可用的参数。

  * `num_layers` (默认值 `1`)：输入特征经过的完全连接层的堆叠数量。它们的输出被投影到特征的输出空间中。
  * `fc_size` (默认值 `256`)：`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout` (默认值 `0`)：Dropout 比率。

#### 二进制输出特征和解码器
二进制特征能够用于二进制分类或输出为单一概率。只有一个解码器可用于二进制特征，它是一个(可能是空的)堆叠的完全连接层，然后被 sigmoid 函数投影到一个数字。

这些是二进制输出特征的可用参数

  * `reduce_input` (默认值 `sum`)：定义如何在第一维(如果算上批次维数，则是第二维)上减少不是向量而是矩阵或更高阶张量的输入。可用的值有：`sum`，`mean` 或 `avg`，`max`，`concat` (沿第一个维度连接)，`last` (返回第一个维度的最后一个向量)。
  * `dependencies` (默认值 `[]`)：它所依赖的输出特征。有关详细解释，请参阅[输出特征依赖关系](https://ludwig-ai.github.io/ludwig-docs/user_guide/#output-features-dependencies)。
  * `reduce_dependencies` (默认值 `sum`)：定义如何在第一维(如果算上批次维度，则是第二维)上减少一个依赖特征(不是向量，而是一个矩阵或更高阶张量)的输出。可用的值有：`sum`，`mean` 或 `avg`，`max`，`concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)。
  * `loss` (默认值 `{type: cross_entropy, confidence_penalty: 0, robust_lambda: 0, positive_class_weight: 1}`)：是包含损失 `type` 及其超参数的字典。唯一可用的损失 `type` 是 `cross_entropy`（交叉熵），可选参数是 `confidence_penalty`（一个附加项，通过在损失中添加 `a * (max_entropy - entropy) / max_entropy` 项来惩罚过于自信的预测，其中 a 是该参数的值），`robust_lambda`（用 `(1 - robust_lambda) * loss + robust_lambda / 2` 替换损失，这在标签有噪声的情况下很有用）和 `positive_class_weight`（乘以正类[positive class]的损失，增加其重要性）。

这些是二进制输出特征解码器的可用参数

  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，`dropout `，`initializer` 和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。
  * `num_fc_layers` (默认值 `0`)：这是输入特征经过的堆叠的完全连接层的数量。它们的输出被投影到特征的输出空间中。
  * `fc_size` (默认值 `256`)：如果 `fc_layers` 中没有指定 `fc_size`，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，这是默认的 `norm`，将被用于每个层。它表明输出的规范，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `dropout` (默认值 `0`)：Dropout 比率。
  * `use_base` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始值设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `threshold` (默认值 `0.5`)：超过(大于或等于) sigmoid 的预测输出将被映射为1的阈值。

输出特征列表中的二进制特征(具有默认参数)示例：

```yaml
name: binary_column_name
type: binary
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    positive_class_weight: 1
fc_layers: null
num_fc_layers: 0
fc_size: 256
activation: relu
norm: null
dropout: 0.2
weisghts_intializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: l1
bias_regularizer: l1
threshold: 0.5
```

#### 二进制特征度量
每个周期计算出来的唯一可用于二进制特征的度量是 `accuracy` 和 `loss` 本身。如果您将 `validation_field` 设置为二进制特征的名称，那么您可以在配置 `training` 部分将它们中的任何一个设置为 `validation_measure`。

### Numerical 特征<a id='Numerical_特征'></a>
数值特征直接转换为长度为 `n` 的浮点值向量(其中 `n` 是数据集的大小) ，并通过数据集中的列名做为键名添加到 HDF5 中。JSON 元数据文件中没有关于它们的额外信息。

可用于预处理的参数如下

  * `missing_value_strategy` (默认值 `fill_with_const`)：当数值列中缺少一个值时，应该遵循什么策略。该值应该是 `fill_with_const` (用 `fill_value` 参数指定一个特定的值替换缺失的值)，`fill_with_mode` (用列中最常见的值替换缺失的值), `fill_with_mean` (用列中的均值替换缺失的值)，`backfill` (用下一个有效值替换缺失的值)。
  * `fill_value` (默认值 `0`)：在 `missing_value_strategy` 的参数值为 `fill_with_const` 情况下指定的特定值。
  * `normalization` (默认值 `null`)：归一化数值特征类型时使用的技术。可用的选项有 `null`、`zscore`、`minmax` 和 `log1p`。如果值为 `null`，则不执行归一化。如果值为 `zscore`，则计算平均值和标准偏差，使值的平均值为 `0`，标准偏差为 `1`。如果值为 `minmax`，则计算最小值和最大值，并从值中减去最小值，结果除以最大值和最小值之差。如果`normalization` 是 `log1p`，则返回的值是 `1` 的自然对数加上原始值。注：`log1p` 仅为正值定义。

#### 数值输入特征和编码器
数值特征有两个编码器。一个编码器(`passthrough`)接受来自输入占位符的原始二进制值作为输出返回。输入的大小为`b`，输出的大小为 `b x 1`，其中 `b` 是批次大小。另一个编码器(`dense`)通过一个完全连接层传递原始二进制值。在这种情况下，大小为 `b` 的输入被转换为大小为 `b x h` 的输入。

可用的编码器参数如下：

  * `norm` (默认值 `null`)：在单个神经元之后应用 `norm`。可以为空、批次或层。
  * `tied_weights` (默认值 `null`)：绑定编码器权重的输入特征的名称。它必须具有相同类型和相同编码器参数的特征名称。

`passthrough` 编码器没有额外的参数。

#### Dense 编码器参数
对于 `dense` 编码器，这些是可用的参数。

  * `num_layers` (默认值 `1`)：输入特征经过的完全连接层的堆叠数量。它们的输出被投影到特征的输出空间中。
  * `fc_size` (默认值 `256`)：`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的规范，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout` (默认值 `0`)：Dropout 比率。

输入特征列表中的数值特征示例：

```yaml
name: numerical_column_name
type: numerical
norm: null
tied_weights: null
encoder: dense
num_layers: 1
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activation: relu
dropout: 0
```

#### 数值输出特征与解码器
当需要进行回归时，可以使用数值特征。只有一个解码器可用于数值特征，它是一个(可能是空的)堆叠的完全连接层，然后被投影到一个数字。

这些是数值输出特征的可用参数

  * `reduce_input` (默认值 `sum`)：定义如何在第一维(如果算上批次维数，则是第二维)上减少不是向量而是矩阵或更高阶张量的输入。可用的值有：`sum`，`mean` 或 `avg`，`max`，`concat` (沿第一个维度连接)，`last` (返回第一个维度的最后一个向量)。
  * `dependencies` (默认值 `[]`)：它所依赖的输出特征。有关详细解释，请参阅[输出特征依赖关系](https://ludwig-ai.github.io/ludwig-docs/user_guide/#output-features-dependencies)。
  * `reduce_dependencies` (默认值 `sum`)：定义如何在第一维(如果算上批次维度，则是第二维)上减少一个依赖特征(不是向量，而是一个矩阵或更高阶张量)的输出。可用的值有：`sum`，`mean` 或 `avg`，`max`，`concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)。
  * `loss` (默认值 `{type: mean_squared_error}`)：是一个包含损失 `type` 的字典。可用的损失 `type` 是 `mean_squared_error` 和 `mean_absolute_error`。

这些是数值输出特征解码器的可用参数

  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation，`dropout `，`initializer` 和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。
  * `num_fc_layers` (默认值 `0`)：这是输入特征经过的堆叠的完全连接层的数量。它们的输出被投影到特征的输出空间中。
  * `fc_size` (默认值 `256`)：如果 `fc_layers` 中没有指定 `fc_size`，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，这是默认的 `norm`，将被用于每个层。它表明输出的规范，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `dropout` (默认值 `0`)：Dropout 比率。
  * `use_base` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始值设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `clip` (默认值 `null`)：如果不是 `null`，则指定预测将被剪切到的最小值和最大值。该值可以是一个列表或长度为 `2` 的元组，第一个值表示最小值，第二个值表示最大值。例如，`(-5,5)` 将使所有的预测都在 `[-5,5]` 区间内剪切。

输出特征列表中的数值特征(具有默认参数)示例：

```yaml
name: numerical_column_name
type: numerical
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: mean_squared_error
fc_layers: null
num_fc_layers: 0
fc_size: 256
activation: relu
norm: null
norm_params: null
dropout: 0
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
clip: null
```

#### 数值特征度量
每个周期都计算并可用于数值特征的度量有 `mean_squared_error`、 `mean_absolute_error`、 `r2` 和 `loss` 本身。如果您将 `validation_field` 设置为数值特征的名称，则可以在配置 'training' 部分将它们中的任何一个设置为 `validation_measure`。


### Category 特征<a id='Category_特征'></a>
#### 类别特征预处理
类别特征被转换为大小为 `n` 的整数值向量(其中 `n` 是数据集的大小) ，并通过数据集中的列名做为键名添加到 HDF5 中。将类别映射到整数的方法包括：首先收集数据集列中所有不同类别字符串的字典，然后按照频率对它们进行排序，然后将它们按频率由最频繁到最罕见的顺序递增的分配一个整数 ID (0 分配给一个 `<UNK>` 标记)。列名将被添加到 JSON 文件中，并带有一个关联字典，其中包含：

  1. 从整数到字符串的映射 (`idx2str`)
  2. 从字符串到 id 的映射 (`str2idx `)
  3. 从字符串到频率的映射 (`str2freq `)
  4. 所有标记集合的大小 (`vocab_size`)
  5. 额外的预处理信息(默认情况下如何填充缺失值以及使用什么标记来填充缺失值)

可用于预处理的参数如下

  * `missing_value_strategy` (默认值 `fill_with_const`)：当类别列中缺少一个值时，应该遵循什么策略。该值应该是 `fill_with_const` (用 `fill_value` 参数指定一个特定的值替换缺失的值)，`fill_with_mode` (用列中最常见的值替换缺失的值), `fill_with_mean` (用列中的均值替换缺失的值)，`backfill` (用下一个有效值替换缺失的值)。
  * `fill_value` (默认值 `<UNK>`)：在 `missing_value_strategy` 的参数值为 `fill_with_const` 情况下指定的特定值。
  * `lowercase` (默认值 `false`)：分词器处理前必须小写字符串。
  * `most_common` (默认值 `10000`)：要考虑最常见标记的最大数目，如果数据中有超过这个数量，则那些最不常见的标记将被视为未知。

#### 类别输入特征和编码器
类别特征有三个编码器。`passthrough` 编码器将来自输入占位符的原始整数值传递到大小为 `b x 1` 的输出。另外两个编码器映射到 `dense` 或 `sparse ` 嵌入向量(one-hot 编码)，并返回大小为 `b x h` 的输出，其中 `b` 是批次大小，`h` 是嵌入向量的维数。

输入特征参数。

  * `encoder` (默认值 `dense`)：可能的值是 `passthrough`、`dense` 和 `sparse`。`passthrough` 意味着不修改地传递原始整数值。`dense`  表示嵌入向量被随机初始化，`sparse` 表示它们被初始化为 one-hot 编码。
  * `tied_weights` (默认值 `null`)：绑定编码器权重的输入特征的名称。它必须具有相同类型和相同编码器参数的特征名称。

输入特征列表中的类别特征示例:

```yaml
name: category_column_name
type: category
tied_weights: null
encoder: dense
```

可用的编码器参数：

##### Dense 编码器

  * `embedding_size` (默认值 `256`)：它是最大嵌入向量大小，对于 `dense` 表示，实际大小为 `min(vocabulary_size, embedding_size)`，对于 `sparse` 编码，实际大小为 `vocabulary_size`，其中 `vocabulary_size` 是在训练集中对应列特征中出现的不同字符串数（对于 `<UNK>`，加 `1`）。
  * `embeddings_on_cpu` (默认值 `false`)：默认情况下，如果使用 GPU，嵌入向量存储在 GPU 内存中，因为它允许更快的访问，但在某些情况下，嵌入向量可能非常大，此参数强制将嵌入向量放置在常规内存中，并使用 CPU 来解析它们，由于 CPU 和 GPU 内存之间的数据传输，进程稍微减慢。
  * `pretrained_embeddings` (默认值 `null`)：默认情况下，`dense` 嵌入向量是随机初始化的，但是此参数允许以 [GloVe格式](https://nlp.stanford.edu/projects/glove/) 指定包含嵌入向量的文件的路径。 加载包含嵌入向量的文件时，仅保留词汇表中带有标签的嵌入向量，其余的则被丢弃。 如果词汇表中包含了那些在嵌入向量文件中不能匹配的字符串，则使用所有其他嵌入向量的平均值加上一些随机噪声来初始化它们的嵌入向量，以使它们彼此不同。 仅当 `representation` 为 `dense` 时此参数才有效。
  * `embeddings_trainable` (默认值 `true`)：如果为 `true`，嵌入向量在训练过程中进行训练，如果为 `false`，嵌入向量是固定的。加载预定义的嵌入向量时，可能会很有用，以避免微调它们。 该参数仅在 `representation` 为 `dense` 时才有效，因为 `sparse` 是 one-hot 编码， 不可训练。
  * `dropout` (默认值 `0`)：Dropout 比率。
  * `embedding_initializer` (默认值 `null`)：要使用的初始值设定项。如果为 `null`，则使用每个变量的默认值（在大多数情况下为 `glorot_uniform`）。选项有：`constant`、`identity`、`zeros`、`ones`、`orthogonal`、`normal`、`uniform`、`truncated_normal`、`variance_scaling`、`glorot_normal`、`glorot_uniform`、`xavier_normal`、`xavier_uniform`、`he_normal`、`he_uniform`、`lecun_normal`、`lecun_uniform`。或者，可以使用一个指定的包含 `type` 键(标识初始值设定项的类型)以及其它键和参数的字典，例如 `{type:normal，mean:0，stddev:0}`。要了解每个初始值设定项的参数，请参阅[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `embedding_regularizer` (默认值 `null`)：指定要使用 `l1`、`l2` 或 `l1_l2` 的正则化类型。

##### Sparse 编码器

  * `embedding_size` (默认值 `256`)：它是最大嵌入向量大小，对于 `dense` 表示，实际大小为 `min(vocabulary_size, embedding_size)`，对于 `sparse` 编码，实际大小为 `vocabulary_size`，其中 `vocabulary_size` 是在训练集中对应列特征中出现的不同字符串数（对于 `<UNK>`，加 `1`）。
  * `embeddings_on_cpu` (默认值 `false`)：默认情况下，如果使用 GPU，嵌入向量存储在 GPU 内存中，因为它允许更快的访问，但在某些情况下，嵌入向量可能非常大，此参数强制将嵌入向量放置在常规内存中，并使用 CPU 来解析它们，由于 CPU 和 GPU 内存之间的数据传输，进程稍微减慢。
  * `pretrained_embeddings` (默认值 `null`)：默认情况下，`dense` 嵌入向量是随机初始化的，但是此参数允许以 [GloVe格式](https://nlp.stanford.edu/projects/glove/) 指定包含嵌入向量的文件的路径。 加载包含嵌入向量的文件时，仅保留词汇表中带有标签的嵌入向量，其余的则被丢弃。 如果词汇表中包含了那些在嵌入向量文件中不能匹配的字符串，则使用所有其他嵌入向量的平均值加上一些随机噪声来初始化它们的嵌入向量，以使它们彼此不同。 仅当 `representation` 为 `dense` 时此参数才有效。
  * `embeddings_trainable` (默认值 `true`)：如果为 `true`，嵌入向量在训练过程中进行训练，如果为 `false`，嵌入向量是固定的。加载预定义的嵌入向量时，可能会很有用，以避免微调它们。 该参数仅在 `representation` 为 `dense` 时才有效，因为 `sparse` 是 one-hot 编码， 不可训练。
  * `dropout` (默认值 `0`)：Dropout 比率。
  * `initializer` (默认值 `null`)：要使用的初始值设定项。如果为 `null`，则使用每个变量的默认值（在大多数情况下为 `glorot_uniform`）。选项有：`constant`、`identity`、`zeros`、`ones`、`orthogonal`、`normal`、`uniform`、`truncated_normal`、`variance_scaling`、`glorot_normal`、`glorot_uniform`、`xavier_normal`、`xavier_uniform`、`he_normal`、`he_uniform`、`lecun_normal`、`lecun_uniform`。或者，可以使用一个指定的包含 `type` 键(标识初始值设定项的类型)以及其它键和参数的字典，例如 `{type:normal，mean:0，stddev:0}`。要了解每个初始值设定项的参数，请参阅[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `regularize` (默认值 `true`)：如果 `true`，则将嵌入向量权重添加到通过正则化损失正则化的权重集（如果 `training` 中的 `regularization_lambda`大于 `0`）。
  * `tied_weights` (默认值 `null`)：绑定编码器权重的输入特征的名称。它必须具有相同类型和相同编码器参数的特征名称。


输入特征列表中的类别特征示例：

```yaml
name: category_column_name
type: category
encoder: sparse
tied_weights: null
embedding_size: 256
embeddings_on_cpu: false
pretrained_embeddings: null
embeddings_trainable: true
dropout: 0
initializer: null
regularizer: null
```

#### 类别输出特征和解码器<a id='类别输出特征和解码器'></a>
当需要执行多类别分类时，可以使用类别特征。只有一个解码器可用于类别特征，它是一个(可能是空的)堆叠的完全连接层，然后是一个投影到可用类数大小的向量中，最后是一个 softmax。

```
+--------------+   +---------+   +-----------+
|Combiner      |   |Fully    |   |Projection |   +-------+
|Output        +--->Connected+--->into Output+--->Softmax|
|Representation|   |Layers   |   |Space      |   +-------+
+--------------+   +---------+   +-----------+
```

这些是类别输出特征的可用参数：

  * `reduce_input` (默认值 `sum`)：定义如何在第一维(如果算上批次维数，则是第二维)上减少不是向量而是矩阵或更高阶张量的输入。可用的值有：`sum`，`mean` 或 `avg`，`max`，`concat` (沿第一个维度连接)，`last` (返回第一个维度的最后一个向量)。
  * `dependencies` (默认值 `[]`)：它所依赖的输出特征。有关详细解释，请参阅[输出特征依赖关系](https://ludwig-ai.github.io/ludwig-docs/user_guide/#output-features-dependencies)。
  * `reduce_dependencies` (默认值 `sum`)：定义如何在第一维(如果算上批次维度，则是第二维)上减少一个依赖特征(不是向量，而是一个矩阵或更高阶张量)的输出。可用的值有：`sum`，`mean` 或 `avg`，`max`，`concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)。
  * `loss` (默认值 `{type: softmax_cross_entropy, class_similarities_temperature: 0, class_weights: 1, confidence_penalty: 0, distortion: 1, labels_smoothing: 0, negative_samples: 0, robust_lambda: 0, sampler: null, unique: false}`)：包含损失类型的字典。可用的损失类型为 `softmax_cross_entropy` 和 `sampled_softmax_cross_entropy`。
  * `top_k` (默认值 `3`)：确定参数 `k`，即计算 `top_k` 度量时要考虑的类别数。它计算准确度，但考虑真实类别是否出现在按解码器置信度排序的前 `k` 个预测类别中。

这些是损失参数：

  * `confidence_penalty` (默认值 `0`)：为了惩罚过于自信的预测(低熵)，可以在损失中添加 `a * (max_entropy - entropy) / max_entropy` 项，其中 `a` 是该参数的值。适用于有噪音的标签。
  * `robust_lambda` (默认值 `0`)：用 `(1 - robust_lambda) * loss + robust_lambda / c` 替换损失，其中 `c` 是类的数量，这在有噪声标签的情况下很有用。
  * `class_weights` (默认值 `1`)：该值可以是权重向量，每个类一个权重，乘以该类作为正确标注的数据点的损失。在类分布不平衡的情况下，这是一种替代过采样的方法。向量的顺序遵循 JSON 元数据文件中的类别到整数 ID 的映射(`<UNK>` 类也需要包括在内)。或者，该值可以是一个字典，类字符串作为键值，权重作为值，如 `{class_a: 0.5, class_b: 0.7，…}`。
  * `class_similarities` (默认值 `null`)：如果不是 `null`，它就是一个 `c x c` 矩阵，它是一个包含彼此相似类的列表形态。如果 `class_similarities_temperature` 大于 `0`，则使用它。向量的顺序遵循 JSON 元数据文件中的类别到整数 ID 的映射( `<UNK>` 类也需要包括在内)。
  * `class_similarities_temperature` (默认值 `0`)：是 softmax 的热度参数，在每一行 `class_similarity` 上执行。该softmax 的输出用于确定要提供的监督向量，而不是为每个数据点提供一个 one-hot 向量。它背后的直觉是相似类之间的错误比真正不同类之间的错误更容易容忍。
  * `labels_smoothing` (默认值 `0`)：如果非零，将标签平滑为 `1/num_classes`: `new_onehot_labels = onehot_labels * (1 - label_smoothing) + label_smoothing / num_classes`。
  * `negative_samples` (默认值 `0`)：如果 `type` 为 `sampled_softmax_cross_entropy`，该参数表示要使用多少个负样本。
  * `sampler` (默认值 `null`)：选项是 `fixed_unigram`、`uniform`、`log_uniform`、`learned_unigram`。有关采样器的详细描述，请参阅[TensorFlow文档](https://www.tensorflow.org/api_guides/python/nn#Candidate_Sampling)。
  * `distortion` (默认值 `1`)：当 `loss` 是 `sampled_softmax_cross_entropy`，采样器是 `unigram` 或 `learned_unigram `，这是用来扭曲 `unigram` 概率分布。每个权重首先被提高到失真的幂次方，然后再加入内部的 `unigram` 分布。因此，distortion = 1.0 给出了规则的 `unigram` 采样(由 vocab 文件定义)，而 distortion = 0.0 给出了均匀分布。
  * `unique` (默认值 `false`)：确定批次中所有抽样的类是否唯一。

这些是类别输出特征解码器的可用参数：

  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是:`fc_size`， `norm`， `activation`， `dropout`， `weights_initializer`和`weighs_regularizer`。如果字典中没有这些值，则使用默认值。
  * `num_fc_layers` (默认值 `0`)：这是输入特征经过的堆叠的完全连接层的数量。它们的输出被投影到特征的输出空间中。
  * `fc_size` (默认值 `256`)：如果 `fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `dropout` (默认值 `false`)：确定每个层之后是否应该有一个 dropout 层。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于全连接权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。

输出特征列表中的类别特征(具有默认参数)示例：

```yaml
name: category_column_name
type: category
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: softmax_cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    class_weights: 1
    class_similarities: null
    class_similarities_temperature: 0
    labels_smoothing: 0
    negative_samples: 0
    sampler: null
    distortion: 1
    unique: false
fc_layers: null
num_fc_layers: 0
fc_size: 256
activation: relu
norm: null
norm_params: null
dropout: 0
use_biase: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
top_k: 3
```

#### 类别特征度量
每个周期都计算并可用于类别特征的度量有 `accuracy`、`top_k` (如果真实类别出现在解码器置信度排序的前 `k` 个预测类别中，则将计算精确度视为匹配)和 `loss` 本身。如果将 `validation_field` 设置为类别特征的名称，则可以在配置的 `training` 部分中将其中一个设置为 `validation_measure`。

### Set 特征<a id='Set_特征'></a>
#### 集合特征预处理
集合特征应提供为一串由空格分隔的元素，例如“elem5 elem9 elem6”。字符串值被转换成一个大小为 `n x l`（其中 `n` 是数据集的大小，`i` 是由 `max_size` 参数设置的最大的最小尺寸）的二进制（实际上是 int8）值矩阵，并通过数据集中的列名做为键名添加到 HDF5 中。将集合映射为整数的方法包括首先使用分词器将字符串映射到集合项序列（默认情况下，这是通过在空格上拆分来完成的）。然后收集数据集列中存在的所有不同集合项字符串的字典，然后按照频率对它们进行排序，并将它们按频率由最频繁到最罕见的顺序递增的分配一个整数 ID（0 分配给 `<PAD>` 用于填充，1 分配给 `<UNK>` 项）。列名被添加到 JSON 文件中，相关联的字典包含：

  1. integer 到 string 映射 (`idx2str`)
  2. string 到 id 的映射 (`str2idx`)
  3. string 到 frequency 的映射 (`str2freq`)
  4. 所有集合的最大尺寸 (`max_set_size`)
  5. 额外的预处理信息 (额缺省情况下如何填充缺少的值以及使用什么标记来填充缺少的值)

可用于预处理的参数如下：

  * `tokenizer` (默认值 `space`)：定义如何从数据集列的原始字符串内容映射到一组元素。默认值 `space` 以空格分隔字符串。常见的选项包括：`underscore`(以下划线分隔)，`comma`(以逗号分隔)，`json`(通过 JSON 解析器将字符串解析为集合或列表)。有关所有可用的选项，请参阅[分词器](#分词器)一节。
  * `missing_value_strategy` (默认值 `fill_with_const`)：当类别列中缺少一个值时，应该遵循什么策略。该值应该是 `fill_with_const` (用 `fill_value` 参数指定一个特定的值替换缺失的值)，`fill_with_mode` (用列中最常见的值替换缺失的值), `fill_with_mean` (用列中的均值替换缺失的值)，`backfill` (用下一个有效值替换缺失的值)。
  * `fill_value` (默认值 `<UNK>`)：在 `missing_value_strategy` 的参数值为 `fill_with_const` 情况下指定的特定值。
  * `lowercase` (默认值 `false`)：分词器处理前必须小写字符串。
  * `most_common` (默认值 `10000`)：要考虑最常见标记的最大数目，如果数据中有超过这个数量，则那些最不常见的标记将被视为未知。

#### 集合输入特征和编码器
集合特征有一个编码器，来自输入占位符的原始二进制值首先在稀疏整数列表中转换，然后它们被映射到密集(dense)或稀疏(sparse)嵌入向量(one-hot 编码)，最后它们被聚合并作为输出返回。输入的尺寸为 `b`，输出的尺寸为 `b x h`，其中 `b` 是批次大小，`h` 是嵌入的维度。

```
+-+
|0|          +-----+
|0|   +-+    |emb 2|   +-----------+
|1|   |2|    +-----+   |Aggregation|
|0+--->4+---->emb 4+--->Reduce     +->
|1|   |5|    +-----+   |Operation  |
|1|   +-+    |emb 5|   +-----------+
|0|          +-----+
+-+
```

可用的编码器参数是：

  * `representation` (默认值 `dense`)：可能的值是 `dense` 和 `sparse`。`dense` 表示嵌入向量被随机初始化，`sparse` 表示它们被初始化为 one-hot 编码。
  * `embedding_size` (默认值 `50`)：它是最大嵌入向量大小，对于 `dense` 表示，实际大小为 `min(vocabulary_size, embedding_size)`，对于 `sparse` 编码，实际大小为 `vocabulary_size`，其中 `vocabulary_size` 是在训练集中对应列特征中出现的不同字符串数（对于 `<UNK>`，加 `1`）。
  * `embeddings_trainable` (默认值 `true`)：如果为 `true`，嵌入向量在训练过程中进行训练，如果为 `false`，嵌入向量是固定的。加载预定义的嵌入向量时，可能会很有用，以避免微调它们。 该参数仅在 `representation` 为 `dense` 时才有效，因为 `sparse` 是 one-hot 编码， 不可训练。
  * `pretrained_embeddings` (默认值 `null`)：默认情况下，`dense` 嵌入向量是随机初始化的，但是此参数允许以 [GloVe格式](https://nlp.stanford.edu/projects/glove/) 指定包含嵌入向量的文件的路径。 加载包含嵌入向量的文件时，仅保留词汇表中带有标签的嵌入向量，其余的则被丢弃。 如果词汇表中包含了那些在嵌入向量文件中不能匹配的字符串，则使用所有其他嵌入向量的平均值加上一些随机噪声来初始化它们的嵌入向量，以使它们彼此不同。 仅当 `representation` 为 `dense` 时此参数才有效。
  * `embeddings_on_cpu` (默认值 `false`)：默认情况下，如果使用 GPU，嵌入向量存储在 GPU 内存中，因为它允许更快的访问，但在某些情况下，嵌入向量可能非常大，此参数强制将嵌入向量放置在常规内存中，并使用 CPU 来解析它们，由于 CPU 和 GPU 内存之间的数据传输，进程稍微减慢。
  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，`dropout `，`initializer` 和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用默认值。
  * `num_fc_layers` (默认值 `1`)：这是输入特征经过的堆叠的完全连接层的数量。它们的输出被投影到特征的输出空间中。
  * `fc_size` (默认值 `10`)：`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout` (默认值 `0`)：Dropout 比率。
  * `reduce_output` (默认值 `sum`)：描述用于聚合集合项的嵌入向量的策略。可能的值有 `sum`、`mean`和 `sqrt`（加权和除以权重平方和的平方根）。
  * `tied_weights` (默认值 `null`)：绑定编码器权重的输入特征的名称。它必须具有相同类型和相同编码器参数的特征名称。

输入特征列表中的集合特征示例：

```yaml
name: set_column_name
type: set
representation: dense
embedding_size: 50
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
fc_layers: null
num_fc_layers: 0
fc_size: 10
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0.0
reduce_output: sum
tied_weights: null
```

#### 集合输出特征和解码器
当需要执行多标签分类时，可以使用集合特征。只有一个解码器可用于集合特征，它是一个(可能是空的)堆叠的完全连接层，然后投影到具有可用分类数大小的向量中，最后是一个 sigmoid 函数。

```
+--------------+   +---------+   +-----------+
|Combiner      |   |Fully    |   |Projection |   +-------+
|Output        +--->Connected+--->into Output+--->Sigmoid|
|Representation|   |Layers   |   |Space      |   +-------+
+--------------+   +---------+   +-----------+
```

这些是集合输出特征的可用参数：

  * `reduce_input` (默认值 `sum`)：定义如何在第一维(如果算上批次维数，则是第二维)上减少不是向量而是矩阵或更高阶张量的输入。可用的值有：`sum`，`mean` 或 `avg`，`max`，`concat` (沿第一个维度连接)，`last` (返回第一个维度的最后一个向量)。
  * `dependencies` (默认值 `[]`)：它所依赖的输出特征。有关详细解释，请参阅[输出特征依赖关系](https://ludwig-ai.github.io/ludwig-docs/user_guide/#output-features-dependencies)。
  * `reduce_dependencies` (默认值 `sum`)：定义如何在第一维(如果算上批次维度，则是第二维)上减少一个依赖特征(不是向量，而是一个矩阵或更高阶张量)的输出。可用的值有：`sum`，`mean` 或 `avg`，`max`，`concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)。
  * `loss` (默认值 `{type: sigmoid_cross_entropy}`)：是一个包含损失 `type` 的字典。可用的损失 `type` 是 `sigmoid_cross_entropy`。

这些是集合输出特征解码器的可用参数：

  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，`dropout `，`initializer` 和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。
  * `num_fc_layers` (默认值 `0`)：这是输入特征经过的堆叠的完全连接层的数量。它们的输出被投影到特征的输出空间中。
  * `fc_size` (默认值 `256`)：`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout` (默认值 `0`)：Dropout 比率。
  * `threshold` (默认值 `0.5`) ：超过(大于或等于) sigmoid 的预测输出将被映射为1的阈值。

输出特征列表中的集合特征(具有默认参数)示例：

```yaml
name: set_column_name
type: set
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: sigmoid_cross_entropy
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0.0
threshold: 0.5
```

#### 集合特征度量
每个周期都计算并可用于集合特征的度量有 `jaccard_index` 和 `loss` 本身。如果将 `validation_field` 设置为集合特征的名称，则可以在配置的 `training` 部分中将其中一个设置为 `validation_measure`。

### Bag 特征<a id='Bag_特征'></a>
#### 袋特征预处理
提供的袋特征应该是一串由空格分隔的元素，例如 “element5 element9 element6”。袋特征跟集合特征的处理方法一样，唯一的区别是矩阵有浮点值(频率)。

袋特征有一个编码器，来自输入占位符的原始浮点值首先在稀疏整数列表中转换，然后映射到密集或稀疏嵌入向量（one-hot 编码），聚合为加权和，其中权重为原始浮点值，最后作为输出返回。输入的大小为 `b`，而输出的大小为 `b x h`，其中 `b` 是批次大小，`h` 是嵌入向量的维度。

参数与用于集合输入特征的参数相同，但 `reduce_output` 除外，该参数不适用于这种情况，因为加权和已经起到了减速器的作用。

#### 袋输出特征和解码器
目前还没有可用的袋解码器。

#### 袋特征度量
因为没有解码器，所以也没有袋特征度量。

### Sequence 特征<a id='Sequence_特征'></a>
#### 序列特征预处理
序列特征被转换成大小为 `n x l` 的整数值矩阵(其中 `n` 是数据集的大小，`l` 是由 `sequence_length_limit` 参数设置的最长序列的最小长度) ，并通过数据集中的列名做为键名添加到 HDF5 中。集合映射到整数的方式包括首先使用分词器将字符串映射到词序列(默认按空格来分割完成)。然后收集数据集列中出现的所有不同词的字典，然后按频率对它们进行排序，然后将它们按频率由最频繁到最罕见的顺序递增的分配一个整数 ID (0 分配给 `<PAD>` 用于填充，1 分配给 `<UNK>` 项)。列名将被添加到 JSON 文件中，并带有一个关联字典，其中包含：

  1. integer 到 string 映射 (idx2str)
  2. string 到 id 的映射 (str2idx)
  3. string 到 frequency 的映射 (str2freq)
  4. 所有序列的最大长度 (sequence_length_limit)
  5. 额外的预处理信息 (额缺省情况下如何填充缺少的值以及使用什么标记来填充缺少的值)

可用于预处理的参数如下：

  * `sequence_length_limit` (默认值 `256`)：序列的最大长度。长于此值的序列将被截断，而较短的序列将被填充。
  * `most_common` (默认值 `20000`)：要考虑最常见标记的最大数目，如果数据中有超过这个数量，则那些最不常见的标记将被视为未知。
  * `padding_symbol` (默认值 `<PAD>`)：用作填充符号的字符串。它被映射到词汇表中的整数 ID 0。
  * `unknown_symbol` (默认值 `<UNK>`)：用作未知符号的字符串。它被映射到词汇表中的整数ID 1。
  * `padding` (默认值 `right`)：填充的方向。`right` 和 `left` 是可用的选项。
  * `tokenizer` (默认值 `space`)：定义如何从数据集列的原始字符串内容映射到元素序列。有关所有可用的选项，请参阅[分词器](#分词器)一节。
  * `lowercase` (默认值 `false`)：分词器处理前必须小写字符串。
  * `vocab_file` (默认值 `null`)：指向一个包含序列词汇表的 UTF-8 编码的文件的路径。在每一行中，第一个字符到 `\t` 或 `\n` 为止的字符串被认为是一个单词。
  * `missing_value_strategy` (默认值 `fill_with_const`)：当序列 列中缺少一个值时，应该遵循什么策略。该值应该是 `fill_with_const` (用 `fill_value` 参数指定一个特定的值替换缺失的值)，`fill_with_mode` (用列中最常见的值替换缺失的值), `fill_with_mean` (用列中的均值替换缺失的值)，`backfill` (用下一个有效值替换缺失的值)。
  * `fill_value` (默认值 `""`)：在 `missing_value_strategy` 的参数值为 `fill_with_const` 情况下指定的特定值。

#### 序列输入特征和编码器<a id='序列输入特征和编码器'></a>
序列特征有几个编码器，每个编码器都有自己的参数。输入的大小为 `b`，而输出的大小为 `b x h`，其中 `b` 是批次大小，`h` 是编码器输出的维数。如果需要序列中每个元素的表示（例如标记它们或使用注意力机制），可以将参数 `reduce_output` 指定为 `null`，输出将是一个 `b x s x h` 张量，其中 `s` 是序列的长度。一些编码器由于其内部工作原理，可能需要指定额外的参数，以便为序列的每个元素获得一个表示。例如，`parallel_cnn` 编码器，默认情况下会合并和展平序列维度，然后通过完全连接层传递展平的向量，因此为了获得完整的张量，必须指定 `reduce_output: null`。

序列输入特征参数是：

  * `encoder` (默认值 `parallel_cnn`)：用于对序列进行编码的编码器的名称。可用的是 `embed`, `parallel_cnn`, `stacked_cnn`, `stacked_parallel_cnn`, `rnn`, `cnnrnn`, `transformer` 和 `passthrough`(等价于指定 `null` 或 `None`)。
  * `tied_weights` (默认值 `null`)：绑定编码器权重的输入特征的名称。它必须具有相同类型和相同编码器参数的特征名称。

##### 嵌入式编码器
嵌入式编码器只是将序列中的每个整数映射到一个嵌入向量，创建一个 `b x s x h` 张量，其中 `b` 是批次大小，`s` 是序列的长度，`h` 是嵌入向量大小。对张量沿 `s` 维进行降维，得到批次中每个元素的大小为 `h` 的单个向量。如果您想输出完整的 `b x s x h` 张量，可以指定 `reduce _ output: null`。

```
       +------+
       |Emb 12|
       +------+
+--+   |Emb 7 |
|12|   +------+
|7 |   |Emb 43|   +-----------+
|43|   +------+   |Aggregation|
|65+--->Emb 65+--->Reduce     +->
|23|   +------+   |Operation  |
|4 |   |Emb 23|   +-----------+
|1 |   +------+
+--+   |Emb 4 |
       +------+
       |Emb 1 |
       +------+
```

这些是可用于嵌入式编码器的参数：

  * `representation` (默认值 `dense`)：可能的值是 `dense` 和 `sparse`。`dense` 表示嵌入向量被随机初始化，`sparse` 表示它们被初始化为 one-hot 编码。
  * `embedding_size` (默认值 `256`)：它是最大嵌入向量大小，对于 `dense` 表示，实际大小为 `min(vocabulary_size, embedding_size)`，对于 `sparse` 编码，实际大小为 `vocabulary_size`，其中 `vocabulary_size` 是在训练集中对应列特征中出现的不同字符串数（对于 `<UNK>`，加 `1`）。
  * `embeddings_trainable` (默认值  `true`)：如果为 `true`，嵌入向量在训练过程中进行训练，如果为 `false`，嵌入向量是固定的。加载预定义的嵌入向量时，可能会很有用，以避免微调它们。 该参数仅在 `representation` 为 `dense` 时才有效，因为 `sparse` 是 one-hot 编码， 不可训练。
  * `pretrained_embeddings` (默认值 `null`)：默认情况下，`dense` 嵌入向量是随机初始化的，但是此参数允许以 [GloVe格式](https://nlp.stanford.edu/projects/glove/) 指定包含嵌入向量的文件的路径。 加载包含嵌入向量的文件时，仅保留词汇表中带有标签的嵌入向量，其余的则被丢弃。 如果词汇表中包含了那些在嵌入向量文件中不能匹配的字符串，则使用所有其他嵌入向量的平均值加上一些随机噪声来初始化它们的嵌入向量，以使它们彼此不同。 仅当 `representation` 为 `dense` 时此参数才有效。
  * `embeddings_on_cpu` (默认值 `false`)：默认情况下，如果使用 GPU，嵌入向量存储在 GPU 内存中，因为它允许更快的访问，但在某些情况下，嵌入向量可能非常大，此参数强制将嵌入向量放置在常规内存中，并使用 CPU 来解析它们，由于 CPU 和 GPU 内存之间的数据传输，进程稍微减慢。
  * `dropout` (默认值 0)：Dropout 比率。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `reduce_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 序列长度维减少输出张量。可用的值有: `sum`， `mean` 或 `avg`， `max`， `concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)和 `null`(不减少并返回整个张量)。

序列特征在输入列表中使用嵌入式编码器的示例：

```yaml
name: sequence_column_name
type: sequence
encoder: embed
tied_weights: null
representation: dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
dropout: 0
weights_initializer: null
weights_regularizer: null
reduce_output: sum
```

##### 并行 CNN 编码器
并行 cnn 编码器的灵感来自 [Yoon Kim's Convolutional Neural Network for Sentence Classification](https://arxiv.org/abs/1408.5882)。 它首先将输入整数序列 `b x s`（其中 `b` 是批次大小，而 `s` 是序列的长度）映射到一个嵌入向量序列中，然后将嵌入向量通过多个具有不同滤波器大小的并行 1 维卷积层（默认情况下，4层具有大小为 2、3、4 和 5 的滤波器），然后是最大池化和连接。 接着，将并行卷积层的输出连接在一起的单个向量通过一个堆叠的完全连接层，并以 `b x h` 张量返回，其中 `h` 是最后一个完全连接层的输出大小。 如果要输出完整的 `b x s x h` 张量，则可以指定 `reduce_output: null`。

```
                   +-------+   +----+
                +-->1D Conv+--->Pool+-+
       +------+ |  |Width 2|   +----+ |
       |Emb 12| |  +-------+          |
       +------+ |                     |
+--+   |Emb 7 | |  +-------+   +----+ |
|12|   +------+ +-->1D Conv+--->Pool+-+
|7 |   |Emb 43| |  |Width 3|   +----+ |           +---------+
|43|   +------+ |  +-------+          | +------+  |Fully    |
|65+--->Emb 65+-+                     +->Concat+-->Connected+->
|23|   +------+ |  +-------+   +----+ | +------+  |Layers   |
|4 |   |Emb 23| +-->1D Conv+--->Pool+-+           +---------+
|1 |   +------+ |  |Width 4|   +----+ |
+--+   |Emb 4 | |  +-------+          |
       +------+ |                     |
       |Emb 1 | |  +-------+   +----+ |
       +------+ +-->1D Conv+--->Pool+-+
                   |Width 5|   +----+
                   +-------+
```

这些可用于并行cnn编码器：

  * `representation` (默认值 `dense`)：可能的值是 `dense` 和 `sparse`。`dense` 表示嵌入向量被随机初始化，`sparse` 表示它们被初始化为 one-hot 编码。
  * `embedding_size` (默认值 `256`)：它是最大嵌入向量大小，对于 `dense` 表示，实际大小为 `min(vocabulary_size, embedding_size)`，对于 `sparse` 编码，实际大小为 `vocabulary_size`，其中 `vocabulary_size` 是在训练集中对应列特征中出现的不同字符串数（对于 `<UNK>`，加 `1`）。
  * `embeddings_trainable` (默认值 `true`)：如果为 `true`，嵌入向量在训练过程中进行训练，如果为 `false`，嵌入向量是固定的。加载预定义的嵌入向量时，可能会很有用，以避免微调它们。 该参数仅在 `representation` 为 `dense` 时才有效，因为 `sparse` 是 one-hot 编码， 不可训练。
  * `pretrained_embeddings` (默认值 `null`)：默认情况下，`dense` 嵌入向量是随机初始化的，但是此参数允许以 [GloVe格式](https://nlp.stanford.edu/projects/glove/) 指定包含嵌入向量的文件的路径。 加载包含嵌入向量的文件时，仅保留词汇表中带有标签的嵌入向量，其余的则被丢弃。 如果词汇表中包含了那些在嵌入向量文件中不能匹配的字符串，则使用所有其他嵌入向量的平均值加上一些随机噪声来初始化它们的嵌入向量，以使它们彼此不同。 仅当 `representation` 为 `dense` 时此参数才有效。
  * `embeddings_on_cpu` (默认值 `false`)：默认情况下，如果使用 GPU，嵌入向量存储在 GPU 内存中，因为它允许更快的访问，但在某些情况下，嵌入向量可能非常大，此参数强制将嵌入向量放置在常规内存中，并使用 CPU 来解析它们，由于 CPU 和 GPU 内存之间的数据传输，进程稍微减慢。
  * `conv_layers` (默认值 `null`)：它是一个字典列表，包含所有卷积层的参数。列表的长度决定并行卷积层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是:`filter_size`, `num_filters`, `pool`, `norm`, `activation` 和 `regularize`。如果字典中缺少这些值中的任何一个，则将使用作为编码器参数指定的默认值。如果 `conv_layers` 和 `num_conv_layers` 都为 `null`，则会给 `conv_layers` 赋一个默认列表，值为 `[{filter_size: 2}, {filter_size: 3}, {filter_size: 4}, {filter_size: 5}]`。
  * `num_conv_layers` (默认值 `null`)：如果 `conv_layers` 是 `null`，这是并行卷积层的数量。
  * `filter_size` (默认值 `3`)：如果在 `conv_layers` 中还没有指定 `filter_size`，这将是每个层使用的默认 `filter_size`。它表示一维卷积滤波器的宽度。
  * `num_filters` (默认值 `256`)：如果 `num_filters` 还没有在 `conv_layers` 中指定，这是默认的将用于每个层的 `num_filters`。它表示滤波器的数量，进而表示一维卷积的输出通道。
  * `pool_function` (默认值 `max`)：池化函数——`max` 为最大值。`average `， `avg` 或 `mean` 都将计算平均值。
  * `pool_size` (默认值 `null`)：如果在 `conv_layers` 中尚未指定 `pool_size`，则将为每个层使用默认的 `pool_size`。它表示卷积操作后沿 `s` 序列维执行的最大池化的大小。
  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，`dropout `，`initializer` 和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。如果 `fc_layers` 和 `num_fc_layers` 都为 `null`，则会给 `fc_layers` 分配一个默认列表，值为 `[{fc_size: 512}, {fc_size: 256}]` (仅适用于 `reduce_output` 不为 `null` 的情况)。
  * `num_fc_layers` (默认值 `null`)：如果 `fc_layers` 是 `null`，这是堆叠的完全连接层数(仅适用于 `reduce_output` 不是 `null`)。
  * `fc_size` (默认值 `256`)：`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout` (默认值 `0`)：Dropout 比率。
  * `reduce_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 序列长度维减少输出张量。可用的值有: `sum`， `mean` 或 `avg`， `max`， `concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)和 `null`(不减少并返回整个张量)。

序列特征在输入列表中使用并行 cnn 编码器的示例：

```yaml
name: sequence_column_name
type: sequence
encoder: parallel_cnn
tied_weights: null
representation: dense
embedding_size: 256
embeddings_on_cpu: false
pretrained_embeddings: null
embeddings_trainable: true
conv_layers: null
num_conv_layers: null
filter_size: 3
num_filters: 256
pool_function: max
pool_size: null
fc_layers: null
num_fc_layers: null
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0.0
reduce_output: sum
```

##### 堆叠式 CNN 编码器
堆叠式 cnn 编码器的灵感来自于 [Xiang Zhang at all's Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)。它首先将输入整数序列 `b x s`（其中 `b` 是批次大小，`s` 是序列长度）映射到一个嵌入向量序列中，然后将嵌入向量通过一个具有不同滤波器大小的堆叠的卷积层（默认情况下为 6 层，滤波器大小为 7、7、3、3、3 和 3），然后是一个可选的最终池化和扁平化操作。接着，这个单一的扁平化向量通过一个堆叠的完全连接层，并作为 `bxh` 张量返回，其中 `h` 是最后一个完全连接层的输出大小。如果要输出完整的 `b x s x h` 张量，可以将所有 `conv_layers` 的 `pool_size` 指定为 `null` 和 `reduce_output: null`，而如果 `pool_size` 的值不同于 `null` 和 `reduce_output: null`，则返回的张量将为 `b x s'x h`，其中 `s'` 是最后一个卷积层的输出宽度。

```
       +------+
       |Emb 12|
       +------+
+--+   |Emb 7 |
|12|   +------+
|7 |   |Emb 43|   +----------------+  +---------+
|43|   +------+   |1D Conv         |  |Fully    |
|65+--->Emb 65+--->Layers          +-->Connected+->
|23|   +------+   |Different Widths|  |Layers   |
|4 |   |Emb 23|   +----------------+  +---------+
|1 |   +------+
+--+   |Emb 4 |
       +------+
       |Emb 1 |
       +------+
```

以下是堆叠式 CNN 编码器 可用的参数：

  * `representation` (默认值 `dense`)：可能的值是 `dense` 和 `sparse`。`dense` 表示嵌入向量被随机初始化，`sparse` 表示它们被初始化为 one-hot 编码。
  * `embedding_size` (默认值 `256`)：它是最大嵌入向量大小，对于 `dense` 表示，实际大小为 `min(vocabulary_size, embedding_size)`，对于 `sparse` 编码，实际大小为 `vocabulary_size`，其中 `vocabulary_size` 是在训练集中对应列特征中出现的不同字符串数（对于 `<UNK>`，加 `1`）。
  * `embeddings_trainable` (默认值 `true`)：如果为 `true`，嵌入向量在训练过程中进行训练，如果为 `false`，嵌入向量是固定的。加载预定义的嵌入向量时，可能会很有用，以避免微调它们。 该参数仅在 `representation` 为 `dense` 时才有效，因为 `sparse` 是 one-hot 编码， 不可训练。
  * `pretrained_embeddings` (默认值 `null`)：默认情况下，`dense` 嵌入向量是随机初始化的，但是此参数允许以 [GloVe格式](https://nlp.stanford.edu/projects/glove/) 指定包含嵌入向量的文件的路径。 加载包含嵌入向量的文件时，仅保留词汇表中带有标签的嵌入向量，其余的则被丢弃。 如果词汇表中包含了那些在嵌入向量文件中不能匹配的字符串，则使用所有其他嵌入向量的平均值加上一些随机噪声来初始化它们的嵌入向量，以使它们彼此不同。 仅当 `representation` 为 `dense` 时此参数才有效。
  * `embeddings_on_cpu` (默认值 `false`)：默认情况下，如果使用 GPU，嵌入向量存储在 GPU 内存中，因为它允许更快的访问，但在某些情况下，嵌入向量可能非常大，此参数强制将嵌入向量放置在常规内存中，并使用 CPU 来解析它们，由于 CPU 和 GPU 内存之间的数据传输，进程稍微减慢。
  * `conv_layers` (默认值 `null`)：它是一个字典列表，包含所有卷积层的参数。列表的长度决定并行卷积层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是:`filter_size`, `num_filters`, `pool`, `norm`, `activation` 和 `regularize`。如果字典中缺少这些值中的任何一个，则将使用作为编码器参数指定的默认值。如果 `conv_layers` 和 `num_conv_layers` 都为 `null`，则会给 `conv_layers` 赋一个默认列表，值为 `[{filter_size: 7, pool_size: 3, regularize: false}, {filter_size: 7, pool_size: 3, regularize: false}, {filter_size: 3, pool_size: null, regularize: false}, {filter_size: 3, pool_size: null, regularize: false}, {filter_size: 3, pool_size: null, regularize: true}, {filter_size: 3, pool_size: 3, regularize: true}]`。
  * `num_conv_layers` (默认值 `null`)：如果 `conv_layers` 是 `null`，这是堆叠卷积层的数量。
  * `filter_size` (默认值 `3`)：如果在 `conv_layers` 中还没有指定 `filter_size`，这将是每个层使用的默认 `filter_size`。它表示一维卷积滤波器的宽度。
  * `num_filters` (默认值 `256`)：如果 `num_filters` 还没有在 `conv_layers` 中指定，这是默认的将用于每个层的 `num_filters`。它表示滤波器的数量，进而表示一维卷积的输出通道。
  * `strides` (默认值 `1`)：卷积的步长。
  * `padding` (默认值 `same`)：`valid` 或 `same`。
  * `dilation_rate` (默认值 `1`)：用于扩张卷积的扩张率。
  * `pool_function` (默认值 `max`)：池化函数——`max` 为最大值。`average `， `avg` 或 `mean` 都将计算平均值。
  * `pool_size` (默认值 `null`)：如果在 `conv_layers` 中尚未指定 `pool_size`，则将为每个层使用默认的 `pool_size`。它表示卷积操作后沿 `s` 序列维执行的最大池化的大小。
  * `pool_strides` (默认值 `null`)：缩小因子。
  * `pool_padding` (默认值 `same`)：`valid` 或 `same`。
  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`， 和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。如果 `fc_layers` 和 `num_fc_layers` 都为 `null`，则会给 `fc_layers` 分配一个默认列表，值为 `[{fc_size: 512}, {fc_size: 256}] (only applies if reduce_output is not null)`。
  * `num_fc_layers` (默认值 `null`)：如果 `fc_layers` 是 `null`，这是堆叠的完全连接层数(仅适用于 `reduce_output` 不是 `null`)。
  * `fc_size` (默认值 `256`)：`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout` (默认值 `0`)：Dropout 比率。
  * `reduce_output` (默认值 `max`)：定义如果张量的秩大于 2，如何沿 `s` 序列长度维减少输出张量。可用的值有: `sum`， `mean` 或 `avg`， `max`， `concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)和 `null`(不减少并返回整个张量)。

序列特征在输入列表中使用堆叠式 CNN 编码器的示例：

```yaml
name: sequence_column_name
type: sequence
encoder: stacked_cnn
tied_weights: null
representation: dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
conv_layers: null
num_conv_layers: null
filter_size: 3
num_filters: 256
strides: 1
padding: same
dilation_rate: 1
pool_function: max
pool_size: null
pool_strides: null
pool_padding: same
fc_layers: null
num_fc_layers: null
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
reduce_output: max
```

##### 堆叠式并行 CNN 编码器
叠层式并行 cnn 编码器是并行 cnn 和堆叠式 cnn 编码器的组合，其中堆叠的每一层由并行卷积层组成。它首先将输入整数序列 `b x s`（其中 `b` 是批次大小，`s` 是序列长度）映射到一个嵌入向量序列中，然后将嵌入向量通过一个堆叠的几个具有不同过滤器大小的平行一维卷积层，接着是可选的最终池化和扁平化操作。最后，这个单一的扁平化向量通过一个堆叠的完全连接层，并作为 `bxh` 张量返回，其中 `h` 是最后一个完全连接层的输出大小。如果要输出完整的 `b x s x h` 张量，可以指定 `reduce_output: null`。

```
                   +-------+                      +-------+
                +-->1D Conv+-+                 +-->1D Conv+-+
       +------+ |  |Width 2| |                 |  |Width 2| |
       |Emb 12| |  +-------+ |                 |  +-------+ |
       +------+ |            |                 |            |
+--+   |Emb 7 | |  +-------+ |                 |  +-------+ |
|12|   +------+ +-->1D Conv+-+                 +-->1D Conv+-+
|7 |   |Emb 43| |  |Width 3| |                 |  |Width 3| |                   +---------+
|43|   +------+ |  +-------+ | +------+  +---+ |  +-------+ | +------+  +----+  |Fully    |
|65+--->Emb 65+-+            +->Concat+-->...+-+            +->Concat+-->Pool+-->Connected+->
|23|   +------+ |  +-------+ | +------+  +---+ |  +-------+ | +------+  +----+  |Layers   |
|4 |   |Emb 23| +-->1D Conv+-+                 +-->1D Conv+-+                   +---------+
|1 |   +------+ |  |Width 4| |                 |  |Width 4| |
+--+   |Emb 4 | |  +-------+ |                 |  +-------+ |
       +------+ |            |                 |            |
       |Emb 1 | |  +-------+ |                 |  +-------+ |
       +------+ +-->1D Conv+-+                 +-->1D Conv+-+
                   |Width 5|                      |Width 5|
                   +-------+                      +-------+
```

以下是堆叠式并行 cnn 编码器的可用参数：

  * `representation` (默认值 `dense`)：可能的值是 `dense` 和 `sparse`。`dense` 表示嵌入向量被随机初始化，`sparse` 表示它们被初始化为 one-hot 编码。
  * `embedding_size` (默认值 `256`)：它是最大嵌入向量大小，对于 `dense` 表示，实际大小为 `min(vocabulary_size, embedding_size)`，对于 `sparse` 编码，实际大小为 `vocabulary_size`，其中 `vocabulary_size` 是在训练集中对应列特征中出现的不同字符串数（对于 `<UNK>`，加 `1`）。
  * `embeddings_trainable` (默认值 `true`)：如果为 `true`，嵌入向量在训练过程中进行训练，如果为 `false`，嵌入向量是固定的。加载预定义的嵌入向量时，可能会很有用，以避免微调它们。 该参数仅在 `representation` 为 `dense` 时才有效，因为 `sparse` 是 one-hot 编码， 不可训练。
  * `pretrained_embeddings` (默认值 `null`)：默认情况下，`dense` 嵌入向量是随机初始化的，但是此参数允许以 [GloVe格式](https://nlp.stanford.edu/projects/glove/) 指定包含嵌入向量的文件的路径。 加载包含嵌入向量的文件时，仅保留词汇表中带有标签的嵌入向量，其余的则被丢弃。 如果词汇表中包含了那些在嵌入向量文件中不能匹配的字符串，则使用所有其他嵌入向量的平均值加上一些随机噪声来初始化它们的嵌入向量，以使它们彼此不同。 仅当 `representation` 为 `dense` 时此参数才有效。
  * `embeddings_on_cpu` (默认值 `false`)：默认情况下，如果使用 GPU，嵌入向量存储在 GPU 内存中，因为它允许更快的访问，但在某些情况下，嵌入向量可能非常大，此参数强制将嵌入向量放置在常规内存中，并使用 CPU 来解析它们，由于 CPU 和 GPU 内存之间的数据传输，进程稍微减慢。
  * `stacked_layers` (默认值 `null`)：
  * `num_stacked_layers` (默认值 `null`)：
  * `filter_size` (默认值 `3`)：如果在 `conv_layers` 中还没有指定 `filter_size`，这将是每个层使用的默认 `filter_size`。它表示一维卷积滤波器的宽度。
  * `num_filters` (默认值 `256`)：如果 `num_filters` 还没有在 `conv_layers` 中指定，这是默认的将用于每个层的 `num_filters`。它表示滤波器的数量，进而表示一维卷积的输出通道。
  * `pool_function` (默认值 `max`)：池化函数——`max` 为最大值。`average `， `avg` 或 `mean` 都将计算平均值。
  * `pool_size` (默认值 `null`)：如果在 `conv_layers` 中尚未指定 `pool_size`，则将为每个层使用默认的 `pool_size`。它表示卷积操作后沿 `s` 序列维执行的最大池化的大小。
  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`， 和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。如果 `fc_layers` 和 `num_fc_layers` 都为 `null`，则会给 `fc_layers` 分配一个默认列表，值为 `[{fc_size: 512}, {fc_size: 256}]` (仅适用于 `reduce_output` 不为 `null` 的情况)。
  * `num_fc_layers` (默认值 `null`)：如果 `fc_layers` 是 `null`，这是堆叠的完全连接层数(仅适用于 `reduce_output` 不是 `null`)。
  * `fc_size` (默认值 `256`)：`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout` (默认值 `0`)：Dropout 比率。
  * `reduce_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 序列长度维减少输出张量。可用的值有: `sum`， `mean` 或 `avg`， `max`， `concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)和 `null`(不减少并返回整个张量)。

序列特征在输入列表中使用堆叠式并行 CNN 编码器的示例：

```yaml
name: sequence_column_name
type: sequence
encoder: stacked_parallel_cnn
tied_weights: null
representation: dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
stacked_layers: null
num_stacked_layers: null
filter_size: 3
num_filters: 256
pool_function: max
pool_size: null
fc_layers: null
num_fc_layers: null
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
reduce_output: max
```

##### RNN 编码器
rnn 编码器的工作原理是首先将输入整数序列 `b x s`（其中 `b` 是批次大小，`s` 是序列长度）映射到一个嵌入向量序列中，然后通过一个堆叠的循环层（默认为 1 层）传递嵌入向量，接着执行归约操作，默认情况下只返回最后一个输出，但可以执行其他归约函数。如果要输出完整的 `b x s x h`，其中 `h` 是最后一个 rnn 层的输出大小，可以指定 `reduce_output: null`。

```
       +------+
       |Emb 12|
       +------+
+--+   |Emb 7 |
|12|   +------+
|7 |   |Emb 43|                 +---------+
|43|   +------+   +----------+  |Fully    |
|65+--->Emb 65+--->RNN Layers+-->Connected+->
|23|   +------+   +----------+  |Layers   |
|4 |   |Emb 23|                 +---------+
|1 |   +------+
+--+   |Emb 4 |
       +------+
       |Emb 1 |
       +------+
```

以下是 rnn 编码器的可用参数：

  * `representation` (默认值 `dense`)：可能的值是 `dense` 和 `sparse`。`dense` 表示嵌入向量被随机初始化，`sparse` 表示它们被初始化为 one-hot 编码。
  * `embedding_size` (默认值 `256`)：它是最大嵌入向量大小，对于 `dense` 表示，实际大小为 `min(vocabulary_size, embedding_size)`，对于 `sparse` 编码，实际大小为 `vocabulary_size`，其中 `vocabulary_size` 是在训练集中对应列特征中出现的不同字符串数（对于 `<UNK>`，加 `1`）。
  * `embeddings_trainable` (默认值 `true`)：如果为 `true`，嵌入向量在训练过程中进行训练，如果为 `false`，嵌入向量是固定的。加载预定义的嵌入向量时，可能会很有用，以避免微调它们。 该参数仅在 `representation` 为 `dense` 时才有效，因为 `sparse` 是 one-hot 编码， 不可训练。
  * `pretrained_embeddings` (默认值 `null`)：默认情况下，`dense` 嵌入向量是随机初始化的，但是此参数允许以 [GloVe格式](https://nlp.stanford.edu/projects/glove/) 指定包含嵌入向量的文件的路径。 加载包含嵌入向量的文件时，仅保留词汇表中带有标签的嵌入向量，其余的则被丢弃。 如果词汇表中包含了那些在嵌入向量文件中不能匹配的字符串，则使用所有其他嵌入向量的平均值加上一些随机噪声来初始化它们的嵌入向量，以使它们彼此不同。 仅当 `representation` 为 `dense` 时此参数才有效。
  * `embeddings_on_cpu` (默认值 `false`)：默认情况下，如果使用 GPU，嵌入向量存储在 GPU 内存中，因为它允许更快的访问，但在某些情况下，嵌入向量可能非常大，此参数强制将嵌入向量放置在常规内存中，并使用 CPU 来解析它们，由于 CPU 和 GPU 内存之间的数据传输，进程稍微减慢。
  * `num_layers` (默认值 `1`)：循环层的堆叠数量。
  * `state_size` (默认值 `256`)：rnn 状态的大小。
  * `cell_type` (默认值 `rnn`)：要使用的循环单元的类型。 可用值包括： `rnn`, `lstm`, `lstm_block`, `lstm`, `ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`。 有关单元之间差异的参考，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell)。 我们建议在 CPU 上使用 `block` 变体，在 GPU 上使用 `cudnn` 变体，因为它们提高了速度。
  * `bidirectional` (默认值 `false`)：如果为 `true`，则两个循环网络将在前向和后向进行编码，并将它们的输出连接起来。
  * `activation` (默认值 `tanh`)：使用的激活函数。
  * `recurrent_activation` (默认值 `sigmoid`)：在循环步骤中使用的激活函数。
  * `unit_forget_bias` (默认值 `true`)：如果为 `true`，在初始化时给遗忘门的偏差加 1。
  * `recurrent_initializer` (默认值 `orthogonal`)：初始化设定循环矩阵权重。
  * `recurrent_regularizer` (默认值 `null`)：正则化函数应用于循环矩阵权值。
  * `dropout` (默认值 `0.0`)：Dropout 比率。
  * `recurrent_dropout` (默认值 `0.0`)：循环状态的 dropout 比率。
  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，`initializer`， 和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。如果 `fc_layers` 和 `num_fc_layers` 都为 `null`，则会给 `fc_layers` 分配一个默认列表，值为 `[{fc_size: 512}, {fc_size: 256}]` (仅适用于 `reduce_output` 不为 `null` 的情况)。
  * `num_fc_layers` (默认值 `null`)：如果 `fc_layers` 是 `null`，这是堆叠的完全连接层数(仅适用于 `reduce_output` 不是 `null`)。
  * `fc_size` (默认值 `256`)：`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `fc_activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `fc_dropout` (默认值 `0`)：Dropout 比率。
  * `reduce_output` (默认值 `last`)：定义如果张量的秩大于 2，如何沿 `s` 序列长度维减少输出张量。可用的值有: `sum`， `mean` 或 `avg`， `max`， `concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)和 `null`(不减少并返回整个张量)。

序列特征在输入列表中使用 RNN 编码器的示例：

```yaml
name: sequence_column_name
type: sequence
encoder: rnn
tied_weights: null
representation': dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
num_layers: 1
state_size: 256
cell_type: rnn
bidirectional: false
activation: tanh
recurrent_activation: sigmoid
unit_forget_bias: true
recurrent_initializer: orthogonal
recurrent_regularizer: null
dropout: 0.0
recurrent_dropout: 0.0
fc_layers: null
num_fc_layers: null
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
fc_activation: relu
fc_dropout: 0
reduce_output: last
```

##### CNN RNN 编码器
`cnnrnn` 编码器的工作原理是，首先将输入整数序列 `b x s`（其中 `b` 是批次大小，`s` 是序列长度）映射到一个嵌入向量序列中，然后将嵌入向量通过一个堆叠的卷积层（默认为 2），然后是一个堆叠的循环层（默认为 1），接着是归约操作，默认情况下只返回最后一个输出，但可以执行其他归约函数。如果要输出完整的 `b x s x h`(其中 `h` 是最后一个 rnn 层的输出大小)可以指定 `reduce_output: null`。

```
       +------+
       |Emb 12|
       +------+
+--+   |Emb 7 |
|12|   +------+
|7 |   |Emb 43|                                +---------+
|43|   +------+   +----------+   +----------+  |Fully    |
|65+--->Emb 65+--->CNN Layers+--->RNN Layers+-->Connected+->
|23|   +------+   +----------+   +----------+  |Layers   |
|4 |   |Emb 23|                                +---------+
|1 |   +------+
+--+   |Emb 4 |
       +------+
       |Emb 1 |
       +------+
```

以下是 cnn rnn 编码器的可用参数：

  * `representation` (默认值 `dense`)：可能的值是 `dense` 和 `sparse`。`dense` 表示嵌入向量被随机初始化，`sparse` 表示它们被初始化为 one-hot 编码。
  * `embedding_size` (默认值 `256`)：它是最大嵌入向量大小，对于 `dense` 表示，实际大小为 `min(vocabulary_size, embedding_size)`，对于 `sparse` 编码，实际大小为 `vocabulary_size`，其中 `vocabulary_size` 是在训练集中对应列特征中出现的不同字符串数（对于 `<UNK>`，加 `1`）。
  * `embeddings_trainable` (默认值 `true`)：如果为 `true`，嵌入向量在训练过程中进行训练，如果为 `false`，嵌入向量是固定的。加载预定义的嵌入向量时，可能会很有用，以避免微调它们。 该参数仅在 `representation` 为 `dense` 时才有效，因为 `sparse` 是 one-hot 编码， 不可训练。
  * `pretrained_embeddings` (默认值 `null`)：默认情况下，`dense` 嵌入向量是随机初始化的，但是此参数允许以 [GloVe格式](https://nlp.stanford.edu/projects/glove/) 指定包含嵌入向量的文件的路径。 加载包含嵌入向量的文件时，仅保留词汇表中带有标签的嵌入向量，其余的则被丢弃。 如果词汇表中包含了那些在嵌入向量文件中不能匹配的字符串，则使用所有其他嵌入向量的平均值加上一些随机噪声来初始化它们的嵌入向量，以使它们彼此不同。 仅当 `representation` 为 `dense` 时此参数才有效。
  * `embeddings_on_cpu` (默认值 `false`)：默认情况下，如果使用 GPU，嵌入向量存储在 GPU 内存中，因为它允许更快的访问，但在某些情况下，嵌入向量可能非常大，此参数强制将嵌入向量放置在常规内存中，并使用 CPU 来解析它们，由于 CPU 和 GPU 内存之间的数据传输，进程稍微减慢。
  * `conv_layers` (默认值 `null`)：它是一个字典列表，包含所有卷积层的参数。列表的长度决定并行卷积层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是:`filter_size`, `num_filters`, `pool`, `norm`, `activation` 和 `regularize`。如果字典中缺少这些值中的任何一个，则将使用作为编码器参数指定的默认值。如果 `conv_layers` 和 `num_conv_layers` 都为 `null`，则会给 `conv_layers` 赋一个默认列表，值为 `[{filter_size: 7, pool_size: 3, regularize: false}, {filter_size: 7, pool_size: 3, regularize: false}, {filter_size: 3, pool_size: null, regularize: false}, {filter_size: 3, pool_size: null, regularize: false}, {filter_size: 3, pool_size: null, regularize: true}, {filter_size: 3, pool_size: 3, regularize: true}]`。
  * `num_conv_layers` (默认值 `1`)：如果 `conv_layers` 是 `null`，这是堆叠卷积层的数量。
  * `num_filters` (默认值 `256`)：如果 `num_filters` 还没有在 `conv_layers` 中指定，这是默认的将用于每个层的 `num_filters`。它表示滤波器的数量，进而表示一维卷积的输出通道。
  * `filter_size` (默认值 `5`)：如果在 `conv_layers` 中还没有指定 `filter_size`，这将是每个层使用的默认 `filter_size`。它表示一维卷积滤波器的宽度。
  * `strides` (默认值 `1`)：卷积的步长。
  * `padding` (默认值 `same`)：`valid` 或 `same`。
  * `dilation_rate` (默认值 `1`)：用于扩张卷积的扩张率。
  * `conv_activation` (默认值 `relu`)：卷积层激活函数。
  * `conv_dropout` (默认值 `0.0`)：卷积层的 dropout 比率。
  * `pool_function` (默认值 `max`)：池化函数——`max` 为最大值。`average `， `avg` 或 `mean` 都将计算平均值。
  * `pool_size` (默认值 `2`)：如果在 `conv_layers` 中尚未指定 `pool_size`，则将为每个层使用默认的 `pool_size`。它表示卷积操作后沿 `s` 序列维执行的最大池化的大小。
  * `pool_strides` (默认值 `null`)：缩小因子。
  * `pool_padding` (默认值 `same`)：`valid` 或 `same`。
  * `num_rec_layers` (默认值 `1`)：循环层数。
  * `state_size` (默认值 `256`)：rnn 状态的大小。
  * `cell_type` (默认值 `rnn`)：要使用的循环单元的类型。 可用值包括： `rnn`, `lstm`, `lstm_block`, `lstm`, `ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`。 有关单元之间差异的参考，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell)。 我们建议在 CPU 上使用 `block` 变体，在 GPU 上使用 `cudnn` 变体，因为它们提高了速度。
  * `bidirectional` (默认值 `false`)：如果为 `true`，则两个循环网络将在前向和后向进行编码，并将它们的输出连接起来。
  * `activation` (默认值 `tanh`)：使用的激活函数。
  * `recurrent_activation` (默认值 `sigmoid`)：在循环步骤中使用的激活函数。
  * `unit_forget_bias` (默认值 true)：如果为 `true`，在初始化时给遗忘门的偏差加 1。
  * `recurrent_initializer` (默认值 `orthogonal`)：初始化设定循环矩阵权重。
  * `recurrent_regularizer` (默认值 `null`)：正则化函数应用于循环矩阵权值。
  * `dropout` (默认值 `0.0`)：Dropout 比率。
  * `recurrent_dropout` (默认值 `0.0`)：循环状态的 dropout 比率。
  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，`initializer`， 和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。如果 `fc_layers` 和 `num_fc_layers` 都为 `null`，则会给 `fc_layers` 分配一个默认列表，值为 `[{fc_size: 512}, {fc_size: 256}]` (仅适用于 `reduce_output` 不为 `null` 的情况)。
  * `num_fc_layers` (默认值 `null`)：如果 `fc_layers` 是 `null`，这是堆叠的完全连接层数(仅适用于 `reduce_output` 不是 `null`)。
  * `fc_size` (默认值 `256`)：`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `fc_activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `fc_dropout` (默认值 `0`)：Dropout 比率。
  * `reduce_output` (默认值 `last`)：定义如果张量的秩大于 2，如何沿 `s` 序列长度维减少输出张量。可用的值有: `sum`， `mean` 或 `avg`， `max`， `concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)和 `null`(不减少并返回整个张量)。

序列特征在输入列表中使用 cnn rnn 编码器的示例：

```yaml
name: sequence_column_name
type: sequence
encoder: cnnrnn
tied_weights: null
representation: dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
conv_layers: null
num_conv_layers: 1
num_filters: 256
filter_size: 5
strides: 1
padding: same
dilation_rate: 1
conv_activation: relu
conv_dropout: 0.0
pool_function: max
pool_size: 2
pool_strides: null
pool_padding: same
num_rec_layers: 1
state_size: 256
cell_type: rnn
bidirectional: false
activation: tanh
recurrent_activation: sigmoid
unit_forget_bias: true
recurrent_initializer: orthogonal
recurrent_regularizer: null
dropout: 0.0
recurrent_dropout: 0.0
fc_layers: null
num_fc_layers: null
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
fc_activation: relu
fc_dropout: 0
reduce_output: last
```

##### TRANSFORMER 编码器
transformer 编码器实现了一个堆叠的 transformer 块，复制了注意力机制，并在末尾添加了一个可选的完全连接层堆叠。

```
       +------+                     
       |Emb 12|                     
       +------+                     
+--+   |Emb 7 |                     
|12|   +------+                     
|7 |   |Emb 43|   +-------------+   +---------+ 
|43|   +------+   |             |   |Fully    |
|65+---+Emb 65+---> Transformer +--->Connected+->
|23|   +------+   | Blocks      |   |Layers   |
|4 |   |Emb 23|   +-------------+   +---------+
|1 |   +------+                     
+--+   |Emb 4 |                     
       +------+                     
       |Emb 1 |                     
       +------+                     
```

以下是 transformer 编码器的可用参数：

  * `representation` (默认值 `dense`)：可能的值是 `dense` 和 `sparse`。`dense` 表示嵌入向量被随机初始化，`sparse` 表示它们被初始化为 one-hot 编码。
  * `embedding_size` (默认值 `256`)：它是最大嵌入向量大小，对于 `dense` 表示，实际大小为 `min(vocabulary_size, embedding_size)`，对于 `sparse` 编码，实际大小为 `vocabulary_size`，其中 `vocabulary_size` 是在训练集中对应列特征中出现的不同字符串数（对于 `<UNK>`，加 `1`）。
  * `embeddings_trainable` (默认值 `true`)：如果为 `true`，嵌入向量在训练过程中进行训练，如果为 `false`，嵌入向量是固定的。加载预定义的嵌入向量时，可能会很有用，以避免微调它们。 该参数仅在 `representation` 为 `dense` 时才有效，因为 `sparse` 是 one-hot 编码， 不可训练。
  * `pretrained_embeddings` (默认值 `null`)：默认情况下，`dense` 嵌入向量是随机初始化的，但是此参数允许以 [GloVe格式](https://nlp.stanford.edu/projects/glove/) 指定包含嵌入向量的文件的路径。 加载包含嵌入向量的文件时，仅保留词汇表中带有标签的嵌入向量，其余的则被丢弃。 如果词汇表中包含了那些在嵌入向量文件中不能匹配的字符串，则使用所有其他嵌入向量的平均值加上一些随机噪声来初始化它们的嵌入向量，以使它们彼此不同。 仅当 `representation` 为 `dense` 时此参数才有效。
  * `embeddings_on_cpu` (默认值 `false`)：默认情况下，如果使用 GPU，嵌入向量存储在 GPU 内存中，因为它允许更快的访问，但在某些情况下，嵌入向量可能非常大，此参数强制将嵌入向量放置在常规内存中，并使用 CPU 来解析它们，由于 CPU 和 GPU 内存之间的数据传输，进程稍微减慢。
  * `num_layers` (默认值 `1`)：transformer 块数量。
  * `hidden_size` (默认值 `256`)：transformer 块内隐藏表示的大小。它通常与 `embeddding_size` 相同，但如果两个值不同，投影层将被添加到第一个 transformer块之前。
  * `num_heads` (默认值 `8`)：transformer 块内的自注意力头数(多头注意力机制——译者注)。
  * `transformer_fc_size` (默认值 `256`)：transformer 块内自注意后完全连接层的大小 。这通常与 `hidden_size` 和 `embeddding_size` 相同。
  * `dropout` (默认值 `0.1`)：transformer 块的 dropout 比率。
  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，`initializer`， 和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。如果 `fc_layers` 和 `num_fc_layers` 都为 `null`，则会给 `fc_layers` 分配一个默认列表，值为 `[{fc_size: 512}, {fc_size: 256}]` (仅适用于 `reduce_output` 不为 `null` 的情况)。
  * `num_fc_layers` (默认值 `0`)：这是堆叠的完全连接层数(仅适用于 `reduce_output` 不是 `null`)。
  * `fc_size` (默认值 `256`)：`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `fc_activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `fc_dropout` (默认值 `0`)：Dropout 比率。
  * `reduce_output` (默认值 `last`)：定义如果张量的秩大于 2，如何沿 `s` 序列长度维减少输出张量。可用的值有: `sum`， `mean` 或 `avg`， `max`， `concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)和 `null`(不减少并返回整个张量)。

序列特征在输入列表中使用 transformer 编码器的示例：

```yaml
name: sequence_column_name
type: sequence
encoder: transformer
tied_weights: null
representation: dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
num_layers: 1
hidden_size: 256
num_heads: 8
transformer_fc_size: 256
dropout: 0.1
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
fc_activation: relu
fc_dropout: 0
reduce_output: last
```

##### PASSTHROUGH 编码器
passthrough 编码器只需将每个输入值转换为一个浮点值，并向输入张量添加一个维度，从而创建一个 `b x s x 1` 张量，其中 `b` 是批次大小，`s` 是序列长度。张量将沿 `s` 维缩减，以获得批次中每个元素的大小为 `h` 的单个向量。如果要输出完整的 `b x s x h` 张量，可以指定 `reduce_output: null`。此编码器对于 `sequence` 或 `text` 特征并不真正有用，但对于 `timeseries` 特征可能有用，因为它允许在模型的后期使用它们而无需任何处理，例如在序列组合器中。

```
+--+   
|12|   
|7 |                    +-----------+
|43|   +------------+   |Aggregation|
|65+--->Cast float32+--->Reduce     +->
|23|   +------------+   |Operation  |
|4 |                    +-----------+
|1 |   
+--+   
```

这些是可用于 passthrough 编码器的参数：

  * `reduce_output` (默认值 `null`)：定义如果张量的秩大于 2，如何沿 `s` 序列长度维减少输出张量。可用的值有: `sum`， `mean` 或 `avg`， `max`， `concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)和 `null`(不减少并返回整个张量)。

序列特征在输入列表中使用 passthrough 编码器的示例：

```yaml
name: sequence_column_name
type: sequence
encoder: passthrough
reduce_output: null
```

#### 序列输出特征和解码器
当需要执行序列标记（对输入序列的每个元素进行分类）或序列生成时，可以使用序列特征。有两个解码器可用于那些名为 `tagger` 和 `generator` 的任务。

这些是序列输出特征的可用参数：

  * `reduce_input` (默认值 `sum`)：定义如何在第一维(如果算上批次维数，则是第二维)上减少不是向量而是矩阵或更高阶张量的输入。可用的值有：`sum`，`mean` 或 `avg`，`max`，`concat` (沿第一个维度连接)，`last` (返回第一个维度的最后一个向量)。
  * `dependencies` (默认值 `[]`)：它所依赖的输出特征。有关详细解释，请参阅[输出特征依赖关系](https://ludwig-ai.github.io/ludwig-docs/user_guide/#output-features-dependencies)。
  * `reduce_dependencies` (默认值 `sum`)：定义如何在第一维(如果算上批次维度，则是第二维)上减少一个依赖特征(不是向量，而是一个矩阵或更高阶张量)的输出。可用的值有：`sum`，`mean` 或 `avg`，`max`，`concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)。
  * `loss` (默认值 `{type: softmax_cross_entropy, class_similarities_temperature: 0, class_weights: 1, confidence_penalty: 0, distortion: 1, labels_smoothing: 0, negative_samples: 0, robust_lambda: 0, sampler: null, unique: false}`)：是一个包含损失 `type` 的字典。可用的损失 `type` 是 `softmax_cross_entropy` 和 `sampled_softmax_cross_entropy` 。关于这两种损失的详细信息，请参考[类别输出特征部分](#类别输出特征和解码器)。

##### TAGGER 解码器
在 `tagger` 的情况下，解码器是一个堆叠的完全连接层（可能是空的），然后是一个大小为 `b x s x c` 的张量投影，其中 `b` 是批次大小，`s` 是序列的长度，`c` 是类别数量，接着是 `softmax_cross_entropy`。此解码器要求其输入形状为 `b x s x h`，其中 `h` 是一个隐藏维度，它是输入的序列、文本或时间序列特征的输出，没有减少输出或基于序列的组合器的输出。如果改为提供 `b x h` 输入，则在构建模型期间将引发错误。

```
Combiner
Output

+---+                 +----------+   +-------+
|emb|   +---------+   |Projection|   |Softmax|
+---+   |Fully    |   +----------+   +-------+
|...+--->Connected+--->...       +--->...    |
+---+   |Layers   |   +----------+   +-------+
|emb|   +---------+   |Projection|   |Softmax|
+---+                 +----------+   +-------+
```

以下是 TAGGER 解码器的可用参数：

  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，`dropout`，`initializer`， 和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。
  * `num_fc_layers` (默认值 `0`)：这是输入特征经过的堆叠的完全连接层的数量。它们的输出被投影到特征的输出空间中。
  * `fc_size` (默认值 `256`)：`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout` (默认值 `0`)：Dropout 比率。
  * `attention` (默认值 `false`)：如果为 `true`，则在预测前应用一个多头自注意层。
  * `attention_embedding_size` (默认值 `256`)：多头自注意层的嵌入向量大小。
  * `attention_num_heads` (默认值 `8`)：在多头自注意层中注意头的数目。

输出特征列表中的序列特征(具有默认参数)使用 tagger 解码器示例：

```yaml
name: sequence_column_name
type: sequence
decoder: tagger
reduce_input: null
dependencies: []
reduce_dependencies: sum
loss:
    type: softmax_cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    class_weights: 1
    class_similarities: null
    class_similarities_temperature: 0
    labels_smoothing: 0
    negative_samples: 0
    sampler: null
    distortion: 1
    unique: false
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
attention: false
attention_embedding_size: 256
attention_num_heads: 8
```

##### GENERATOR 解码器
在 `generator` 的情况下，解码器是一个堆叠的完全连接层（可能是空的），后跟一个 rnn，该 rnn 根据自己先前的预测生成输出，并生成一个大小为 `b x s' x c` 的张量，其中 `b` 是批次大小，`s` 是生成序列的长度，`c` 是类别的数量，接着是 `softmax_cross_entropy`。在训练期间，采用 teacher forcing，这意味着目标列表同时作为输入和输出（移位 1），而在评估时，贪婪解码（一次生成一个标记并将其作为下一步的输入）通过束搜索(beam search)执行(默认束宽为 1)。默认情况下，生成器需要一个 `b x h` 形的输入张量，其中 `h` 是一个隐藏维度。`h` 向量（在一个可选的完全连接层堆叠之后）馈入 rnn 生成器。一个例外是当生成器使用注意力时，如在这种情况下，输入张量的预期大小为 `b x s x h`，这是输入的序列、文本或时间序列特征的输出，没有减少输出，或者是基于序列的组合器的输出。如果使用 rnn 将 `b x h` 输入提供给生成器—解码器并使用注意力，则在建模过程中会出现错误。

```
                            Output     Output
                               1  +-+    ... +--+    END
                               ^    |     ^     |     ^
+--------+   +---------+       |    |     |     |     |
|Combiner|   |Fully    |   +---+--+ | +---+---+ | +---+--+
|Output  +--->Connected+---+RNN   +--->RNN... +--->RNN   |
|        |   |Layers   |   +---^--+ | +---^---+ | +---^--+
+--------+   +---------+       |    |     |     |     |
                              GO    +-----+     +-----+
```

以下是 GENERATOR 解码器的可用参数：

  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，`dropout`，`initializer`， 和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。
  * `num_fc_layers` (默认值 `0`)：这是输入特征经过的堆叠的完全连接层的数量。它们的输出被投影到特征的输出空间中。
  * `fc_size` (默认值 `256`)：`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout` (默认值 `0`)：Dropout 比率。
  * `cell_type` (默认值 `rnn`)：要使用的循环单元的类型。 可用值包括： `rnn`, `lstm`, `lstm_block`, `lstm`, `ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`。 有关单元之间差异的参考，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell)。 我们建议在 CPU 上使用 `block` 变体，在 GPU 上使用 `cudnn` 变体，因为它们提高了速度。
  * `state_size` (默认值 `256`)：rnn 状态的大小。
  * `embedding_size` (默认值 `256`)：如果 `tied_target_embeddings` 为 `false`，则输入的嵌入向量和之前的 `softmax_cross_entropy` 权重没有捆绑在一起，并且可以有不同的大小，此参数描述生成器输入的嵌入向量大小。
  * `beam_width` (默认值 `1`)：利用束搜索从 rnn 生成器中进行采样。默认情况下，当束宽为 `1` 时，只会生成一个总是使用最有可能的下一个标记的贪婪序列，但束宽可以增加。这通常会带来更好的性能，但代价是更多的计算量和更慢的生成速度。
  * `attention` (默认值 `null`)：循环生成器可能使用注意力机制。可用的是 `bahdanau` 和 `luong`（有关更多信息，请参阅[TensorFlow文档](https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention)）。当 `attention` 不为 `null` 时，输入张量的预期大小为 `b x s x h`，这是输入的序列、文本或时间序列特征的输出，没有减少输出或基于序列的组合器的输出。如果使用 rnn 将 `b x h` 输入提供给生成器—解码器并使用注意力，则在建模过程中会出现错误。
  * `tied_embeddings` (默认值 `null`)：如果 `null`，目标的嵌入向量将被随机初始化，而如果值是输入特征的名称，输入特征的嵌入向量将被用作目标的嵌入向量。输入特征的 `vocabulary_size` 必须与输出特征相同，并且它必须有一个嵌入矩阵(例如，二进制和数字特征将没有嵌入矩阵)。在这种情况下，`embeddding_size` 将与 `state_size` 相同。这对于实现模型的编码和解码部分共享参数的自动编码器是很有用的。
  * `max_sequence_length` (默认值 `0`)：(原文为空——译者注)

输出特征列表中的序列特征(具有默认参数)使用 generator 解码器示例：

```yaml
name: sequence_column_name
type: sequence
decoder: generator
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: softmax_cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    class_weights: 1
    class_similarities: null
    class_similarities_temperature: 0
    labels_smoothing: 0
    negative_samples: 0
    sampler: null
    distortion: 1
    unique: false
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
cell_type: rnn
state_size: 256
embedding_size: 256
beam_width: 1
attention: null
tied_embeddings: null
max_sequence_length: 0
```

#### 序列特征度量<a id='序列特征度量'></a>
每个周期都计算并可用于序列特征的度量有 `accuracy`(计算预测序列的所有元素在所有数据点的数量上正确的数据点的数量)，`token_accuracy`(计算在所有数据点的数量上正确预测的所有序列中的元素的数量)，`last_accuracy`(精度仅考虑序列的最后一个元素，有助于确保生成或标记特殊的序列结束标记)，`edit_distance`(预测值和正确标注序列之间的 levenshtein 距离)，`perplexity`(基于模型的正确标注序列的困惑度)和损失本身。如果将 `validation_field` 设置为类别特征的名称，则可以在配置的 `training` 部分中将其中一个设置为 `validation_measure`。

### Text 特征<a id='Text_特征'></a>
#### 文本特征预处理
文本特征的处理方式与序列特征相同，只是有一些区别。有两种不同的分词，一种按每个字符拆分，另一种按空格和标点拆分，并将两个不同的键添加到 HDF5 文件中，一个包含字符矩阵，一个包含单词矩阵。同样的事情也发生在 JSON 文件中，它包含用于将字符映射到整数(及其逆)和单词映射到整数(及其逆)的字典。在配置中，您可以指定是字符级别或单词级别的表示。

可用于预处理的参数如下：

  * `char_tokenizer` (默认值 `characters`)：定义如何从数据集列的原始字符串内容映射到字符序列。默认值和唯一可用的选项是 `characters`，其行为是在每个字符处拆分字符串。
  * `char_vocab_file` (默认值 `null`)：
  * `char_sequence_length_limit` (默认值 `1024`)：以字符表示的文本的最大长度。长于此值的文本将被截断，而较短的序列将被填充。
  * `char_most_common` (默认值 `70`)：要考虑的最常见字符的最大数目。如果数据包含超过这个数量，最不常见的字符将被视为未知字符。
  * `word_tokenizer` (默认值 `space_punct`)：定义如何从dataset列的原始字符串内容映射到元素序列。有关可用选项，请参阅[分词器](#分词器)一节。
  * `pretrained_model_name_or_path` (默认值 `null`)：
  * `word_vocab_file` (默认值 `null`)：
  * `word_sequence_length_limit` (默认值 `256`)：以单词表示的文本的最大长度。长于此值的文本将被截断，而较短的文本将被填充。
  * `word_most_common` (默认值 `20000`)：要考虑的最常见单词的最大数目。如果数据包含超过这个数量，最不常见的单词将被视为未知。
  * `padding_symbol` (默认值 `<PAD>`)：用作填充符号的字符串。它被映射到词汇表中的整数 ID 0。
  * `unknown_symbol` (默认值 `<UNK>`)：用作未知符号的字符串。它被映射到词汇表中的整数ID 1。
  * `padding` (默认值 `right`)：填充的方向。`right` 和 `left` 是可用的选项。
  * `lowercase` (默认值 `false`)：分词器处理前必须小写字符串。
  * `missing_value_strategy` (默认值 `fill_with_const`)：当文本列中缺少一个值时，应该遵循什么策略。该值应该是 `fill_with_const` (用 `fill_value` 参数指定一个特定的值替换缺失的值)，`fill_with_mode` (用列中最常见的值替换缺失的值), `fill_with_mean` (用列中的均值替换缺失的值)，`backfill` (用下一个有效值替换缺失的值)。
  * `fill_value` (默认值 `""`)：在 `missing_value_strategy` 的参数值为 `fill_with_const` 情况下指定的特定值。

文本预处理示例：

```yaml
name: text_column_name
type: text
level: word
preprocessing:
    char_tokenizer: characters
    char_vocab_file: null
    char_sequence_length_limit: 1024
    char_most_common: 70
    word_tokenizer: space_punct
    pretrained_model_name_or_path: null
    word_vocab_file: null
    word_sequence_length_limit: 256
    word_most_common: 20000
    padding_symbol: <PAD>
    unknown_symbol: <UNK>
    padding: right
    lowercase: false
    missing_value_strategy: fill_with_const
    fill_value: ""
```

#### 文本输入特征和编码器
文本输入特征参数是：

  * `encoder` (默认值 `parallel_cnn`)：用于输入文本特征的编码器。可用的编码器来自[序列输入特征和编码器](#序列输入特征和编码器)以及这些对文本特定的编码器：`bert`, `gpt`, `gpt2`, `xlnet`, `xlm`, `roberta`, `distilbert`, `ctrl`, `camembert`, `albert`, `t5`, `xlmroberta`, `flaubert`, `electra`, `longformer` 和 `auto-transformer`。
  * `level` (默认值 `word`)：`word` 指定使用文本单词，`char` 使用单个字符。
  * `tied_weights` (默认值 `null`)：绑定编码器权重的输入特征的名称。它必须具有相同类型和相同编码器参数的特征名称。

##### BERT 编码器
`bert` 编码器使用 Hugging Face transformers 包加载一个预先训练的 [BERT](https://arxiv.org/abs/1810.04805) 模型。

  * `pretrained_model_name_or_path` (默认值 `bert-base-uncased`)：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `cls_pooled`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`cls_pool`、`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。

##### GPT 编码器
`gpt` 编码器使用 Hugging Face transformers 包加载一个预先训练的 [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 模型。

  * `pretrained_model_name_or_path` (默认值 `openai-gpt`)：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。

##### GPT-2 编码器
`gpt2` 编码器使用 Hugging Face transformers 包加载一个预先训练的 [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 模型。

  * `pretrained_model_name_or_path` (默认值 `gpt2`)：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。

##### XLNET 编码器
`xlnet` 编码器使用 Hugging Face transformers 包加载一个预先训练的 [XLNet](https://arxiv.org/abs/1906.08237) 模型。

  * `pretrained_model_name_or_path` (默认值 `xlnet-base-cased`)：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。

##### XLM 编码器
`xlm` 编码器使用 Hugging Face transformers 包加载一个预先训练的 [XLM](https://arxiv.org/abs/1901.07291) 模型。

  * `pretrained_model_name_or_path` (默认值 `xlm-mlm-en-2048`)：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。


##### ROBERTA 编码器
`roberta` 编码器使用 Hugging Face transformers 包加载一个预先训练的 [RoBERTa](https://arxiv.org/abs/1907.11692) 模型。

  * `pretrained_model_name_or_path` (默认值 `roberta-base`)：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`cls_pool`、`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。

##### DISTILBERT 编码器
`distilbert` 编码器使用 Hugging Face transformers 包加载一个预先训练的 [DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5) 模型。

  * `pretrained_model_name_or_path` (默认值 `istilbert-base-uncased`)：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。

##### CTRL 编码器
`ctrl` 编码器使用 Hugging Face transformers 包加载一个预先训练的 [CTRL](https://arxiv.org/abs/1909.05858) 模型。

  * `pretrained_model_name_or_path` (默认值 `ctrl`)：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。

##### CAMEMBERT 编码器
`camembert` 编码器使用 Hugging Face transformers 包加载一个预先训练的 [CamemBERT](https://arxiv.org/abs/1911.03894) 模型。

  * `pretrained_model_name_or_path` (默认值 `jplu/tf-camembert-base`)：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`cls_pool`、`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。

##### ALBERT 编码器
`albert` 编码器使用 Hugging Face transformers 包加载一个预先训练的 [ALBERT](https://arxiv.org/abs/1909.11942) 模型。

  * `pretrained_model_name_or_path` (默认值 `albert-base-v2`)：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`cls_pool`、`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。

##### T5 编码器
`t5` 编码器使用 Hugging Face transformers 包加载一个预先训练的 [T5](https://arxiv.org/pdf/1910.10683.pdf) 模型。

  * `pretrained_model_name_or_path` (默认值 `t5-small`)：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。

##### XLM-ROBERTA 编码器
`xlmroberta` 编码器使用 Hugging Face transformers 包加载一个预先训练的 [XLM-RoBERTa](https://arxiv.org/abs/1911.02116) 模型。

  * `pretrained_model_name_or_path` (默认值 `jplu/tf-xlm-reoberta-base`)：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`cls_pool`、`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。

##### FLAUBERT 编码器
`flaubert` 编码器使用 Hugging Face transformers 包加载一个预先训练的 [FlauBERT](https://arxiv.org/abs/1912.05372) 模型。

  * `pretrained_model_name_or_path` (默认值 `jplu/tf-flaubert-base-uncased`)：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。

##### ELECTRA 编码器
`electra` 编码器使用 Hugging Face transformers 包加载一个预先训练的 [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) 模型。

  * `pretrained_model_name_or_path` (默认值 `google/electra-small-discriminator`)：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。

##### LONGFORMER 编码器
`longformer` 编码器使用 Hugging Face transformers 包加载一个预先训练的 [Longformer](https://arxiv.org/pdf/2004.05150.pdf) 模型。

  * `pretrained_model_name_or_path` (默认值 `allenai/longformer-base-4096`)：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`cls_pool`、`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。

##### AUTO-TRANSFORMER 编码器
`auto_transformer` 编码器使用 Hugging Face transformers 包加载一个预先训练的模型。对于不适合的其他预训练 transformer 编码器，这是最好的选择。

  * `pretrained_model_name_or_path` ：它可以是模型的名称，也可以是下载模型的路径。有关可用变型的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/transformers/model_doc/bert.html)。
  * `reduced_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 文本长度维减少输出张量。可用的值有：`sum`、`mean` 或 `avg`、`max`、`concat`(沿第一个维度连接)、`last`(返回第一个维度的最后一个向量)和 `null` (不减少并返回整个张量)。
  * `trainable` (默认值 `false`)：如果为 `true`，编码器的权重将被训练，否则它们将被冻结。

##### 使用示例
文本输入特征编码器用法示例：

```yaml
name: text_column_name
type: text
level: word
encoder: bert
tied_weights: null
pretrained_model_name_or_path: bert-base-uncased
reduced_output: cls_pooled
trainable: false
```

#### 文本输出特征和解码器
解码器与序列特征相同。唯一的区别是，您可以指定一个额外的 `level` 参数，它的值可能是 `word` 或 `char`，以强制使用文本单词或字符作为输入(默认情况下，编码器将使用 `word`)。

使用默认值的文本输入特征示例：

```yaml
name: sequence_column_name
type: text
level: word
decoder: generator
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: softmax_cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    class_weights: 1
    class_similarities: null
    class_similarities_temperature: 0
    labels_smoothing: 0
    negative_samples: 0
    sampler: null
    distortion: 1
    unique: false
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
cell_type: rnn
state_size: 256
embedding_size: 256
beam_width: 1
attention: null
tied_embeddings: null
max_sequence_length: 0
```

#### 文本特征度量
与[序列特征度量](#序列特征度量)相同。

### Time Series 特征<a id='Time_Series_特征'></a>
#### 时间序列特征预处理
时间序列特征的处理方式与序列特征相同，唯一的区别是 HDF5 文件中的矩阵没有整数值，而是浮点值。而且，JSON 文件中不需要任何映射。

#### 时间序列输入特征和编码器
编码器与[序列输入特征和编码器](#序列输入特征和编码器)相同。唯一的区别是时间序列特征在开始时没有嵌入向量层，因此 `b x s` 占位符（其中`b` 是批次大小，`s` 是序列长度）直接映射到 `b x s x 1` 张量，然后传递到不同的序列编码器。

#### 时间序列输出特性和解码器
目前还没有时间序列解码器(WIP) ，因此时间序列不能作为输出特征。

#### 时间序列特征度量
由于目前没有时间序列解码器，因此也没有时间序列度量。

### Audio 特征<a id='Audio_特征'></a>
#### 音频特征预处理
Ludwig 支持使用 Python SoundFile 库读取音频文件，因此支持 WAV、 FLAC、 OGG 和 MAT 文件。

  * `audio_file_length_limit_in_s` (默认值 `7.5`)：浮点值，以秒为单位定义音频文件的最大限制。所有长于此限制的文件都将被截断。所有短于此限制的文件都用 `padding_value` 填充。
  * `missing_value_strategy` (默认值 `backfill`)：当音频列中缺少一个值时，应该遵循什么策略。该值应该是 `fill_with_const` (用 `fill_value` 参数指定一个特定的值替换缺失的值)，`fill_with_mode` (用列中最常见的值替换缺失的值), `fill_with_mean` (用列中的均值替换缺失的值)，`backfill` (用下一个有效值替换缺失的值)。
  * `in_memory` (默认值 `true`)：定义音频数据集在训练过程中是驻留在内存中还是从磁盘动态获取(对于大数据集很有用)。在后一种情况下，每次训练迭代都会从磁盘中获取一批输入音频的训练。目前只支持 `in_memory` = true。
  * `padding_value` (默认值 `0`)：用于填充的浮点值。
  * `norm` (默认值 `null`)：可用于输入数据的标准化方法。支持的方法：`null`（数据未标准化），`per_file`（z-norm 应用于“per file”级别）。
  * `audio_feature` (默认值 `{ type: raw }`)：将音频特征 `type` 以及附加参数 `type != raw` 作为输入的字典。以下参数可以/应该在字典中定义：
    * `type` (默认值 `raw`)：定义要使用的音频特征的类型。目前支持的类型有 `raw`、`stft`、`stft_phase`、`group_delay`。要了解更多细节，请查阅[音频输入特征和编码器](https://ludwig-ai.github.io/ludwig-docs/user_guide/#audio-input-features-and-encoders)。
    * `window_length_in_s`：定义用于短时间傅里叶变换的窗口长度(仅在 `type != raw` 时需要)。
    * `window_shift_in_s`：定义用于短时间傅里叶变换(也称为 hop_length)的窗口移位(仅在 `type != raw` 时需要)。
    * `num_fft_points` (音频文件的默认值 `window_length_in_s * sample_rate`)：定义用于短时间傅里叶变换的 fft 点数。如果 `num_fft_points > window_length_in_s * sample_rate`，那么信号在最后是零填充的。`num_fft_points` 必须是 `>= window_length_in_s * sample_rate`(仅在 `type != raw` 时需要)。
    * `window_type` (默认值 `hamming`)：在短时傅里叶变换之前定义信号加权的类型窗口。[scipy's window function](scipy’s window function)提供的所有窗口都可以使用（仅当 `type！=raw`）。
    * `num_filter_bands`：定义滤波器组中使用的滤波器数量(仅在 `type == fbank` 时需要)。

预处理规范示例(假设音频文件的采样率为 16000)：

```yaml
name: audio_path
type: audio
preprocessing:
  audio_file_length_limit_in_s: 7.5
  audio_feature:
    type: stft
    window_length_in_s: 0.04
    window_shift_in_s: 0.02
    num_fft_points: 800
    window_type: boxcar
```

#### 音频输入特征和编码器
音频文件根据 `preprocessing` 中的 `audio_feature` 中的 `type` 转换为以下类型之一。

  * `raw`：音频文件被转换成大小为 `N x L x W` 的浮点值张量(其中 `N` 是数据集的大小，`L` 对应于 `audio_file_length_limit_in_s * sample_rate and W = 1`)。
  * `stft`：音频被转换成 `stft` 量级。音频文件转换成大小为 `N x L x W`(在 `N` 是数据集的大小，`L` 对应于 `ceil(audio_file_length_limit_in_s * sample_rate - window_length_in_s * sample_rate + 1/ window_shift_in_s * sample_rate) + 1` 和 `W` 对应于 `num_fft_points / 2`)的浮点值张量。
  * `fbank`：音频文件转换为 FBANK 特征（也称为对数 Mel-滤波器组值）。FBANK 特征是根据[HTK Book](http://www.inf.u-szeged.hu/~tothl/speech/htkbook.pdf)中的定义实现的：Raw Signal -> Preemphasis -> DC mean removal -> stft magnitude -> Power spectrum: stft^2 -> mel-filter bank values: triangular filters equally spaced on a Mel-scale are applied -> log-compression: log()（不了解，故不翻译——译者注）。总的来说，音频文件被转换成一个大小为 `N x L x W` 的浮点值张量，其中 `N，L` 等于 `stft` 中的值，而 `W` 等于 `num_filter_bands`。
  * `stft_phase`：每个 stft-bin 的相位信息被附加到 `stft` 量级（存疑——译者注），以便音频文件被转换成大小为 `N x L x 2W` 的浮点值张量，其中 `N，L，W` 等于 `stft` 中的值。
  * `group_delay`：本文根据此[论文](https://www.ias.ac.in/article/fullyext/sadh/036/05/0745-0782)中的方程式(23)将音频转换为群延迟特征。群延迟特征具有与 `stft` 相同的张量大小。

编码器和[序列输入特征和编码器](#序列输入特征和编码器)是相同的。唯一的区别在于时间序列特征在开始时没有嵌入向量层，所以 `b x s` 的占位符(`b` 是批次大小，`s`是序列长度)直接映射到一个 `b x s x w` (其中 `w` 是如上所述的 `W`)张量，然后传递给不同的序列编码器。

#### 音频输出特征和解码器
目前没有音频解码器(WIP) ，所以音频不能用作输出特征。

#### 音频特征度量
由于目前没有音频解码器可用，所以也没有音频度量。

### Image 特征<a id='Image_特征'></a>
#### 图像特征预处理
Ludwig 支持灰度和彩色图像。通道的数量是推断出来的，但要确保您所有的图像都有相同的通道数量。在预处理过程中，原始图像文件被转换成 numpy ndarray 并保存为 hdf5 格式。数据集中的所有图像应该具有相同的大小。如果它们有不同的大小，则必须在特征预处理参数中指定一个 `resize_method`，以及目标 `width` 和 `height`。

  * `missing_value_strategy` (默认值 `backfill`)：当图像列中缺少一个值时，应该遵循什么策略。该值应该是 `fill_with_const` (用 `fill_value` 参数指定一个特定的值替换缺失的值)，`fill_with_mode` (用列中最常见的值替换缺失的值), `fill_with_mean` (用列中的均值替换缺失的值)，`backfill` (用下一个有效值替换缺失的值)。
  * `in_memory` (默认值 `true`)：定义图像数据集在训练过程中是驻留在内存中还是从磁盘动态获取(对于大数据集很有用)。在后一种情况下，每次训练迭代都会从磁盘中获取一批输入图像的训练。
  * `num_processes` (默认值 `1`)：指定要为预处理图像运行的进程数。
  * `resize_method` (默认值 `crop_or_pad`)：可用选项：`crop_or_pad`——将大于指定的 `width` 和 `height` 的图像裁剪到所需大小，或使用“边填充”填充较小的图像；`interpolate`——使用插值将图像调整到指定的 `width` 和 `height`。
  * `height` (默认值 `null`)：如果需要调整大小，则必须设置图像高度(以像素为单位)。
  * `width` (默认值 `null`)：如果需要调整大小，则必须设置图像宽度(以像素为单位)。
  * `num_channels` (默认值 `null`)：图像中的通道数量。默认情况下，如果值为 `null`，将使用数据集的第一张图像的通道数，如果数据集中有一张图像具有不同的通道数，将报告一个错误。如果指定的值不为 `null`，则数据集中的图像将适应指定的大小。如果该值为 `1`，所有有多个通道的图像都将被灰度化并缩小为一个通道(透明度将丢失)。当值为 `3` 时，1 通道的所有图像将重复 3 次以获得 3 通道，4 通道的图像将失去透明通道。如果值为 `4`，那么所有小于 4 个通道的图像都将剩下的通道填充为 0。
  * `scaling` (默认值 `pixel_normalization`)：在图像上执行哪种缩放。默认情况下，执行 `pixel_normalization`，将每个像素值除以 255，但也可以使用 `pixel_standardization`，即使用[TensorFlow's per image standardization](https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization)。

根据不同的应用，最好不要超过 `256 x 256` 的大小，因为在大多数情况下，较大的大小在性能方面不会提供太多的优势，同时它们会大大减缓训练和推断，并且使向前和向后传递消耗更多的内存，导致在内存有限或者是显卡内存有限的机器内存溢出。

预处理规范示例：

```yaml
name: image_feature_name
type: image
preprocessing:
  height: 128
  width: 128
  resize_method: interpolate
  scaling: pixel_normalization
```

#### 图像输入特征和编码器
输入图像特征转换为大小 `N x H x W x C`(其中 `N` 是数据集的大小，`H x W` 是一个可以调整的指定图片大小，`C` 是通道的数量)的浮点值张量，并通过数据集中的列名做为键名添加到 HDF5 中。列名被添加到 JSON 文件中，并带有一个相关联的字典，其中包含关于调整大小的预处理信息。

目前有两种支持图像的编码器：卷积堆叠编码器和 ResNet 编码器，可以通过设置 `encoder` 参数为 `stacked_cnn` 或在配置输入特征字典中设置为 `resnet`(`stacked_cnn` 是默认的)。


##### 卷积堆叠编码器
卷积堆叠编码器采用以下可选参数：

  * `conv_layers` (默认值 `null`)：它是一个字典列表，包含所有卷积层的参数。列表的长度决定堆叠卷积层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是:`filter_size`, `num_filters`, `pool_size`, `norm`, `activation` 和 `regularize`。如果字典中缺少这些值中的任何一个，则将使用作为编码器参数指定的默认值。如果 `conv_layers` 和 `num_conv_layers` 都为 `null`，则会给 `conv_layers` 赋一个默认列表，值为 `[{filter_size: 7, pool_size: 3, regularize: false}, {filter_size: 7, pool_size: 3, regularize: false}, {filter_size: 3, pool_size: null, regularize: false}, {filter_size: 3, pool_size: null, regularize: false}, {filter_size: 3, pool_size: null, regularize: true}, {filter_size: 3, pool_size: 3, regularize: true}]`。
  * `num_conv_layers` (默认值 `null`)：如果 `conv_layers` 是 `null`，这是堆叠卷积层的数量。
  * `filter_size` (默认值 `3`)：如果在 `conv_layers` 中还没有指定 `filter_size`，这将是每个层使用的默认 `filter_size`。它表示一维卷积滤波器的宽度。
  * `num_filters` (默认值 `256`)：如果 `num_filters` 还没有在 `conv_layers` 中指定，这是默认的将用于每个层的 `num_filters`。它表示滤波器的数量，进而表示二维卷积的输出通道。
  * `strides` (默认值 `(1, 1)`)：指定卷积沿高度和宽度的步长。
  * `padding` (默认值 `valid`)：`valid` 或 `same`。
  * `dilation_rate` (默认值 `(1, 1)`)：用于扩张卷积的扩张率。
  * `conv_use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `conv_weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `conv_bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `conv_bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `conv_activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `conv_norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `conv_norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `conv_activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `conv_dropout` (默认值 `0`)：Dropout 比率。
  * `pool_function` (默认值 `max`)：池化函数——`max` 为最大值。`average `， `avg` 或 `mean` 都将计算平均值。
  * `pool_size` (默认值 `(2, 2)`)：如果在 `conv_layers` 中尚未指定 `pool_size`，则将为每个层使用默认的 `pool_size`。它表示卷积操作后沿 `s` 序列维执行的最大池化的大小。
  * `pool_strides` (默认值 `null`)：缩小因子。
  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。如果 `fc_layers` 和 `num_fc_layers` 都为 `null`，则会给 `fc_layers` 分配一个值为 `[{fc_size: 512}， {fc_size: 256}]` 的默认列表(仅适用于 `reduce_output` 不为 `null` 的情况)。
  * `num_fc_layers` (默认值 `1`)：这是堆叠的完全连接层的数量。
  * `fc_size` (默认值 `256`)：`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `fc_use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `fc_weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `fc_bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `fc_weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `fc_bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `fc_activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `fc_norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `fc_norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `fc_activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `fc_dropout` (默认值 `0`)：Dropout 比率。

在图像特征输入列表中使用卷积堆叠编码器示例：

```yaml
name: image_column_name
type: image
encoder: stacked_cnn
tied_weights: null
conv_layers: null
num_conv_layers: null
filter_size: 3
num_filters: 256
strides: (1, 1)
padding: valid
dilation_rate: (1, 1)
conv_use_bias: true
conv_weights_initializer: glorot_uniform
conv_bias_initializer: zeros
weights_regularizer: null
conv_bias_regularizer: null
conv_activity_regularizer: null
conv_norm: null
conv_norm_params: null
conv_activation: relu
conv_dropout: 0
pool_function: max
pool_size: (2, 2)
pool_strides: null
fc_layers: null
num_fc_layers: 1
fc_size: 256
fc_use_bias: true
fc_weights_initializer: glorot_uniform
fc_bias_initializer: zeros
fc_weights_regularizer: null
fc_bias_regularizer: null
fc_activity_regularizer: null
fc_norm: null
fc_norm_params: null
fc_activation: relu
fc_dropout: 0
preprocessing:  # example pre-processing
    height: 28
    width: 28
    num_channels: 1
```

##### RESNET 编码器
ResNet 编码器接受以下可选参数：

  * `resnet_size` (默认值 `50`)：ResNet 模型的大小，必须是以下值之一：`8`, `14`, `18`, `34`, `50`, `101`, `152`, `200`。
  * `num_filters` (默认值 `16`)：它表示滤波器的数量，进而表示二维卷积的输出通道。
  * `kernel_size` (默认值 `3`)：用于卷积核的大小。
  * `conv_stride` (默认值 `1`)：初始卷积层的步长。
  * `first_pool_size` (默认值 `null`)：用于第一个池化层的池大小。如果没有，则跳过第一个池化层。
  * `batch_norm_momentum` (默认值 `0.9`)：批标准化的 `momentum`。[TensorFlow's implementation](https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py#L36)中的建议参数是 `0.997`，但这导致训练时和测试时的标准化之间存在很大差异，因此默认值是更保守的 `0.9`。
  * `batch_norm_epsilon` (默认值 `0.001`)：批标准化的 `epsilon`。[TensorFlow's implementation](https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py#L37)中建议的参数是 `1e-5`，但这导致训练时和测试时的标准化之间存在很大差异，所以默认值是比较保守的 `0.001`。
  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。如果 `fc_layers` 和 `num_fc_layers` 都为 `null`，则会给 `fc_layers` 分配一个值为 `[{fc_size: 512}， {fc_size: 256}]` 的默认列表(仅适用于 `reduce_output` 不为 `null` 的情况)。
  * `num_fc_layers` (默认值 `1`)：这是堆叠的完全连接层的数量。
  * `fc_size` (默认值 `256`)：如果`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout` (默认值 `0`)：Dropout 比率。

在图像特征输入列表中使用 ResNet 编码器示例：

```yaml
name: image_column_name
type: image
encoder: resnet
tied_weights: null
resnet_size: 50
num_filters: 16
kernel_size: 3
conv_stride: 1
first_pool_size: null
batch_norm_momentum: 0.9
batch_norm_epsilon: 0.001
fc_layers: null
num_fc_layers: 1
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
preprocessing:
    height: 224
    width: 224
    num_channels: 3
```

#### 图像输出特征和解码器
目前还没有图像解码器(WIP) ，因此图像不能用作输出特征。

#### 图像特征度量
由于目前没有图像解码器可用，所以也没有图像度量。

### Date 特征<a id='Date_特征'></a>
#### 日期特征预处理
Ludwig 将尝试自动推断日期格式，但可以提供一个具体的格式。该格式与[日期时间包文档](https://docs.python.org/2/library/time.html#time.strptime)的描述相同。

  * `missing_value_strategy` (默认值 `fill_with_const`)：当图像列中缺少一个值时，应该遵循什么策略。该值应该是 `fill_with_const` (用 `fill_value` 参数指定一个特定的值替换缺失的值)，`fill_with_mode` (用列中最常见的值替换缺失的值), `fill_with_mean` (用列中的均值替换缺失的值)，`backfill` (用下一个有效值替换缺失的值)。
  * `fill_value` (默认值 `""`)：在 `missing_value_strategy` 为 `fill_value` 的情况下替换缺失的值。可以是一个日期时间字符串，如果为空，则使用当前日期时间。
  * `datetime_format` (默认值 `null`)：这个参数可以是 `null`，表示日期时间格式是自动推断的，也可以是日期时间格式字符串。

预处理规范示例：

```yaml
name: date_feature_name
type: date
preprocessing:
  missing_value_strategy: fill_with_const
  fill_value: ''
  datetime_format: "%d %b %Y"
```

#### 日期输入特征和编码器
输入日期特征被转换成大小为 `N x 8` 的整数值张量(其中 `N` 是数据集的大小，8 个维度包含年、月、日、周、yearday、时、分和秒)，并通过数据集中的列名做为键名添加到 HDF5 中。

目前有两种编码器支持日期：Embed 编码器和 Wave 编码器，它们可以在配置的输入特征字典中设置 `encoder` 参数为 `embed` 或 `wave` 来指定(`embed` 是默认的)。

##### EMBED 编码器
这个编码器通过一个神经元的完全连接层传递年份，并嵌入日期的所有其他元素，将它们连接起来，并通过完全连接层传递连接的表示。它采用以下可选参数：

  * `embedding_size` (默认值 `10`)：这是所采用的最大嵌入向量尺寸。
  * `embeddings_on_cpu` (默认值 `false`)：默认情况下，如果使用 GPU，嵌入向量存储在 GPU 内存中，因为它允许更快的访问，但在某些情况下，嵌入向量可能非常大，此参数强制将嵌入向量放置在常规内存中，并使用 CPU 来解析它们，由于 CPU 和 GPU 内存之间的数据传输，进程稍微减慢。
  * `dropout` (默认值 `false`)：确定在返回编码器输出之前是否应该有一个 dropout 层。
  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。如果 `fc_layers` 和 `num_fc_layers` 都为 `null`，则会给 `fc_layers` 分配一个值为 `[{fc_size: 512}， {fc_size: 256}]` 的默认列表(仅适用于 `reduce_output` 不为 `null` 的情况)。
  * `num_fc_layers` (默认值 `0`)：这是堆叠的完全连接层的数量。
  * `fc_size` (默认值 `10`)：如果`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout` (默认值 `0`)：Dropout 比率。

在输入特征列表中输入日期特征时使用 embed 编码器的例子：

```yaml
name: date_column_name
type: date
encoder: embed
embedding_size: 10
embeddings_on_cpu: false
dropout: false
fc_layers: null
num_fc_layers: 0
fc_size: 10
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
```

##### WAVE 编码器
该编码器通过一个神经元的完全连接层传递年份，并通过取其正弦值与不同周期（12个月，31天等）来表示日期的所有其他元素，将它们串联起来，并通过完全连接层传递串联表示。它采用以下可选参数：

  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。如果 `fc_layers` 和 `num_fc_layers` 都为 `null`，则会给 `fc_layers` 分配一个值为 `[{fc_size: 512}， {fc_size: 256}]` 的默认列表(仅适用于 `reduce_output` 不为 `null` 的情况)。
  * `num_fc_layers` (默认值 `0`)：这是堆叠的完全连接层的数量。
  * `fc_size` (默认值 `10`)：如果`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout` (默认值 `0`)：Dropout 比率。

在输入特征列表中输入日期特征时使用 wave 编码器的例子：

```yaml
name: date_column_name
type: date
encoder: wave
fc_layers: null
num_fc_layers: 0
fc_size: 10
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
```

#### 日期输出特征和解码器
目前没有日期解码器(WIP) ，所以日期不能用作输出特征。

#### 日期特征度量
由于目前没有日期解码器可用，所以也没有日期度量。

### H3 特征<a id='H3_特征'></a>
H3 是一个表示地理空间数据的索引系统。有关详细信息，请参阅：[https://eng.uber.com/h3/](https://eng.uber.com/h3/)。

#### H3 特征预处理
Ludwig 将自动解析 H3 64位 编码格式。预处理参数为：

  * `missing_value_strategy` (默认值 `fill_with_const`)：当 H3 列中缺少一个值时，应该遵循什么策略。该值应该是 `fill_with_const` (用 `fill_value` 参数指定一个特定的值替换缺失的值)，`fill_with_mode` (用列中最常见的值替换缺失的值), `fill_with_mean` (用列中的均值替换缺失的值)，`backfill` (用下一个有效值替换缺失的值)。
  * `fill_value` (默认值 `576495936675512319`)：在 `missing_value_strategy` 为 `fill_value` 的情况下替换缺失的值。这是一个与 H3 特征值兼容的 64位 整数。默认的值编码模式为 `1`，边为 `0`，分辨率为 `0`，基本单元格为 `0`。

预处理规范示例：

```yaml
name: h3_feature_name
type: h3
preprocessing:
  missing_value_strategy: fill_with_const
  fill_value: 576495936675512319
```

#### H3 输入特征和编码器
目前有三种编码器支持 H3：Embed 编码器、Weighted Sum Embed 编码器和 RNN 编码器，它们可以在配置的输入特征字典中设置 `encoder` 参数为 `embed` 或 `weighted_sum` 或 `rnn` 来指定(`embed` 是默认的)。

##### EMBED 编码器
该编码器对 H3 表示的每个组件(模式、边、分辨率、基本单元格和子单元格)进行嵌入编码。值为 `0` 的子单元格将被掩盖。在嵌入之后，所有的嵌入都被求和并可选地通过一个堆叠的完全连接层。它接受以下可选参数：

  * `embedding_size`  (默认值 `10`)：这是所采用的最大嵌入向量尺寸。
  * `embeddings_on_cpu`  (默认值 `false`)：默认情况下，如果使用 GPU，嵌入向量存储在 GPU 内存中，因为它允许更快的访问，但在某些情况下，嵌入向量可能非常大，此参数强制将嵌入向量放置在常规内存中，并使用 CPU 来解析它们，由于 CPU 和 GPU 内存之间的数据传输，进程稍微减慢。
  * `fc_layers`  (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。
  * `num_fc_layers`  (默认值 `0`)：这是堆叠的完全连接层的数量。
  * `fc_size`  (默认值 `10`)：如果`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias`  (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer`  (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer`  (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer`  (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer`  (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer`  (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm`  (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params`  (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation`  (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout`  (默认值 `0`)：Dropout 比率。

在输入特征列表中输入 H3 特征时使用 embed 编码器的例子：

```yaml
name: h3_column_name
type: h3
encoder: embed
embedding_size: 10
embeddings_on_cpu: false
fc_layers: null
num_fc_layers: 0
fc_size: 10
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
```

##### WEIGHTED SUM EMBED 编码器
该编码器对 H3 表示的每个组件(模式、边、分辨率、基本单元格和子单元格)进行嵌入编码。值为 `0` 的子单元格将被掩盖。在嵌入之后，所有的嵌入都以加权和(带有学习权值)求和并可选地通过一个堆叠的完全连接层。它接受以下可选参数：

  * `embedding_size` (默认值 `10`)：这是所采用的最大嵌入向量尺寸。
  * `embeddings_on_cpu` (默认值 `false`)：默认情况下，如果使用 GPU，嵌入向量存储在 GPU 内存中，因为它允许更快的访问，但在某些情况下，嵌入向量可能非常大，此参数强制将嵌入向量放置在常规内存中，并使用 CPU 来解析它们，由于 CPU 和 GPU 内存之间的数据传输，进程稍微减慢。
  * `should_softmax` (默认值 `false`)：确定加权和的权重是否应该在使用前通过 softmax 层。
  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。
  * `num_fc_layers` (默认值 `0`)：这是堆叠的完全连接层的数量。
  * `fc_size` (默认值 `10`)：如果`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout` (默认值 `0`)：Dropout 比率。
  * `reduce_output` (默认值 `sum`)：定义如果张量的秩大于 2，如何沿 `s` 序列长度维减少输出张量。可用的值有: `sum`， `mean` 或 `avg`， `max`， `concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)和 `null`(不减少并返回整个张量)。

在输入特征列表中输入 H3 特征时使用 weighted sum embed 编码器的例子：

```yaml
name: h3_column_name
type: h3
encoder: weighted_sum
embedding_size: 10
embeddings_on_cpu: false
should_softmax: false
fc_layers: null
num_fc_layers: 0
fc_size: 10
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
reduce_output: sum
```

##### RNN 编码器
该编码器对 H3 表示的每个组件(模式、边、分辨率、基本单元格和子单元格)进行嵌入编码。值为 `0` 的子单元格将被掩盖。在嵌入之后，所有嵌入都通过 RNN 编码器。这背后的直觉是，从基本单元格开始，子单元格的序列可以看作是一个编码所有 H3 六元树中路径的序列，因此编码采用递归模型。它采用以下可选参数：

  * `embedding_size` (默认值 `10`)：这是所采用的最大嵌入向量尺寸。
  * `embeddings_on_cpu` (默认值 `false`)：默认情况下，如果使用 GPU，嵌入向量存储在 GPU 内存中，因为它允许更快的访问，但在某些情况下，嵌入向量可能非常大，此参数强制将嵌入向量放置在常规内存中，并使用 CPU 来解析它们，由于 CPU 和 GPU 内存之间的数据传输，进程稍微减慢。
  * `num_layers` (默认值 `1`)：堆叠的循环层数。
  * `state_size` (默认值 `256`)：rnn 状态的大小。
  * `cell_type` (默认值 `rnn`)：要使用的循环单元的类型。 可用值包括： `rnn`, `lstm`, `lstm_block`, `lstm`, `ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`。 有关单元之间差异的参考，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell)。 我们建议在 CPU 上使用 `block` 变体，在 GPU 上使用 `cudnn` 变体，因为它们提高了速度。
  * `bidirectional` (默认值 `false`)：如果为 `true`，则两个循环网络将在前向和后向进行编码，并将它们的输出连接起来。
  * `activation` (默认值 `'tanh'`)：使用的激活函数。
  * `recurrent_activation` (默认值 `sigmoid`)：在循环步骤中使用的激活函数。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `unit_forget_bias` (默认值 `true`)：如果为 `true`，在初始化时给遗忘门的偏差加 1。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `recurrent_initializer` (默认值 `orthogonal`)：初始化设定循环矩阵权重。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `recurrent_regularizer` (默认值 `null`)：正则化函数应用于循环矩阵权值。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `dropout` (默认值 `0.0`)：Dropout 比率。
  * `recurrent_dropout` (默认值 `0.0`)：循环状态的 dropout 比率。
  * `initializer` (默认值 `null`)：要使用的初始值设定项。如果为 `null`，则使用每个变量的默认值（在大多数情况下为 `glorot_uniform`）。选项有：`constant`、`identity`、`zeros`、`ones`、`orthogonal`、`normal`、`uniform`、`truncated_normal`、`variance_scaling`、`glorot_normal`、`glorot_uniform`、`xavier_normal`、`xavier_uniform`、`he_normal`、`he_uniform`、`lecun_normal`、`lecun_uniform`。或者，可以使用一个指定的包含 `type` 键(标识初始值设定项的类型)以及其它键和参数的字典，例如 `{type:normal，mean:0，stddev:0}`。要了解每个初始值设定项的参数，请参阅[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `regularize` (默认值 `true`)：如果 `true`，则将嵌入向量权重添加到通过正则化损失正则化的权重集（如果 `training` 中的 `regularization_lambda`大于 `0`）。
  * `reduce_output` (默认值 `last`)：定义如果张量的秩大于 2，如何沿 `s` 序列长度维减少输出张量。可用的值有: `sum`， `mean` 或 `avg`， `max`， `concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)和 `null`(不减少并返回整个张量)。

在输入特征列表中输入 H3 特征时使用 rnn 编码器的例子：

```yaml
name: h3_column_name
type: h3
encoder: rnn
embedding_size: 10
embeddings_on_cpu: false
num_layers: 1
cell_type: rnn
state_size: 10
bidirectional: false
activation: tanh
recurrent_activation: sigmoid
use_bias: true
unit_forget_bias: true
weights_initializer: glorot_uniform
recurrent_initializer: orthogonal
bias_initializer: zeros
weights_regularizer: null
recurrent_regularizer: null
bias_regularizer: null
activity_regularizer: null
dropout: 0.0
recurrent_dropout: 0.0
initializer: null
regularize: true
reduce_output: last
```

#### H3 输出特征和解码器
目前没有 H3 解码器(WIP) ，所以 H3 不能用作输出特征。

#### H3 特征度量
由于目前没有 H3 解码器可用，所以也没有 H3 度量。

### Vector 特征<a id='Vector_特征'></a>
向量特征允许一次提供一组有序的数值。这对于提供从其他模型获得的预先训练过的表示或激活，或者提供多变量输入和输出都很有用。向量特征的一个有趣的用途是提供一个概率分布作为多类分类问题的输出，而不是像分类特征那样只提供正确的类。这对于蒸馏和噪声感知损失是有用的。

#### 向量特征预处理
数据应为空格分隔的数值。如：“1.0 0.0 1.04 10.49”。所有向量的大小都应该相同。

预处理参数：

  * `vector_size` (默认值 `null`)：向量的大小。如果没有提供，它将从数据推断。
  * `missing_value_strategy` (默认值 `fill_with_const`)：当向量列中缺少一个值时，应该遵循什么策略。该值应该是 `fill_with_const` (用 `fill_value` 参数指定一个特定的值替换缺失的值)，`fill_with_mode` (用列中最常见的值替换缺失的值), `fill_with_mean` (用列中的均值替换缺失的值)，`backfill` (用下一个有效值替换缺失的值)。
  * `fill_value` (默认值 `""`)：在 `missing_value_strategy` 为 `fill_value` 的情况下替换缺失的值。

#### 向量特征编码器
向量特征支持两种编码器：`dense` 和 `passthrough`。只有 `dense` 编码器有额外的参数，如下所示。

##### DENSE 编码器
对于向量特征，您可以使用 dense 编码器(堆叠的完全连接层)。它的参数如下：

  * `layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。如果 `fc_layers` 和 `num_fc_layers` 都为 `null`，则会给 `fc_layers` 分配一个值为 `[{fc_size: 512}， {fc_size: 256}]` 的默认列表(仅适用于 `reduce_output` 不为 `null` 的情况)。
  * `num_layers` (默认值 `0`)：堆叠的完全连接层数量。
  * `fc_size` (默认值 `256`)：如果`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout` (默认值 `0`)：Dropout 比率。

在输入特征列表中输入向量特征时使用 dense 编码器的例子：

```yaml
name: vector_column_name
type: vector
encoder: dense
layers: null
num_layers: 0
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
```

#### 向量特征解码器
当需要在噪声感知损失的情况下执行多类分类或任务是多元回归时，可以使用向量特征。对于向量特征，只有一个解码器可用，它是一个(可能是空的)堆叠的完全连接层，然后被投影到一个向量(在多类分类情况下，可在后面可选的跟一个softmax)。

```
+--------------+   +---------+   +-----------+
|Combiner      |   |Fully    |   |Projection |   +------------------+
|Output        +--->Connected+--->into Output+--->Softmax (optional)|
|Representation|   |Layers   |   |Space      |   +------------------+
+--------------+   +---------+   +-----------+
```

这些是设置输出特征的可用参数：

  * `reduce_input` (默认值 `sum`)：定义如何在第一维(如果算上批次维数，则是第二维)上减少不是向量而是矩阵或更高阶张量的输入。可用的值有：`sum`，`mean` 或 `avg`，`max`，`concat` (沿第一个维度连接)，`last` (返回第一个维度的最后一个向量)。
  * `dependencies` (默认值 `[]`)：它所依赖的输出特征。有关详细解释，请参阅[输出特征依赖关系](https://ludwig-ai.github.io/ludwig-docs/user_guide/#output-features-dependencies)。
  * `reduce_dependencies` (默认值 `sum`)：定义如何在第一维(如果算上批次维度，则是第二维)上减少一个依赖特征(不是向量，而是一个矩阵或更高阶张量)的输出。可用的值有：`sum`，`mean` 或 `avg`，`max`，`concat`(沿第一个维度连接)，`last`(返回第一个维度的最后一个向量)。
  * `softmax` (默认值 `false`)：确定是否在解码器的末尾应用 softmax。它对于预测总和为 `1` 且可以解释为概率的值的向量很有用。
  * `loss` (默认值 `{type: mean_squared_error}`)：是一个包含损失 `type` 的字典。可用的损失 `type` 有 `mean_squared_error`，`mean_absolute_error` 和 `softmax_cross_entropy`(只有当 `softmax` 为 `true` 时才使用)。

这些是一组输出特征解码器的可用参数：

  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。
  * `num_fc_layers` (默认值 `0`)：这是堆叠的完全连接层的数量。
  * `fc_size` (默认值 `256`)：如果`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `clip` (默认值 `null`)：如果不是 `null`，则指定预测将被剪切到的最小值和最大值。该值可以是一个列表或长度为 `2` 的元组，第一个值表示最小值，第二个值表示最大值。例如，`(-5,5)` 将使所有的预测都在 `[-5,5]` 区间内剪切。

输出特征列表中的向量特征(具有默认参数)示例：

```yaml
name: vector_column_name
type: vector
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: sigmoid_cross_entropy
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
activation: relu
clip: null
```

#### 向量特征度量
每个周期计算出来的可用于向量特征的度量是 `mean_squared_error`， `mean_absolute_error`， `r2` 和 `loss` 本身。如果您将 `validation_field` 设置为向量特征的名称，那么您可以在配置 `training` 部分将它们中的任何一个设置为 `validation_measure`。

### 组合器<a id='组合器'></a>
组合器是模型的一部分，它将编码器的所有输入特征的输出进行组合，然后将组合的表示提供给不同的输出解码器。如果你没有指定一个组合器，将使用 `concat` 组合器。

#### Concat 组合器
Concat 组合器假设编码器的所有输出都是尺寸为 `b x h` 的张量，其中 `b` 是批次大小，`h` 是隐藏维，对于每个输入，它可以是不同的。它沿着 `h` 维进行连接，然后(可选地)将连接张量通过一个堆叠的完全连接层。它返回最终的 `b x h` 张量，其中 `h'` 是最后一个完全连接层的大小，或者是在没有完全连接的层的情况下，所有输入 `h` 的大小总和。如果只有一个输入特征，并且没有指定完全连接层，则输入特征的输出将作为输出直接传递。

```
+-----------+
|Input      |
|Feature 1  +-+
+-----------+ |            +---------+
+-----------+ | +------+   |Fully    |
|...        +--->Concat+--->Connected+->
+-----------+ | +------+   |Layers   |
+-----------+ |            +---------+
|Input      +-+
|Feature N  |
+-----------+
```

这些是一个 concat 组合器的可用参数：

  * `fc_layers` (默认值 `null`)：它是一个字典列表，包含所有完全连接层的参数。列表的长度决定堆叠的完全连接层的数量，每个字典的内容决定特定层的参数。每个层的可用参数是：`fc_size`，`norm`，`activation`，和 `regulalize`。如果字典中缺少这些值中的任何一个，则将使用作为解码器参数指定的默认值。
  * `num_fc_layers` (默认值 `0`)：这是堆叠的完全连接层的数量。
  * `fc_size` (默认值 `256`)：如果`fc_size` 没有在 `fc_layers` 中指定，这是默认的 `fc_size`，将用于每个层。它表示一个完全连接层的输出大小。
  * `use_bias` (默认值 `true`)：布尔值，层是否使用偏置向量。
  * `weights_initializer` (默认值 `glorot_uniform`)：权重矩阵的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `bias_initializer` (默认值 `zeros`)：偏置向量的初始化设定项。选项是——`constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`。另外，也可以指定一个具有 `type` 键(标识初始化设定项的类型)及其它键值对的字典，例如 `{type: normal, mean: 0, stddev: 0}`。要了解每个初始化设定项的参数，请参考[TensorFlow文档](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)。
  * `weights_regularizer` (默认值 `null`)：应用于权重矩阵的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `bias_regularizer` (默认值 `null`)：应用于偏置向量的正则化函数。有效值为 `l1`、`l2` 或 `l1_l2`。
  * `activity_regularizer` (默认值 `null`)：应用于输出层的正则化函数。有效值为 `l1`、`l2`或 `l1_l2`。
  * `norm` (默认值 `null`)：如果 `norm` 没有在 `fc_layers` 中指定，默认的 `norm`将用于每个层。它表明输出的标准化，它可以是 `null`，`batch` 或 `layer`。
  * `norm_params` (默认值 `null`)：`norm` 为 `batch` 或 `layer` 时使用的参数。有关与 `batch` 一起使用的参数的信息，请参阅 [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)；有关 `layer`，请参阅[Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)。
  * `activation` (默认值 `relu`)：如果 `fc_layers` 中没有指定 `activation`，这是默认的 `activation`，将用于每个层。它表明应用于输出的激活函数。
  * `dropout` (默认值 `0`)：Dropout 比率。

配置中的 concat 组合器示例：

```yaml
type: concat
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: true
weights_initializer: 'glorot_uniform'
bias_initializer: 'zeros'
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
```

#### Sequence Concat 组合器<a id='Sequence_Concat_组合器'></a>
Sequence Concat  组合器假设编码器至少有一个输出是大小为 `b x s x h` 的张量，其中 `b` 是批次大小，`s` 是序列的长度，`h` 是隐藏维数。sequence / text / sequential 输入可以使用 `main_sequence_feature` 参数指定，该参数应该以 sequential 特征的名称作为值。如果没有指定 `main_sequence_feature`，那么组合器将按照配置中定义的顺序查找所有特征，并寻找一个具有 3 阶张量输出(序列、文本或时间序列)的特征。如果没有找到，它将引发一个异常，否则该特征的输出将被用于连接序列 `s` 维上的其他特征。

如果有其它 3 阶输出张量的输入特征，则组合器将它们与 `s` 维一起连接，这意味着所有这些特征必须具有相同的 `s` 维，否则将引发错误。具体来说，由于序列特征的占位符的尺寸为 `[None，None]` ，为了使 `BucketedBatcher` 将更长的序列修剪到其实际长度，不能在建模时检查序列是否具有相同的长度，当一个数据点具有两个不同长度的序列特征时，训练过程中会返回一个维度不匹配错误。

其他具有 `b x h` 2 阶张量输出的特征将被复制 `s` 次并连接到 `s` 维上。最终的输出是 `b x s x h'` 张量，其中 `h'` 是所有输入特征的 `h` 维的拼接的大小。

```
Sequence
Feature
Output

+---------+
|emb seq 1|
+---------+
|...      +--+
+---------+  |  +-----------------+
|emb seq n|  |  |emb seq 1|emb oth|   +------+
+---------+  |  +-----------------+   |      |
             +-->...      |...    +-->+Reduce+->
Other        |  +-----------------+   |      |
Feature      |  |emb seq n|emb oth|   +------+
Output       |  +-----------------+
             |
+-------+    |
|emb oth+----+
+-------+
```

这些是 sequence concat 组合器的可用参数：

  * `main_sequence_feature` (默认值 `null`)：要将其他特征的输出连接到的序列/文本/时间序列特征的名称。如果没有指定 `main_sequence_feature`，那么组合器将按照配置中定义的顺序查找所有特征，并寻找一个具有 3 阶张量输出(序列、文本或时间序列)的特性。如果没有找到，它将引发一个异常，否则该特征的输出将被用于连接序列 `s` 维上的其他特征。如果有其他 3 阶输出张量的输入特征，那么组合器将把它们与 `s` 维连接起来，这意味着它们都必须具有相同的 `s` 维，否则将抛出一个错误。
  * `reduce_output` (默认值 `null`)：描述用于聚合集合项的嵌入向量的策略。可能的值有 `null`、`sum`、`mean` 和 `sqrt`（加权和除以权重平方和的平方根）。

配置中的 sequence concat 组合器示例：

```yaml
type: sequence_concat
main_sequence_feature: null
reduce_output: null
```

#### Sequence 组合器
序列组合器将 sequence concat 组合器与序列编码器堆叠在一起。所有关于 [Sequence Concat 组合器](#Sequence_Concat_组合器) 的输入张量的阶描述的考虑也适用于本例，但主要区别在于，此组合器使用 sequence concat 组合器的 `b x s x h'` 输出，其中 `b` 是批次大小，`s` 是序列长度，`h'` 是所有输入特征的隐藏维数之和，作为 [序列输入特征和编码器](#序列输入特征和编码器) 中所描述的任何序列编码器的输入。有关序列编码器及其参数的更多详细信息，请参阅该节。此外，所有关于序列编码器输出形状的考虑也适用于这种情况。

```
Sequence
Feature
Output

+---------+
|emb seq 1|
+---------+
|...      +--+
+---------+  |  +-----------------+
|emb seq n|  |  |emb seq 1|emb oth|   +--------+
+---------+  |  +-----------------+   |Sequence|
             +-->...      |...    +-->+Encoder +->
Other        |  +-----------------+   |        |
Feature      |  |emb seq n|emb oth|   +--------+
Output       |  +-----------------+
             |
+-------+    |
|emb oth+----+
+-------+
```

配置中的 Sequence 组合器示例：

```yaml
type: sequence
main_sequence_feature: null
encoder: parallel_cnn
... encoder parameters ...
```

## 分布式训练<a id='分布式训练'></a>
您可以使用 [Horovod](https://github.com/uber/horovod) 来分配模型的训练和预测，它允许在具有多个 GPUs 的单个机器上进行训练，也可以在具有多个 GPUs 的多台机器上进行训练。

为了使用分布式训练，您必须按照 [Horovod 安装说明](https://github.com/uber/horovod#install) 详细安装 Horovod (包括安装 [OpenMPI](https://www.open-mpi.org/) 或其他 [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) 实现或 [Gloo](https://github.com/facebookincubator/gloo))，然后安装两个包：

```shell
pip install horovod mpi4py
```

实际上，Horovod 的工作方式是增加批次大小，并将每个批次的一部分分配到不同的节点，并以智能且可扩展的方式收集所有节点的梯度。 它还会调整学习率，以平衡批次大小的增加。 其优点是训练速度几乎与节点数成线性比例关系。

`experiment`、`train` 和 `predict` 命令接受一个 `—use_horovod`参数，该参数指示以分布式方式使用 Horovod 进行模型构建、训练和预测。在调用 Ludwig 的命令之前，必须提供一个 `horovodrun` 命令，指定要使用哪些机器 和/或 哪些 GPUs，以及更多的参数。例如，为了在一个有四个 GPUs 的本地机器上训练一个 Ludwig 模型，您可以运行：

```shell
horovodrun -np 4 \
    ludwig train --use_horovod ...other Ludwig parameters...
```

而在四个远程机器上，每个机器都有四个 gpu，你可以运行：

```shell
horovodrun -np 16 \
    -H server1:4,server2:4,server3:4,server4:4 \
    ludwig train --use_horovod ...other Ludwig parameters...
```

这同样适用于 `experiment`, `predict` 和 `test`。

关于 Horovod 安装和运行参数的更多细节可以在 [Horovod文档](https://github.com/uber/horovod) 中找到。

## 超参数优化<a id='超参数优化'></a>
为了执行超参数优化，必须在 Ludwig 配置中提供 `hyperopt` 根键。其中包括要优化的度量，要优化的参数，要使用的采样器，以及如何执行优化。

在 `hyperopt` 配置中可以定义的不同参数有：`goal`，它指示是否最小化或最大化某个度量值，或任意数据集分割上的任意输出特征的损失。可用值有——`minimize`(默认值) 或 `maximize`。`output_feature` 是一个 `str`，包含我们想要优化的度量或损失的输出特征名称。可用值是——`combined`(默认值)或配置中提供的任何输出特征的名称。`combined` 是一种特殊的输出特征，允许对所有输出特征的总损失和度量进行优化。`metric` 是我们想要优化的度量。默认值是 `loss`，但根据 `output_feature` 中定义的特征类型，可以使用不同的度量和损失。检查特定输出特征类型的度量部分，找出可用的度量标准。`split` 是我们想要计算度量的数据分割。默认情况下，它是 `validation` 分割，但您也可以灵活地指定 `train` 或 `test` 分割。`parameters` 部分由一组要优化的超参数组成。它们作为键(参数的名称)和与之相关联的值(定义搜索空间)提供。根据超参数的类型不同，其值也不同。类型可以是 `float`， `int` 和 `category`。`sampler` 部分包含用于对超参数值进行采样的采样器类型及其配置。当前可用的采样器类型有 `grid` 和 `random`。采样器配置参数改变采样器行为，例如 `random`，您可以设置多少随机采样来绘制。`executor` 部分指定如何执行超参数优化。其可以在本地以串行方式执行，也可以在多个工作器之间并行进行，如果可以的话，还可以使用 GPUs。

示例：

```yaml
hyperopt:
  goal: minimize
  output_feature: combined
  metric: loss
  split: validation
  parameters:
    utterance.cell_type: ...
    utterance.num_layers: ...
    combiner.num_fc_layers: ...
    section.embedding_size: ...
    preprocessing.text.vocab_size: ...
    training.learning_rate: ...
    training.optimizer.type: ...
    ...
  sampler:
    type: grid  # random, ...
    # sampler parameters...
  executor:
    type: serial  # parallel, ...
    # executor parameters...
```

在 `parameters` 部分中，是用于引用 配置中 嵌套的参数。例如，要引用 `learning_rate`，必须使用 `training.learning_rate`。如果要引用的参数位于输入或输出特征的内部，则该特征的名称将作为起点。例如，要引用 `utterance` 特征的 `cell_type`，请使用名称 `utterance.cell_type`。

### 超参数<a id='超参数'></a>
#### Float 参数
对于浮点值，要指定的参数是：

  * low：参数可以具有的最小值。
  * high：参数可以具有的最大值。
  * scale：`linear`(默认值) 或 `log`。
  * steps：可选步数。

例如 `range: (0.0, 1.0)， steps: 3` 将产生 `[0.0,0.5,1.0]` 作为抽样的潜在值，而如果 `steps` 未指定，则将使用 `0.0` 和 `1.0` 之间的全部值。

示例：

```yaml
training.learning_rate:
  type: real
  low: 0.001
  high: 0.1
  steps: 4
  scale: linear
```

#### Int 参数
对于整数值，要指定的参数是：

  * low：参数可以具有的最小值。
  * high：参数可以具有的最大值。
  * steps：可选步数。

例如，`range: (0, 10)， steps: 3` 将会生成 `[0,5,10]` 以供搜索，而如果没有指定 `steps`，则会使用 `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`。

示例：

```yaml
combiner.num_fc_layers:
  type: int
  low: 1
  high: 4
```

#### Category 参数
对于 `category` 值，要指定的参数是——`values`，一个可能值的列表。列表中每个值的类型并不重要（它们可以是字符串、整数、浮点和其他任何类型，甚至是整个字典）。

示例：

```yaml
utterance.cell_type:
  type: category
  values: [rnn, gru, lstm]
```

### 采样器<a id='采样器'></a>
#### Grid 采样器
`grid` 采样器通过从 `parameters` 部分中提供的超参数的所有可能值的外积中选择所有元素来创建搜索空间。对于 `float` 参数，需要指定 `steps` 的数目。

示例：

```yaml
sampler:
  type: grid
```

#### Random 采样器
`random` 采样器从参数搜索空间中随机采样超参数值。`num_samples`(默认值：`10`)可以在 `sampler` 部分中指定。

示例：

```yaml
sampler:
  type: random
  num_samples: 10
```

#### PySOT 采样器
`pysot` 采样器使用 [pySOT](https://arxiv.org/pdf/1908.00420.pdf) 包进行异步代理优化。这个包实现了许多流行的贝叶斯优化和代理优化方法。默认情况下，pySOT 使用 [Regis and Shoemaker](https://pubsonline.informs.org/doi/10.1287/ijoc.1060.0182) 的随机 RBF(SRBF)方法。SRBF 首先评估一个大小为 `2 * d + 1` 的对称拉丁超立方体设计，其中 `d` 是已优化的超参数数量。当这些点被评估后，SRBF 拟合一个径向基函数代理，并使用这个代理和一个采集函数来选择下一个样本。我们建议使用至少 `10 * d` 的总样本来允许算法收敛。

GitHub 页面提供了更多详细信息：[https://github.com/dme65/pySOT](https://github.com/dme65/pySOT)。

示例：

```yaml
sampler:
  type: pysot
  num_samples: 10
```

### 执行器<a id='执行器'></a>
#### Serial 执行器
`serial` 执行器以序列方式在本地执行超参数优化，每次执行由所选取样器获得一组采样参数中的元素。

示例：

```yaml
executor:
  type: serial
```

#### Parallel 执行器
`parallel` 执行器并行进行超参数优化，同时执行所选采样器获得采样参数集合中的元素。训练和评估模型的并行工作器的最大数量由参数 `num_workers` (默认值：`2`)定义。

在使用 GPUs 进行训练时，提供给命令行接口的 `gpus` 参数包含要使用的 GPUs 列表，如果没有提供 `gpus` 参数，将使用所有可用的 GPUs。也可以提供 `gpu_fraction` 参数，但它会根据 `num_workers` 进行修改，以便并行地执行任务。例如，如果 `num_workers: 4` 和 2 个 GPUs 可用(疑原文有误——译者注)，如果提供的 `gpu_fraction` 大于 `0.5`，它将被 `0.5` 替换。也提供了一个 `epsilon`(默认值：`0.01`)参数来允许使用额外的 GPU 空闲内存——要使用的 GPU 比例定义为 `(# GPU / #workers) - epsilon`。

示例：

```yaml
executor:
  type: parallel
  num_workers: 2
  epsilon: 0.01
```

#### Fiber 执行器
[Fiber](https://github.com/uber/fiber) 是一个用于现代计算机集群的 Python 分布式计算库。`fiber` 执行器在计算机集群上并行执行超参数优化，以实现大规模的并行。查阅[这里](https://uber.github.io/fiber/platforms/)以获得支持的集群类型。

Fiber 执行器需要安装 `fiber`：

```shell
pip install fiber
```

参数：

  * `num_workers`：用于训练和评估模型的并行工作器的数量。缺省值是2。
  * `num_cpus_per_worker`：为每个工作器分配多少个 CPU 核。
  * `num_gpus_per_worker`：为每个工作器分配多少个 GPUs。
  * `fiber_backend`：Fiber 使用的后端。如果要在集群上运行超参数优化，就需要设置这个参数。默认值是 `local`。可用值为 `local`、`kubernetes`、`docker`。关于支持平台的详细信息，请查看 [Fiber 文档](https://uber.github.io/fiber/platforms/)。

示例：

```yaml
executor:
  type: fiber
  num_workers: 10
  fiber_backend: kubernetes
  num_cpus_per_worker: 2
  num_gpus_per_worker: 1
```

**运行 Fiber 执行器：**

Fiber 运行在一个计算机集群上，使用 Docker 封装所有的代码和依赖项。为了运行一个由 Fiber 支持的超参数搜索，您必须创建一个 Docker 文件来封装您的代码和依赖项。

示例：

```docker
FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN apt-get -y update && apt-get -y install git libsndfile1

RUN git clone --depth=1 https://github.com/ludwig-ai/ludwig.git
RUN cd ludwig/ \
    && pip install -r requirements.txt -r requirements_text.txt \
          -r requirements_image.txt -r requirements_audio.txt \
          -r requirements_serve.txt -r requirements_viz.txt \
    && python setup.py install

RUN pip install fiber

RUN mkdir /data
ADD train.csv /data/data.csv
ADD hyperopt.yaml /data/hyperopt.yaml

WORKDIR /data
```

在该 Dockerfile 中，数据 `data.csv` 与指定模型和超参数优化参数的 `hyperopt.yaml` 一起嵌入到 Docker 中。 如果您的数据太大而无法直接添加到 Docker 映像中，请参阅 [Fiber's 文档](https://uber.github.io/fiber/advanced/#working-with-persistent-storage) 以获取有关如何为 Fiber 工作器使用共享持久性存储的说明。 示例 `hyperopt.yaml` 如下所示：

```yaml
input_features:
  -
    name: x
    type: numerical
output_features:
  -
    name: y
    type: category
training:
  epochs: 1
hyperopt:
  sampler:
    type: random
    num_samples: 50
  executor:
    type: fiber
    num_workers: 10
    fiber_backend: kubernetes
    num_cpus_per_worker: 2
    num_gpus_per_worker: 1
  parameters:
    training.learning_rate:
      type: float
      low: 0.0001
      high: 0.1
    y.num_fc_layers:
      type: int
      low: 0
      high: 2
```

使用 Fiber 运行超参数优化与其他执行程序有一点不同，因为涉及到 docker 的构建和推送，因此负责这些方面的 `fiber run` 命令可用于在集群上运行超参数优化：

`fiber run ludwig hyperopt --dataset train.csv -cf hyperopt.yaml`

查看 [Fiber 文档](https://uber.github.io/fiber/getting-started/#running-on-a-computer-cluster) 了解更多关于在集群上运行的细节。

### 完整的超参数优化示例<a id='完整的超参数优化示例'></a>

YAML 示例：

```yaml
input_features:
  -
    name: utterance
    type: text
    encoder: rnn
    cell_type: lstm
    num_layers: 2
  -
    name: section
    type: category
    representation: dense
    embedding_size: 100
combiner:
  type: concat
  num_fc_layers: 1
output_features:
  -
    name: class
    type: category
preprocessing:
  text:
    word_vocab_size: 10000
training:
  learning_rate: 0.001
  optimizer:
    type: adam
hyperopt:
  goal: maximize
  output_feature: class
  metric: accuracy
  split: validation
  parameters:
    training.learning_rate:
      type: float
      low: 0.0001
      high: 0.1
      steps: 4
      scale: log
    training.optimizaer.type:
      type: category
      values: [sgd, adam, adagrad]
    preprocessing.text.word_vocab_size:
      type: int
      low: 700
      high: 1200
      steps: 5
    combiner.num_fc_layers:
      type: int
      low: 1
      high: 5
    utterance.cell_type:
      type: category
      values: [rnn, gru, lstm]
  sampler:
    type: random
    num_samples: 12
  executor:
    type: parallel
    num_workers: 4
```

CLI 示例：

```shell
ludwig hyperopt --dataset reuters-allcats.csv --config "{input_features: [{name: utterance, type: text, encoder: rnn, cell_type: lstm, num_layers: 2}], output_features: [{name: class, type: category}], training: {learning_rate: 0.001}, hyperopt: {goal: maximize, output_feature: class, metric: accuracy, split: validation, parameters: {training.learning_rate: {type: float, low: 0.0001, high: 0.1, steps: 4, scale: log}, utterance.cell_type: {type: category, values: [rnn, gru, lstm]}}, sampler: {type: grid}, executor: {type: serial}}}"
```


## 集成<a id='集成'></a>
Ludwig 提供了一个可扩展的接口来集成第三方系统。要激活特定的集成，只需将其标志插入命令行。每个集成可能有特定的需求和用途。

Ludwig 支持以下集成：

  * `--comet`：记录训练度量、环境信息、测试结果、可视化等数据到 [Comet.ML](https://comet.ml/)。需要一个免费可用的帐户。有关更多详细信息，请参见 [Running Ludwig with Comet](https://www.comet.ml/docs/python-sdk/ludwig/#running-ludwig-with-comet)。
  * `--wandb`：记录训练指标、配置参数、环境信息和训练模型的数据到 [Weights & Biases](https://www.wandb.com/)。有关更多详细信息，请参见 [W&B Quickstart](https://docs.wandb.com/quickstart)。

有关集成的更多信息，请参见 [开发人员指南](https://ludwig-ai.github.io/ludwig-docs/developer_guide/)。

## 编程接口【API】<a id='编程接口【API】'></a>
Ludwig 的功能也可以通过 API 访问。这个 API 由一个 `LudwigModel` 类组成，可以用配置字典初始化，然后用数据集训练(可以在内存中，也可以从文件加载)。预训练的模型可以被加载，并且能够获得对新数据集的预测(可以在内存中，也可以从文件加载)。

[API 文档](https://ludwig-ai.github.io/ludwig-docs/api/) 中提供了 `LudwigModel` 中所有可用函数的详细文档。

### 训练一个模型<a id='训练一个模型'></a>
要训练一个模型，首先要使用 `LudwigModel()` 和配置字典来初始化它，然后使用数据或数据集文件调用 `train()` 函数。

```python
from ludwig.api import LudwigModel

config = {...}
model = LudwigModel(config)
training_statistics, preprocessed_data, output_directory = model.train(dataset=dataset_file_path)
# or
training_statistics, preprocessed_data, output_directory = model.train(dataset=dataframe)
```

`config` 是一个字典，它具有与 YAML 配置文件相同的“键-值”结构，因为它在技术上等同于将 YAML 文件解析到 Python 字典中。请注意，Python 的 `None` 将替换掉 YAML 中的 `null`，同样的 `True/False` 将替换掉 `true/false`。`train_statistics` 是每个输出特征的训练统计字典，包含每个周期的损失和度量值。由 `experiment` 和 `train` 命令生成的 `training_statistics.json` 文件，其内容完全相同。`preprocessed_data` 是包含这三个数据集 `(training_set, validation_set, test_set)` 的元组。`output_directory` 是存储训练结果的文件路径。

### 加载一个预先训练好的模型<a id='加载一个预先训练好的模型'></a>
为了加载一个预先训练好的 Ludwig 模型，您必须调用 `LudwigModel` 类的静态函数 `load()` 来提供包含模型的路径。

```python
from ludwig.api import LudwigModel

model = LudwigModel.load(model_path)
```

### 预测<a id='预测'></a>
无论是一个新训练的模型还是加载一个预训练好的模型，都可以使用模型对象的 `predict()` 函数对新数据进行预测。数据集必须包含与模型的所有输入特征名称相同的列。

```python
predictions, output_directory = model.predict(dataset=dataset_file_path)
#or
predictions, output_directory = model.predict(dataset=dataframe)
```

`predictions` 将是一个包含所有输出特征的预测置信度/概率的数据。 `output_directory` 是预测的临时文件路径。

如果想度量预测的质量，您可以运行：

```python
evaluation_statistics, predictions, output_directory = model.evaluate(dataset=dataset_file_path)
#or
evaluation_statistics, predictions, output_directory = model.evaluate(dataset=dataframe)
```

在这种情况下，数据集还应该包含具有所有输出特征相同名称的列，由于它们的内容将被做为真实值并跟预测值进行比较从而计算度量值，因此 `evaluation_statistics` 将是一个字典，根据每个输出特征的类型包含若干质量度量（例如，`category` 特征将有一个精度度量和误差矩阵，以及其他度量，与之相关，而数字特征将具有均方损失和 R2 等度量）。

## 可视化<a id='可视化'></a>
通过使用 `visualize` 命令，可以从 `train`、`predict` 和 `experiment` 的结果文件中获得一些可视化。该命令有几个参数，但不是所有的可视化都使用这些参数。让我们首先介绍通用脚本的参数，然后，对于每个可用的可视化，我们将讨论所需的特定参数以及它们产生的可视化效果。

```shell
usage: ludwig visualize [options]

This script analyzes results and shows some nice plots.

optional arguments:
  -h, --help            show this help message and exit
  -g GROUND_TRUTH, --ground_truth GROUND_TRUTH
                        ground truth file
  -sf SPLIT_FILE, --split_file SPLIT_FILE
                        file containing split values used in conjunction with ground truth file
  -gm GROUND_TRUTH_METADATA, --ground_truth_metadata GROUND_TRUTH_METADATA
                        input metadata JSON file
  -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        directory where to save plots.If not specified, plots
                        will be displayed in a window
  -ff {pdf,png}, --file_format {pdf,png}
                        file format of output plots
  -v {binary_threshold_vs_metric,calibration_1_vs_all,calibration_multiclass,compare_classifiers_multiclass_multimetric,compare_classifiers_performance_changing_k,compare_classifiers_performance_from_pred,compare_classifiers_performance_from_prob,compare_classifiers_performance_subset,compare_classifiers_predictions,compare_classifiers_predictions_distribution,compare_performance,confidence_thresholding,confidence_thresholding_2thresholds_2d,confidence_thresholding_2thresholds_3d,confidence_thresholding_data_vs_acc,confidence_thresholding_data_vs_acc_subset,confidence_thresholding_data_vs_acc_subset_per_class,confusion_matrix,frequency_vs_f1,hyperopt_hiplot,hyperopt_report,learning_curves,roc_curves,roc_curves_from_test_statistics}, --visualization {binary_threshold_vs_metric,calibration_1_vs_all,calibration_multiclass,compare_classifiers_multiclass_multimetric,compare_classifiers_performance_changing_k,compare_classifiers_performance_from_pred,compare_classifiers_performance_from_prob,compare_classifiers_performance_subset,compare_classifiers_predictions,compare_classifiers_predictions_distribution,compare_performance,confidence_thresholding,confidence_thresholding_2thresholds_2d,confidence_thresholding_2thresholds_3d,confidence_thresholding_data_vs_acc,confidence_thresholding_data_vs_acc_subset,confidence_thresholding_data_vs_acc_subset_per_class,confusion_matrix,frequency_vs_f1,hyperopt_hiplot,hyperopt_report,learning_curves,roc_curves,roc_curves_from_test_statistics}
                        type of visualization
  -ofn OUTPUT_FEATURE_NAME, --output_feature_name OUTPUT_FEATURE_NAME
                        name of the output feature to visualize
  -gts GROUND_TRUTH_SPLIT, --ground_truth_split GROUND_TRUTH_SPLIT
                        ground truth split - 0:train, 1:validation, 2:test
                        split
  -tf THRESHOLD_OUTPUT_FEATURE_NAMES [THRESHOLD_OUTPUT_FEATURE_NAMES ...], --threshold_output_feature_names THRESHOLD_OUTPUT_FEATURE_NAMES [THRESHOLD_OUTPUT_FEATURE_NAMES ...]
                        names of output features for 2d threshold
  -pred PREDICTIONS [PREDICTIONS ...], --predictions PREDICTIONS [PREDICTIONS ...]
                        predictions files
  -prob PROBABILITIES [PROBABILITIES ...], --probabilities PROBABILITIES [PROBABILITIES ...]
                        probabilities files
  -trs TRAINING_STATISTICS [TRAINING_STATISTICS ...], --training_statistics TRAINING_STATISTICS [TRAINING_STATISTICS ...]
                        training stats files
  -tes TEST_STATISTICS [TEST_STATISTICS ...], --test_statistics TEST_STATISTICS [TEST_STATISTICS ...]
                        test stats files
  -hs HYPEROPT_STATS_PATH, --hyperopt_stats_path HYPEROPT_STATS_PATH
                        hyperopt stats file
  -mn MODEL_NAMES [MODEL_NAMES ...], --model_names MODEL_NAMES [MODEL_NAMES ...]
                        names of the models to use as labels
  -tn TOP_N_CLASSES [TOP_N_CLASSES ...], --top_n_classes TOP_N_CLASSES [TOP_N_CLASSES ...]
                        number of classes to plot
  -k TOP_K, --top_k TOP_K
                        number of elements in the ranklist to consider
  -ll LABELS_LIMIT, --labels_limit LABELS_LIMIT
                        maximum numbers of labels. If labels in dataset are
                        higher than this number, "rare" label
  -ss {ground_truth,predictions}, --subset {ground_truth,predictions}
                        type of subset filtering
  -n, --normalize       normalize rows in confusion matrix
  -m METRICS [METRICS ...], --metrics METRICS [METRICS ...]
                        metrics to dispay in threshold_vs_metric
  -pl POSITIVE_LABEL, --positive_label POSITIVE_LABEL
                        label of the positive class for the roc curve
  -l {critical,error,warning,info,debug,notset}, --logging_level {critical,error,warning,info,debug,notset}
                        the level of logging to use
```

关于这些参数的一些补充信息：

  * 列表参数被认为是对齐的，这意味着 `predictions`、`probabilities`、`training_statistics`、`test_statistics` 和 `model_names` 一起被索引，例如，在列表中产生第二个预测的模型名称将是模型名称中的第二个。
  * 其中 `ground_truth` 和 `ground_truth_metadata` 分别是训练预处理过程中获取的 `HDF5` 和 `JSON` 文件。如果您计划使用可视化，那么请确保在训练时不要使用 `skip_save_preprocessing`。这些文件是必需的，因为它们包含预处理时执行的分割，因此很容易从它们中提取测试集。
  * `output_feature_name` 是用于创建可视化的输出特征。

其他参数将针对每种可视化进行详细说明，因为不同的参数使用它们的方式有所不同。

生成可视化效果的示例命令基于运行两个实验并对它们进行比较。实验本身采用以下方法进行：

```shell
ludwig experiment --experiment_name titanic --model_name Model1 --dataset train.csv -cf titanic_model1.yaml
ludwig experiment --experiment_name titanic --model_name Model2 --dataset train.csv -cf titanic_model2.yaml
```

为此，您需要下载 [Titanic Kaggle competition dataset](https://www.kaggle.com/c/titanic/) 以获得 `train.csv`。请注意，下面与每种可视化相关联的图像不是来自泰坦尼克号的数据集。这两个模型是用 `titanic_model1.yaml` 定义的。

```yaml
input_features:
    -
        name: Pclass
        type: category
    -
        name: Sex
        type: category
    -
        name: Age
        type: numerical
        preprocessing:
          missing_value_strategy: fill_with_mean
    -
        name: SibSp
        type: numerical
    -
        name: Parch
        type: numerical
    -
        name: Fare
        type: numerical
        preprocessing:
          missing_value_strategy: fill_with_mean
    -
        name: Embarked
        type: category

output_features:
    -
        name: Survived
        type: binary
```

还有 `titanic_model2.yaml`

```yaml
input_features:
    -
        name: Pclass
        type: category
    -
        name: Sex
        type: category
    -
        name: SibSp
        type: numerical
    -
        name: Parch
        type: numerical
    -
        name: Embarked
        type: category

output_features:
    -
        name: Survived
        type: binary
```

### 学习曲线<a id='学习曲线'></a>
#### learning\_curves
这种可视化的参数：

  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `training_statistics`
  * `model_names`

对于每个模型(在 `training_statistics` 和 `model_names` 的对齐列表中)以及模型的每个输出特征和度量值，它生成一个折线图，显示在训练集和验证集的训练期间度量值是如何变化的。如果未指定 `output_feature_name`，则绘制所有输出特征。

示例命令：

```shell
ludwig visualize --visualization learning_curves \
  --output_feature_name Survived \
  --training_statistics results/titanic_Model1_0/training_statistics.json \
       results/titanic_Model2_0/training_statistics.json \
  --model_names Model1 Model2
```

![](https://ludwig-ai.github.io/ludwig-docs/images/learning_curves_loss.png)

![](https://ludwig-ai.github.io/ludwig-docs/images/learning_curves_accuracy.png)

### 混淆矩阵<a id='混淆矩阵'></a>
#### confusion\_matrix
这种可视化的参数：

  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `test_statistics`
  * `model_names`
  * `top_n_classes`
  * `normalize`

对于每个模型（在 `test_statistics` 和 `model_names` 的对齐列表中），它为在 `test_statistics` 中有混淆矩阵的每个字段的预测生成混淆矩阵的热力图。`top_n_classes` 的值将热力图限制为 `n` 个最频繁的类。

示例命令：

```shell
ludwig visualize --visualization confusion_matrix \
  --ground_truth_metadata results/titanic_Model1_0/model/train_set_metadata.json \
  --test_statistics results/titanic_Model1_0/test_statistics.json \
  --top_n_classes 2 
```

![](https://ludwig-ai.github.io/ludwig-docs/images/confusion_matrix.png)

产生的第二个图，是一个条形图，显示每个类的熵，从最大熵到最小熵排列。

![](https://ludwig-ai.github.io/ludwig-docs/images/confusion_matrix_entropy.png)

### 性能比较<a id='性能比较'></a>
#### compare\_performance
这种可视化的参数：

  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `test_statistics`
  * `model_names`

对于每个模型（在 `test_statistics` 和 `model_names` 的对齐列表中），它在条形图中生成条，每个条形图对应于 `test_statistics` 文件中指定的 `output_feature_name` 的可用总体度量。

示例命令：

```shell
ludwig visualize --visualization compare_performance \
  --output_feature_name Survived \
  --test_statistics results/titanic_Model1_0/test_statistics.json \
       results/titanic_Model2_0/test_statistics.json \
  --model_names Model1 Model2 
```

![](https://ludwig-ai.github.io/ludwig-docs/images/compare_performance.png)

#### compare\_classifiers\_performance\_from\_prob
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `ground_truth_split`
  * `probabilities`
  * `model_names`
  * `top_n_classes`
  * `labels_limit`

`output_feature_name` 需要是一个类别。对于每个模型(在 `probabilities` 和 `model_names` 的对齐列表中)，它在条形图中生成条，每个条形图对应一个从指定的 `output_feature_name` 的预测概率实时计算出的总体度量。

示例命令：

```shell
ludwig visualize --visualization compare_classifiers_performance_from_prob \
  --ground_truth train.hdf5 \
  --output_feature_name Survived \
  --probabilities results/titanic_Model1_0/Survived_probabilities.csv \
        results/titanic_Model2_0/Survived_probabilities.csv \
  --model_names Model1 Model2 
```

![](https://ludwig-ai.github.io/ludwig-docs/images/compare_classifiers_performance_from_prob.png)

#### compare\_classifiers\_performance\_from\_pred
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `ground_truth_split`
  * `predictions`
  * `model_names`
  * `labels_limit`

`output_feature_name` 需要是一个类别。对于每个模型(在 `predictions` 和 `model_names` 的对齐列表中)，它在条形图中生成条，每个条形图对应一个从指定的 `output_feature_name` 的预测实时计算出的总体度量。

示例命令：

```shell
ludwig visualize --visualization compare_classifiers_performance_from_pred \
  --ground_truth train.hdf5 \
  --ground_truth_metadata train.json \
  --output_feature_name Survived \
  --predictions results/titanic_Model1_0/Survived_predictions.csv \
        results/titanic_Model2_0/Survived_predictions.csv \
  --model_names Model1 Model2 
```

![](https://ludwig-ai.github.io/ludwig-docs/images/compare_classifiers_performance_from_pred.png)

#### compare\_classifiers\_performance\_subset
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `ground_truth_split`
  * `probabilities`
  * `model_names`
  * `top_n_classes`
  * `labels_limit`
  * `subset`

`output_feature_name` 需要是一个类别。对于每个模型(在 `predictions` 和 `model_names` 的对齐列表中)，它在条形图中生成一个条，每个条形图是根据指定的 `output_feature_name` 的概率预测实时计算的总体度量，只考虑整个训练集的一个子集。获取子集的方法是使用 `top_n_classes` 和 `subset` 参数。

如果 `subset` 的值为 `ground_truth`，那么只有标注类位于最频繁的 `n` 中的数据点才会被视为测试集，并且会显示从原始集保留的数据点的百分比。

示例命令：

```shell
ludwig visualize --visualization compare_classifiers_performance_subset \
  --ground_truth train.hdf5 \
  --ground_truth_metadata train.json \
  --output_feature_name Survived \
  --probabilities results/titanic_Model1_0/Survived_probabilities.csv \
           results/titanic_Model2_0/Survived_probabilities.csv \
  --model_names Model1 Model2 \
  --top_n_classes 2 \
  --subset ground_truth 
```

![](https://ludwig-ai.github.io/ludwig-docs/images/compare_classifiers_performance_subset_gt.png)

如果 `subset` 的值是 `predictions`，那么只有模型预测的类位于最频繁的 `n` 个类中的数据点才会被视为测试集，并且每个模型都会显示从原始集保留的数据点的百分比。

![](https://ludwig-ai.github.io/ludwig-docs/images/compare_classifiers_performance_subset_pred.png)

#### compare\_classifiers\_performance\_changing\_k
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `ground_truth_split`
  * `probabilities`
  * `model_names`
  * `top_k`
  * `labels_limit`

`output_feature_name` 需要是一个类别。对于每个模型(在 `probabilities` 和 `model_names` 的对齐列表中)，它会产生一个折线图，当指定的 `output_feature_name` 的 `k` 值从 1 改变至 `top_k` 时该折线图显示 Hits@K 度量(如果模型在第一个 `k` 中产生预测，则将其视为正确)。

示例命令：

```shell
ludwig visualize --visualization compare_classifiers_performance_changing_k \
  --ground_truth train.hdf5 \
  --output_feature_name Survived \
  --probabilities results/titanic_Model1_0/Survived_probabilities.csv \
         results/titanic_Model2_0/Survived_probabilities.csv \
  --model_names Model1 Model2 \
  --top_k 5 
```

![](https://ludwig-ai.github.io/ludwig-docs/images/compare_classifiers_performance_changing_k.png)

#### compare\_classifiers\_multiclass\_multimetric
这种可视化的参数：

  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `test_statistics`
  * `model_names`
  * `top_n_classes`

`output_feature_name` 需要是一个类别。对于每个模型(在 `test_statistics` 和 `model_names` 的对齐列表中)它生成四张图，显示指定的 `output_feature_name` 的几个类上模型的精度、召回和F1。

第一张图显示了 `n` 个最常见类的度量。

![](https://ludwig-ai.github.io/ludwig-docs/images/compare_classifiers_multiclass_multimetric_topk.png)

第二张图显示了模型性能最好的 `n` 个类的度量。

![](https://ludwig-ai.github.io/ludwig-docs/images/compare_classifiers_multiclass_multimetric_bestk.png)

第三张图显示了模型性能最差的 `n` 个类的度量。

![](https://ludwig-ai.github.io/ludwig-docs/images/compare_classifiers_multiclass_multimetric_worstk.png)

第四张图显示了所有类的度量，按频率排序。如果类的数量真的很多，这可能会变得不可读。

![](https://ludwig-ai.github.io/ludwig-docs/images/compare_classifiers_multiclass_multimetric_sorted.png)

### 比较分类器的预测<a id='比较分类器的预测'></a>
#### compare\_classifiers\_predictions
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `ground_truth_split`
  * `predictions`
  * `model_names`
  * `labels_limit`

`output_feature_name` 需要是一个类别，必须有且只有两个模型(在 `predictions` 和 `model_names` 的对齐列表中)。这种可视化会生成一个饼图，以比较两个模型对指定的 `output_feature_name` 的预测。

示例命令：

```shell
ludwig visualize --visualization compare_classifiers_predictions \
  --ground_truth train.hdf5 \
  --output_feature_name Survived \
  --predictions results/titanic_Model1_0/Survived_predictions.csv \
          results/titanic_Model2_0/Survived_predictions.csv \
  --model_names Model1 Model2 
```

![](https://ludwig-ai.github.io/ludwig-docs/images/compare_classifiers_predictions.png)

#### compare\_classifiers\_predictions\_distribution
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `ground_truth_split`
  * `predictions`
  * `model_names`
  * `label_limits`

`output_feature_name` 需要是一个类别。这种可视化生成了一个雷达图，比较了指定的 `output_feature_name` 的前 10 个类的模型预测分布。

![](https://ludwig-ai.github.io/ludwig-docs/images/compare_classifiers_predictions_distribution.png)

### 可信阈值<a id='可信阈值'></a>
#### confidence\_thresholding
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `ground_truth_split`
  * `probabilities`
  * `model_names`
  * `labels_limit`

`output_feature_name` 需要是一个类别。对于每个模型(在 `probabilities` 和 `model_names` 的对齐列表中)，它生成一对折线，指示模型的准确性和数据覆盖率，同时增加指定的 `output_feature_name` 预测概率的阈值(x 轴)。

![](https://ludwig-ai.github.io/ludwig-docs/images/confidence_thresholding.png)

#### confidence\_thresholding\_data\_vs_acc
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `ground_truth_split`
  * `probabilities`
  * `model_names`
  * `labels_limit`

`output_feature_name` 需要是一个类别。对于每个模型(在 `probabilities` 和 `model_names` 的对齐列表中)，它生成一条折线，指示模型的准确性和数据覆盖率，同时增加指定的 `output_feature_name` 的预测概率的阈值。与 `confence_thresholding` 的不同之处在于，它使用了两个轴而不是三个轴，没有了可视化阈值，且覆盖范围是 x 轴而不是阈值。

![](https://ludwig-ai.github.io/ludwig-docs/images/confidence_thresholding_data_vs_acc.png)

#### confidence\_thresholding\_data\_vs\_acc\_subset
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `ground_truth_split`
  * `probabilities`
  * `model_names`
  * `top_n_classes`
  * `labels_limit`
  * `subset`

`output_feature_name needs` 需要是一个类别。为每个模型(在 `probabilities` 和 `model_names` 的对齐列表中)，它生成一条折线，指示模型的准确性和数据覆盖率，同时增加指定的 `output_feature_name` 的预测概率的阈值，只考虑完整的训练集的一个子集，获取子集的方法是使用 `top_n_classes` 和 `subset` 参数。与 `confence_thresholding` 的不同之处在于，它使用了两个轴而不是三个轴，没有了可视化阈值，且覆盖范围是 x 轴而不是阈值。

如果 `subset` 的值是 `ground_truth`，那么只有标注类位于最频繁的 `n` 类中的数据点才会被视为测试集，并且会显示从原始集保留的数据点的百分比。如果 `subset` 的值是 `predictions`，那么只有模型预测的类位于最频繁的 `n` 个类中的数据点才会被视为测试集，并且每个模型都会显示从原始集保留的数据点的百分比。

![](https://ludwig-ai.github.io/ludwig-docs/images/confidence_thresholding_data_vs_acc_subset.png)

#### confidence\_thresholding\_data\_vs\_acc\_subset\_per\_class
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `ground_truth_split`
  * `probabilities`
  * `model_names`
  * `top_n_classes`
  * `labels_limit`
  * `subset`

`output_feature_name needs` 需要是一个类别。为每个模型(在 `probabilities` 和 `model_names` 的对齐列表中)，它生成一条折线，指示模型的准确性和数据覆盖率，同时增加指定的 `output_feature_name` 的预测概率的阈值，只考虑完整的训练集的一个子集，获取子集的方法是使用 `top_n_classes` 和 `subset` 参数。与 `confence_thresholding` 的不同之处在于，它使用了两个轴而不是三个轴，没有了可视化阈值，且覆盖范围是 x 轴而不是阈值。

如果 `subset` 的值是 `ground_truth`，那么只有标注类位于最频繁的 `n` 类中的数据点才会被视为测试集，并且会显示从原始集保留的数据点的百分比。如果 `subset` 的值是 `predictions`，那么只有模型预测的类位于最频繁的 `n` 个类中的数据点才会被视为测试集，并且每个模型都会显示从原始集保留的数据点的百分比。

与 `confence_thresholding_data_vs_acc_subset` 的区别在于，它在 `top_n_classes` 中为每个类生成一个图。

![](https://ludwig-ai.github.io/ludwig-docs/images/confidence_thresholding_data_vs_acc_subset_per_class_1.png)

![](https://ludwig-ai.github.io/ludwig-docs/images/confidence_thresholding_data_vs_acc_subset_per_class_4.png)

#### confidence\_thresholding\_2thresholds\_2d
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `ground_truth_split`
  * `threshold_output_feature_names`
  * `probabilities`
  * `model_names`
  * `labels_limit`

`threshold_output_feature_names` 必须恰好是两个，要么是类别，要么是二进制。`probabilities ` 也必须恰好为两个，与 `threshold_output_feature_names` 对齐。`model_names` 必须是一个。它生成了三个图形。

第一个图形显示了几条半透明的折线。它们总结了由 `confidence_thresholding_2thresholds_3d` 显示的三维曲面，这些曲面的置信度阈值取决于两个 `threshold_output_feature_names` 的预测置信度(x 轴和 y 轴)以及数据覆盖率或精度(z 轴)。每条折线表示投影到精度曲面上的数据覆盖曲面的切片。

![](https://ludwig-ai.github.io/ludwig-docs/images/confidence_thresholding_2thresholds_2d_multiline.png)

第二个图形显示了在第一个图形中显示的所有折线的最大值。

![](https://ludwig-ai.github.io/ludwig-docs/images/confidence_thresholding_2thresholds_2d_maxline.png)

第三个图形显示了获得特定数据覆盖率与精度值对的最大折线和阈值。

![](https://ludwig-ai.github.io/ludwig-docs/images/confidence_thresholding_2thresholds_2d_accthr.png)

#### confidence\_thresholding\_2thresholds\_3d
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `ground_truth_split`
  * `threshold_output_feature_names`
  * `probabilities`
  * `labels_limit`

`threshold_output_feature_names` 必须恰好是两个，要么是类别，要么是二进制。`probabilities` 也必须恰好为两个，与`threshold_output_feature_names` 对齐。图中显示了由 `confence_thresholding_2thresholds_3d` 所显示的三维曲面，其中 x 轴和 y 轴为两个 `threshold_output_feature_names` 预测的置信度阈值，z 轴为数据覆盖率百分比或精度。

![](https://ludwig-ai.github.io/ludwig-docs/images/confidence_thresholding_2thresholds_3d.png)

### 二进制阈值与度量<a id='二进制阈值与度量'></a>
#### binary\_threshold\_vs\_metric
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `ground_truth_split`
  * `probabilities`
  * `model_names`
  * `metrics`
  * `positive_label`

`output_feature_name` 可以是类别或二进制特征。对于在度量中指定的每个 `metric`（选项为 `f1`、`precision`、`recall`、`accurity`），此可视化将根据指定的 `output_feature_name` 的度量绘制模型置信度阈值折线图。如果 `output_feature_name` 是一个类别特征，`positive_label` 表示哪个类被视为正类，其余所有类则将视为负类。它需要是一个整数，要找出类和整数之间的关联，请核对 `ground_truth_metadata` JSON 文件。

![](https://ludwig-ai.github.io/ludwig-docs/images/binary_threshold_vs_metric.png)

### ROC 曲线<a id='ROC_曲线'></a>
#### roc\_curves
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `ground_truth_split`
  * `probabilities`
  * `model_names`
  * `positive_label`

`output_feature_name` 可以是一个类别或二进制特征。这种可视化绘制指定的 `output_feature_name` 的 roc 曲线。如果 `output_feature_name` 是一个类别特征，`positive_label` 表示哪个类将被认为是正类，其余所有类则将视为负类。它需要是一个整数，要找出类和整数之间的关联，请核对 `ground_truth_metadata` JSON 文件。

![](https://ludwig-ai.github.io/ludwig-docs/images/roc_curves.png)

#### roc\_curves\_from\_test\_statistics
这种可视化的参数：

  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `test_statistics`
  * `model_names`
  
`output_feature_name` 需要是二进制特征。这种可视化绘制指定的 `output_feature_name` 的 roc 曲线。

![](https://ludwig-ai.github.io/ludwig-docs/images/roc_curves_from_test_statistics.png)

### 校准图<a id='校准图'></a>
#### calibration\_1\_vs\_all
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `ground_truth_split`
  * `probabilities`
  * `model_names`
  * `top_n_classes`
  * `labels_limit`

`output_feature_name` 必须是类别或二进制。对于每个类或每个 `n` 最常出现的类，如果指定了 `top_n_classes`，它会根据指定的 `output_feature_name` 的预测概率实时生成两个图形。

第一个图形是一个校准曲线，它显示预测的校准，认为当前类为真类，其他所有类为假类，并在图形上为每个模型绘制一条线(在 `probabilities` 和 `model_names` 的对齐列表中)。

![](https://ludwig-ai.github.io/ludwig-docs/images/calibration_1_vs_all_curve.png)

第二个图形显示了认为当前类为真类而其他所有类为假类的预测分布，并绘制了每个模型的分布(在 `probabilities` 和 `model_names` 的对齐列表中)。

![](https://ludwig-ai.github.io/ludwig-docs/images/calibration_1_vs_all_counts.png)

#### calibration\_multiclass
这种可视化的参数：

  * `ground_truth`
  * `split_file`
  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `ground_truth_split`
  * `probabilities`
  * `model_names`
  * `labels_limit`

`output_feature_name` 需要是一个类别。对于每个类，根据指定的 `output_feature_name` 的预测概率生成两个实时计算的图形。

第一个图形是一个校准曲线，它显示了考虑到所有类别的预测的校准，并在图形上为每个模型绘制一条线(在 `probabilities` 和 `model_names` 的对齐列表中)。

![](https://ludwig-ai.github.io/ludwig-docs/images/calibration_multiclass_curve.png)

第二个图形显示了 brier 得分的条形图(计算模型预测的概率是如何校准的)，并在图形上为每个模型绘制一个条形图(在 `probabilities` 和 `model_names` 的对齐列表中)。

![](https://ludwig-ai.github.io/ludwig-docs/images/calibration_multiclass_brier.png)

### 类别频率与 F1 评分<a id='类别频率与_F1_评分'></a>
#### frequency\_vs\_f1
这种可视化的参数：

  * `ground_truth_metadata`
  * `output_directory`
  * `file_format`
  * `output_feature_name`
  * `test_statistics`
  * `model_names`
  * `top_n_classes`

`output_feature_name` 需要是一个类别。对于每个模型(在 `test_statistics` 和 `model_names` 的对齐列表中)，为指定的 `output_feature_name` 生成两个预测统计图形。

生成 `top_n_classes` 的图。第一个图形是直线图，其中一个 x 轴表示不同的类别，两个垂直轴分别用橙色和蓝色表示。橙色的轴是类别的频率，用一条橙线来显示趋势。蓝色的轴是该级别的 F1 评分，用一条蓝线来显示趋势。x 轴上的类别按 f1 评分排序。

![](https://ludwig-ai.github.io/ludwig-docs/images/freq_vs_f1_sorted_f1.png)

第二个图形的结构与第一个相同，但坐标轴被翻转，而且 x 轴上的类是按频率排序的。

![](https://ludwig-ai.github.io/ludwig-docs/images/freq_vs_f1_sorted_freq.png)

### 超参数优化的可视化<a id='超参数优化的可视化'></a>
这里的超参数可视化示例是通过在 [ATIS dataset](https://www.kaggle.com/siddhadev/ms-cntk-atis) 上随机搜索 100 个样本获得的，该数据集用于对给定用户话语的意图进行分类。

#### hyperopt\_report
这种可视化的参数：

  * `output_directory`
  * `file_format`
  * `hyperopt_stats_path`

可视化为 `hyperopt_stats_path` 文件中的每个超参数创建一个图形，外加一个包含一对超参数交互的图形。

每个图形将显示与要优化的度量有关的参数分布。对于 `float` 和 `int` 参数，使用散点图，而对于 `category` 参数，则使用小提琴图。

![](https://ludwig-ai.github.io/ludwig-docs/images/hyperopt_float.png)

![](https://ludwig-ai.github.io/ludwig-docs/images/hyperopt_int.png)

![](https://ludwig-ai.github.io/ludwig-docs/images/hyperopt_category.png)

配对图显示了一对超参数的值如何与要优化的度量关联的热图。

![](https://ludwig-ai.github.io/ludwig-docs/images/hyperopt_pair.png)

#### hyperopt\_hiplot
这种可视化的参数：

  * `output_directory`
  * `file_format`
  * `hyperopt_stats_path`

可视化创建了一个交互式 HTML 页面，使用平行坐标图一次可视化超参数优化的所有结果。

![](https://ludwig-ai.github.io/ludwig-docs/images/hyperopt_hiplot.jpeg)
























