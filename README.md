# 使用Decoder-only的Transformer模型实现时序预测，Implement time series prediction using a Decoder-only Transformer model.

## Chinese Introduction

使用pytorch实现的Decoder-only的Pre-Norm型的Transformer模型，包含SwiGLU作为FeedForward的激活层，RoPE(Rotary Positional Embedding)。使用SMAPE作为损失函数，同时也是评价指标。

### 文件描述

- [interpolation.py](interpolation.py): 预处理数据，包括去除异常候选项、多种插值方法
- [data_visualization.py](data_visualization.py): 可视化数据
- [model.py](model.py): 模型的定义
- [loss.py](loss.py): SMAPE损失函数，同时也是评价指标
- [dataset.py](dataset.py): 模型的dataset
- [train_trans.py](train_trans.py): 训练Transformer模型
- [train_last.py](train_last.py): 测试baseline方法：后面的预测全部使用前面的最后一个值
- [train_mid.py](train_mid.py): 测试baseline方法：后面的预测全部使用前面的中位数
- [train_mlp.py](train_mlp.py): 训练MLP模型

### 运行步骤
1. 运行interpolation.py，对数据进行预处理
2. 运行data_visualization.py，可视化数据（可选）
3. 运行train_().py，训练模型

### 数据集
数据集来自于[这里](https://www.kaggle.com/c/web-traffic-time-series-forecasting/)，如果后续链接失效或者不便下载也可以跟我联系。
数据有14万行，每行是一个时间序列，包含了2015年7月1日到2016年12月31日的数据，数据维度为[14万, 550]。

> 本代码尝试使用了KVcache加速模型推理（模型的predict模块）。不过在代码中，由于位置编码的原因，在使用KV_cache后的Q与逐步预测有所不同（使用KV_cache后，序列长度变短了，变成了1）。
> 在参考了 llamas 的代码（[链接](https://github.com/facebookresearch/llama/blob/main/llama/model.py#280)）后，我依旧没有找到解决这个问题的方法。因此，我不建议使用predict模块进行预测。
> 顺便说一句，如果您能解决这个问题，请一定告诉我。


## English Introduction
### Transformer Model with Pre-Norm and SwiGLU Activation

Implemented a Decoder-only Transformer model using PyTorch. The model includes SwiGLU as the activation layer for the FeedForward network and utilizes Rotary Positional Embedding (RoPE). SMAPE (Symmetric Mean Absolute Percentage Error) is used as both the loss function and evaluation metric.

### File Descriptions

- [interpolation.py](interpolation.py): Preprocesses data, including removing outliers and applying various interpolation methods.
- [data_visualization.py](data_visualization.py): Visualizes data.
- [model.py](model.py): Defines the Transformer model.
- [loss.py](loss.py): Implements the SMAPE loss function, also used as the evaluation metric.
- [dataset.py](dataset.py): Defines the dataset for the model.
- [train_trans.py](train_trans.py): Trains the Transformer model.
- [train_last.py](train_last.py): Tests the baseline method using the last value for all subsequent predictions.
- [train_mid.py](train_mid.py): Tests the baseline method using the median of previous values for all subsequent predictions.
- [train_mlp.py](train_mlp.py): Trains an MLP model.

### Execution Steps

1. Run `interpolation.py` to preprocess the data.
2. Run `data_visualization.py` for optional data visualization.
3. Run `train_().py` to train the model.

### Dataset
The dataset is obtained from [here](https://www.kaggle.com/c/web-traffic-time-series-forecasting/). In case the link becomes inaccessible or inconvenient for download, feel free to contact me. The dataset comprises 140,000 rows, each representing a time series from July 1, 2015, to December 31, 2016, with dimensions [140,000, 550].

> The code attempts to use KVcache to accelerate model inference in the predict module. However, due to positional encoding, the Q value after using KV_cache differs from step-by-step prediction (after using KV_cache, the sequence length becomes 1).
> After referencing llamas' code ([link](https://github.com/facebookresearch/llama/blob/main/llama/model.py#280)), I still haven't found a solution to this issue. Therefore, I do not recommend using the predict module for forecasting. 
> By the way, if you can solve this problem, please let me know.


## If you have any question of the code, or you know how to solve the problem of "kvcache-pos_emb conflict", please contact me or leave an issue. My email:1793706453@qq.com






