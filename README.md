#### 各部分功能
+ assess 对结果评分并统计
+ data 处理数据集
+ dataset 数据集存放目录
+ nnet 神经网络模型
+ sound 语音特征提取
+ do_assess.py 评估测试输出的表现
+ global_config.py 全局配置，包括日志配置
+ run.py 训练测试运行示例
+ util.py 包含各种辅助处理函数

#### 环境
- Ubuntu16.04 python3.5
- Windows


#### 依赖
+ tensorflow-gpu==1.4.1
+ nnresample==0.1.2
+ librosa==0.5.1
+ SoundFile==0.9.0.post1

#### 数据下载
数据下载-百度云盘：[链接](http://pan.baidu.com/s/1bZAQyM) 密码：pqre

语音数据来自TSP，噪声数据来自NOISEX-92。
详细情况见数据文件夹内的readme.md。
下载好后解压到denoising_windows/dataset目录下