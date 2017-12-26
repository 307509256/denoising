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

## install python3.5
sudo apt-get install python-software-properties
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:fkrull/deadsnakes
sudo apt-get update
sudo apt-get install python3.5
sudo cp /usr/bin/python /usr/bin/python_bak
sudo rm /usr/bin/python
sudo ln -s /usr/bin/python3.5 /usr/bin/python

python --version

## install pip3.5
wget https://bootstrap.pypa.io/get-pip.py  
sudo python3 get-pip.py  
sudo pip3 install setuptools --upgrade  
sudo pip3 install ipython[all]

pip -V

## install
pip3 install scipy==1.0.0 audioread==2.1.5 bleach==1.5.0 cffi==1.11.2 decorator==4.1.2 enum34==1.1.6 html5lib==0.9999999 joblib==0.11 librosa==0.5.1 llvmlite==0.21.0 Markdown==2.6.10 nnresample==0.1.2 numba==0.36.1 numpy==1.13.3 protobuf==3.5.0.post1 pycparser==2.18 resampy==0.2.0 scikit-learn==0.19.1 six==1.11.0 SoundFile==0.9.0.post1 tensorflow-tensorboard==0.4.0rc3 Werkzeug==0.13
