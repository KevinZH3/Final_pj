### Task2任务描述
在task2中，完成了基于ResNet和Vision Transfomer对CIFAR100数据集进行图像分类任务
### 代码文件功能简介
在当前路径下，有两个文件夹，分别是resnet和vit。每个文件夹内有4个代码文件，分别是:
- train_model.py：这是模型的训练性能和测试文件，您可通过命令 `python resnet/train_resnet.py` 和 `python vit/train_vit.py` 命令，使用默认超参数设置对模型进行训练。若需要自定义训练细节，模型超参数等，请阅读 `utils.py`文件中命令行解析函数的定义，在根据需要传入相应参数。
- dataset.py：这是数据集处理的文件，实现了数据预处理，CutMix数据增强等函数。您不需调用此文件。
- model.py：这是相应模型结构的定义文件。您不需调用此文件。 
- utils.py：这里实现了各种深度学习训练的实用工具。您不需调用此文件。 
