# Welcome to HyperTS

[English](README.md)

[![Python Versions](https://img.shields.io/pypi/pyversions/hypernets.svg)](https://pypi.org/project/hypernets)
[![TensorFlow Versions](https://img.shields.io/badge/TensorFlow-2.0+-blue.svg)](https://pypi.org/project/deeptables)
[![License](https://img.shields.io/github/license/DataCanvasIO/deeptables.svg)](https://github.com/DataCanvasIO/deeptables/blob/master/LICENSE)

:dizzy: 易用，高效，统一的全管道自动时间序列分析工具，支持时间序列预测，分类及回归。

## 有志者，跟我来！
亲爱的朋友们，我们在为热爱AutoML/NAS的专业人士和学生提供具有挑战性的机会。目前，我们的团队遍布北京(总部)、上海，成都，美国等世界各地，欢迎有志之士加入我们的团队DataCanvas Lab! 请您发送您的简历到 yangjian@zetyun.com. 

## 概览
HyperTS是一个Python工具包，提供了一个端到端的时间序列分析工具。它针对时间序列任务的整个AutoML流程实现了灵活的全覆盖，包含数据清洗，数据预处理，特征工程，模型选择，超参数优化，结果评估以及预测曲线可视化等。

多模驱动, 轻重结合是HyperTS的关键特性。因此，您可以随意切换统计(+机器学习), 深度学习及神经架构搜索等模式来获得强大的评估器。

简单易上手的API。您可以简单操作创建一个实验，然后```run()```它，便会获得一个最佳的全pipeline模型。然后针对得到的model执行```.predict()```, ```.predict_proba()```, ```.evalute()```, ```.plot()```等操作来对做各种各样的时间序列结果分析。

## 安装

HyperTS在Pypi上可用，可以使用pip安装:

```bash
pip install hyperts
```

更多安装细节及注意事项，请看 [安装指南](https://hyperts.readthedocs.io/zh_CN/latest/contents/0200_installation.html).


## 教程

|[中文文档](https://hyperts.readthedocs.io/zh_CN/latest/) / [英文文档](https://hyperts.readthedocs.io/en/latest) | 描述 |
| --------------------------------- | --------------------------------- |
[数据规范](https://hyperts.readthedocs.io/zh_CN/latest/contents/0300_dataformat.html)|HyperTS期待什么样的数据？|
|[快速开始](https://hyperts.readthedocs.io/zh_CN/latest/contents/0400_quick_start.html)| 如何快速正确地使用HyperTS？|
|[进阶之梯](https://hyperts.readthedocs.io/zh_CN/latest/contents/0500_advanced_config.html)|如何释放HyperTS的巨大潜能？|
|[自定义化](https://hyperts.readthedocs.io/zh_CN/latest/contents/0600_user_defined.html)|如何定制化自己的HyperTS?|

## 示例

您可以使用```make_experiment()```快速创建并运行一个实验，其中```train_data```和```task```作为必需的输入参数。在以下预测示例中，我们告诉实验这是一个多变量预测任务，开启```stats```模式(统计)，因为数据包含时间戳和协变量列，因此```timestamp```和```covariables```参数也必须传给实验。

```python
from hyperts.experiment import make_experiment
from hyperts.datasets import load_network_traffic

from sklearn.model_selection import train_test_split

data = load_network_traffic()
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

model = make_experiment(train_data.copy(),
                        task='multivariate-forecast',
                        mode='stats',
                        timestamp='TimeStamp',
                        covariables=['HourSin', 'WeekCos', 'CBWD']).run()

X_test, y_test = model.split_X_y(test_data.copy())

y_pred = model.predict(X_test)

scores = model.evaluate(y_test, y_pred)

model.plot(forecast=y_pred, actual=test_data)
```

![Forecast_Figure](docs/static/images/Actual_vs_Forecast.jpg)

- 更多示例及使用技巧，请移步: [中文示例.](https://github.com/DataCanvasIO/HyperTS/tree/main/examples/zh_CN)



## 关键特性

HyperTS支持以下特性:

**多任务支持:** 时间序列预测、分类及回归。

**多模式支持:** 大量的时序模型，从统计模型到深度学习模型，再到神经架构搜索(开发中)。

**多变量支持:** 支持从单变量到多变量时间序列任务。

**协变量支持:** 深度学习模型支持协变量作为时间序列预测的输入特征。

**概率置信区间:** 时间序列预测可视化可以显示置信区间。

**丰富的指标:** 从MSE、SMAPE、Accuracy到F1-Score，多种性能指标来评估结果，指导模型优化。

**强大的搜索策略:** 采用网格搜索、蒙特卡罗树搜索、进化算法，并结合元学习器，为时间序列分析提供了强大而有效的管道。


## 贡献
如果您想为HyperTS做一些贡献, 请参考 [CONTRIBUTING](CONTRIBUTING.md).

## 相关项目
* [Hypernets](https://github.com/DataCanvasIO/Hypernets): 一个通用的自动机器学习框架。
* [HyperGBM](https://github.com/DataCanvasIO/HyperGBM): 一个集成了多个GBM模型的全Pipeline AutoML工具。
* [HyperDT/DeepTables](https://github.com/DataCanvasIO/DeepTables): 一个面向结构化数据的AutoDL工具。
* [HyperTS](https://github.com/DataCanvasIO/HyperTS): 一个面向时间序列数据的AutoML和AutoDL工具。
* [HyperKeras](https://github.com/DataCanvasIO/HyperKeras): 一个为Tensorflow和Keras提供神经架构搜索和超参数优化的AutoDL工具。
* [HyperBoard](https://github.com/DataCanvasIO/HyperBoard): 一个为Hypernets提供可视化界面的工具。
* [Cooka](https://github.com/DataCanvasIO/Cooka): 一个交互式的轻量级自动机器学习系统。
  
![DataCanvas AutoML Toolkit](docs/static/images/datacanvas_automl_toolkit.png)

## DataCanvas

![datacanvas](docs/static/images/dc_logo_1.png)

HyperTS是由数据科学平台领导厂商 [DataCanvas](https://www.datacanvas.com/) 创建的开源项目。