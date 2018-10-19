---
title: Bagging和Boosting的概念与区别
date: 2018-06-11 23:31:23
categories: 
    - machine learning
tags: 
    - machine learning
---

<!-- TOC -->

- [Bagging和Boosting的概念与区别](#bagging%E5%92%8Cboosting%E7%9A%84%E6%A6%82%E5%BF%B5%E4%B8%8E%E5%8C%BA%E5%88%AB)
- [Bagging算法(套袋法，bootstrap aggregating)](#bagging%E7%AE%97%E6%B3%95%E5%A5%97%E8%A2%8B%E6%B3%95bootstrap-aggregating)
- [Boosting（提升法）](#boosting%E6%8F%90%E5%8D%87%E6%B3%95)
- [Bagging和Boosting 的主要区别](#bagging%E5%92%8Cboosting-%E7%9A%84%E4%B8%BB%E8%A6%81%E5%8C%BA%E5%88%AB)
- [将决策树与以上框架组合成新的算法](#%E5%B0%86%E5%86%B3%E7%AD%96%E6%A0%91%E4%B8%8E%E4%BB%A5%E4%B8%8A%E6%A1%86%E6%9E%B6%E7%BB%84%E5%90%88%E6%88%90%E6%96%B0%E7%9A%84%E7%AE%97%E6%B3%95)

<!-- /TOC -->

#### Bagging和Boosting的概念与区别
- 随机森林属于集成学习(ensemble learning)中的bagging算法，在集成算法中主要分为bagging算法与boosting算法


#### Bagging算法(套袋法，bootstrap aggregating)

- bagging的算法过程如下：
- 从原始样本集中使用Bootstraping 方法随机抽取n个训练样本，共进行k轮抽取，得到k个训练集（k个训练集之间相互独立，元素可以有重复）。
- 对于n个训练样本，我们训练k个模型，（这个模型可根据具体的情况而定，可以是决策树，knn等）
- 对于分类问题：由投票表决产生的分类结果；对于回归问题，由k个模型预测结果的均值作为最后预测的结果（所有模型的重要性相同）。

#### Boosting（提升法）

- boosting的算法过程如下： 
- 对于训练集中的每个样本建立权值wi，表示对每个样本的权重， 其关键在与对于被错误分类的样本权重会在下一轮的分类中获得更大的权重（错误分类的样本的权重增加）。
- 同时加大分类 误差概率小的弱分类器的权值，使其在表决中起到更大的作用，减小分类误差率较大弱分类器的权值，使其在表决中起到较小的作用。每一次迭代都得到一个弱分类器，需要使用某种策略将其组合，最为最终模型，(adaboost给每个迭代之后的弱分类器一个权值，将其线性组合作为最终的分类器,误差小的分类器权值越大。)

#### Bagging和Boosting 的主要区别

- 样本选择上: Bagging采取Bootstraping的是随机有放回的取样，Boosting的每一轮训练的样本是固定的，改变的是每个样的权重。
- 样本权重上：**Bagging采取的是均匀取样，且每个样本的权重相同**，Boosting根据错误率调整样本权重，错误分类的样本权重会变大
- 预测函数上：**Bagging所有的预测函数权值相同**，Boosting中误差越小的预测函数其权值越大。
- 并行计算: Bagging 的各个预测函数可以并行生成, Boosting的各个预测函数必须按照顺序迭代生成.

#### 将决策树与以上框架组合成新的算法

- Bagging + 决策树 = 随机森林
- AdaBoost + 决策树 = 提升树
- gradient + 决策树 = （梯度提升树）GDBT 
