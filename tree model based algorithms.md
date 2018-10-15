### tree based ensemble algorithms 
- 原始的Boost算法是在算法开始的时候，为每个样本赋上一个权重值，初始的时候，每个样本都是同样的重要。在每一步的训练中，得到的模型，会给出每个数据点的估计对错，根据判断的对错，在每一步的训练之后，会增加分错样本的权重，减少分类正确的样本的权重，如果在后续的每一步训练中，如果继续被分错，那么就会被严重的关注，也就是获得了一个比较高的权重。经过N次迭代之后，将会得到N个简单的分类器（basic learner），然后将他们组装起来（可以进行加权，或者进行投票），得到一个最终的模型。

### 主要介绍已下几种ensemble的分类器（tree based algorithms）

- xgboost 
- lightBMG
- GDBT
- Random forest


#### xgboost
- xgboost能自动利用cpu的多线程，而且适当改进了gradient boosting，加了剪枝，控制了模型的复杂程度
- 传统的GBDT算法以CART作为基分类器，**xgboost还可以支持线性分类器**，相当于带L1和L2的逻辑斯谛回归或者线性回归
- 传统的GBDT在优化的时候，使用的是一阶导数信息，**xgboost则对代价函数进行了二阶泰勒展开，同时用到了一阶导数和二阶导数**。顺便提一下，xgboost工具支持自定义代价函数，只要函数可一阶和二阶求导。
- xgboost在代价函数中加入了正则项，用于控制模型的复杂度。**正则项里面包括树的叶子节点的个数、每个叶子节点上输出的score的L2模的平方和**。（ 从Bias-variance tradeoff 的角度来说，正则化降低了模型的variance，使得到的模型更加简单，防止过拟合，这是xgboost优于传统的GBDT的一个特征）
- Shrinkage（缩减），相当于学习速率（xgboost中的eta）。xgboost在进行完一次迭代后，会将叶子节点的权重乘上该系数，**主要是为了削弱每棵树的影响，让后面有更大的学习空间**。实际应用中，一般把eta设置得小一点，然后迭代次数设置得大一点。（sklearn中的GBDT的实现也有学习速率）
- **列抽样（column sampling)**,借鉴了随机森林的做法，支持列抽样可以降低过拟合，同时减少了计算量，这也是xgboost异于传统gbdt的一个特性
- 对缺失值的处理。对于样本有特征确实的情况下，xgboost可以自动学习它的分裂方向。
- xgboost工具支持并行。xgboost的并行并不是tree粒度的并行，xgboost也是需要一次迭代完成之后，才能进行下一次迭代的。（第t次迭代的代价函数包含了前面t-1次的预测值）。xgboost的并行是特征粒度上的。决策树的学习最耗时的步骤是就是对特征进行排序（因为要确定最佳的分割点），xgboost在训练之前，预先对数据进行排序，然后保存成block结构，后面的迭代中重复的使用这个结构，大大的减少了计算量。这个结构也使并行成为可能。在进行节点分裂时，需要计算每个特征的信息增益，最终选择增益最大的那个特征去分裂，那么各个特征的增益计算就可以开多线程计算。
- 可并行的近似直方图算法。树节点在进行分裂时，需要计算每个特征的的每个分裂点的信息增益，即用贪心法枚举所有的可能的分割点。当数据无法一次性载入内存或者在分布式的情况下，贪心的算法效率就会变得很低，所以xgboost还提出了一种，可并行的近似直方图算法，用于高效的生成候选的分割点。
xgboost的目标函数 ![](http://ww1.sinaimg.cn/large/9ebd4c2bgy1fw914wiamjj20fc07uaaz.jpg)
- xgboost 分裂节点时所采用的公式
![](http://ww1.sinaimg.cn/large/9ebd4c2bgy1fw91my4kq3j20gm04ygo5.jpg)

##### The correct answer is marked in red. Please consider if this visually seems a reasonable fit to you. The general principle is we want both a simple and predictive model. The tradeoff between the two is also referred as bias-variance tradeoff in machine learning.

![](http://ww1.sinaimg.cn/large/9ebd4c2bgy1fw920cfbyyj20jj0eudhu.jpg)


