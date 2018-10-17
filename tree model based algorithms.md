<!-- TOC -->

- [tree based ensemble algorithms](#tree-based-ensemble-algorithms)
- [主要介绍以下几种ensemble的分类器（tree based algorithms）](#%E4%B8%BB%E8%A6%81%E4%BB%8B%E7%BB%8D%E4%BB%A5%E4%B8%8B%E5%87%A0%E7%A7%8Densemble%E7%9A%84%E5%88%86%E7%B1%BB%E5%99%A8tree-based-algorithms)
    - [**xgboost**](#xgboost)
    - [**lightGBM**： **基于决策树算法的分布式梯度提升框架**](#lightgbm-%E5%9F%BA%E4%BA%8E%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E7%9A%84%E5%88%86%E5%B8%83%E5%BC%8F%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87%E6%A1%86%E6%9E%B6)
    - [**GBDT(Gradient Boosting Decison Tree)**](#gbdtgradient-boosting-decison-tree)
    - [**随机森林**](#%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97)
    - [**决策树**](#%E5%86%B3%E7%AD%96%E6%A0%91)

<!-- /TOC -->



### tree based ensemble algorithms 
- 原始的Boost算法是在算法开始的时候，为每个样本赋上一个权重值，初始的时候，每个样本都是同样的重要。在每一步的训练中，得到的模型，会给出每个数据点的估计对错，根据判断的对错，在每一步的训练之后，会增加分错样本的权重，减少分类正确的样本的权重，如果在后续的每一步训练中，如果继续被分错，那么就会被严重的关注，也就是获得了一个比较高的权重。经过N次迭代之后，将会得到N个简单的分类器（base learner），然后将他们组装起来（可以进行加权，或者进行投票），得到一个最终的模型。

### 主要介绍以下几种ensemble的分类器（tree based algorithms）


#### **xgboost**
- xgboost能自动利用cpu的多线程，而且适当改进了gradient boosting，加了剪枝，控制了模型的复杂程度
- 传统的GBDT算法以CART作为基分类器，**xgboost还可以支持线性分类器**，相当于带L1和L2的逻辑斯谛回归或者线性回归
- 传统的GBDT在优化的时候，使用的是一阶导数信息，**xgboost则对代价函数进行了二阶泰勒展开，同时用到了一阶导数和二阶导数**。顺便提一下，xgboost工具支持自定义代价函数，只要函数可一阶和二阶求导。
- xgboost在代价函数中加入了正则项，用于控制模型的复杂度。**正则项里面包括树的叶子节点的个数、每个叶子节点上输出的score的L2模的平方和**。（ 从Bias-variance tradeoff 的角度来说，正则化降低了模型的variance，使得到的模型更加简单，防止过拟合，这是xgboost优于传统的GBDT的一个特征）
- Shrinkage（缩减），相当于学习速率（xgboost中的eta）。xgboost在进行完一次迭代后，会将叶子节点的权重乘上该系数，**主要是为了削弱每棵树的影响，让后面有更大的学习空间**。实际应用中，一般把eta设置得小一点，然后迭代次数设置得大一点。（sklearn中的GBDT的实现也有学习速率）
- **列抽样（column sampling)**,借鉴了随机森林的做法，支持列抽样可以降低过拟合，同时减少了计算量，这也是xgboost异于传统gbdt的一个特性
- 对缺失值的处理。对于样本有特征确实的情况下，xgboost可以自动学习它的分裂方向。
- xgboost工具支持并行。xgboost的并行并不是tree粒度的并行，xgboost也是需要一次迭代完成之后，才能进行下一次迭代的。（第t次迭代的代价函数包含了前面t-1次的预测值）。**xgboost的并行是特征粒度上的**。决策树的学习最耗时的步骤是就是对特征进行排序（因为要确定最佳的分割点），**xgboost在训练之前，预先对数据进行排序，然后保存成block结构，后面的迭代中重复的使用这个结构，大大的减少了计算量**。这个结构也使并行成为可能。在进行节点分裂时，需要计算每个特征的信息增益，最终选择增益最大的那个特征去分裂，那么各个特征的增益计算就可以开多线程计算。
- 可并行的近似直方图算法。树节点在进行分裂时，需要计算每个特征的的每个分裂点的信息增益，即用贪心法枚举所有的可能的分割点。当数据无法一次性载入内存或者在分布式的情况下，贪心的算法效率就会变得很低，所以xgboost还提出了一种，**可并行的近似直方图算法，用于高效的生成候选的分割点**。
- xgboost的目标函数 ![](http://ww1.sinaimg.cn/large/9ebd4c2bgy1fw914wiamjj20fc07uaaz.jpg)
- xgboost 分裂节点时所采用的公式
![](http://ww1.sinaimg.cn/large/9ebd4c2bgy1fw91my4kq3j20gm04ygo5.jpg)
- 这个公式形式上与ID3算法与CART算法是一致的，两者做差，得到某种增益。为了限制树的生长，我们可以加入阈值，**当增益大于阈值时才让节点分裂，上式中的gamma即为阈值，它是正则项里叶子节点数T的系数**，所以xhboost在优化目标函数的同时相当于也做了预剪枝。
- **公式中的系数lambda，是正则项leaf score的L2模平方的系数，对leaf score做了平滑，也起到了防止过拟合的作用**，这还是传统GBDT里不具备的特性。

- 总结下来就是两者的区别：
   * xgboost里面的基学习器除了用tree(gbtree)，也可用线性分类器(gblinear)。而GBDT则特指梯度提升决策树算法。
   * xgboost相对于普通gbm的实现，可能具有以下的一些优势：
   * 显式地将树模型的复杂度作为正则项加在优化目标
   * 公式推导里用到了二阶导数信息，而普通的GBDT只用到一阶
   * 允许使用column(feature) sampling来防止过拟合，借鉴了Random Forest的思想。（sklearn里的gbm好像也有类似实现）
   * 实现了一种分裂节点寻找的近似算法，用于加速和减小内存消耗。
   * 节点分裂算法能自动利用特征的稀疏性。
   * data事先排好序并以block的形式存储，利于并行计算
   * penalty function Omega主要是对树的叶子数和叶子分数做惩罚，这点确保了树的简单性
   * 支持分布式计算可以运行在MPI，YARN上，得益于底层支持容错的分布式通信框架rabit。

![](http://ww1.sinaimg.cn/large/9ebd4c2bgy1fw920cfbyyj20jj0eudhu.jpg)
- The correct answer is marked in red. Please consider if this visually seems a reasonable fit to you. The general principle is we want both a simple and predictive model. The tradeoff between the two is also referred as bias-variance tradeoff in machine learning.

#### **lightGBM**： **基于决策树算法的分布式梯度提升框架**

- lightGBM 与xgboost的区别：
    - xgboost使用的是pre-sorted算法（对所有的特征都按照特征的数值进行预排序，在遍历分割点的时候用O(data)的代价函数找个一个特征的最好分割点，能够更加精确的找到数据的分割点。
    - lightGBM 使用的是histogram算法，占用内存更低，数据分割的复杂度更低。
- 决策树生长策略上
    - xgboost采用的是level-wise生长策略，能够同时分类同一层的叶子，从而进行多线程优化，不容易过拟合，但是不加区分的对待同一层的叶子，带来了很多没有必要的开销(有很多的叶子分裂增益较低，没有必要进行搜索和分裂) 
    - lightGBM采用的是leaf-wise的生长策略，每次从当前的叶子中找到分裂增益最大的（一般也是数据量最大）的一个叶子进行分裂，如此循环；但是生长出的决策树枝叶过多，产生过拟合，lightGBM在leaf-wise上增加了一个最大深度的限制，在保证高效率的同时防止过拟合。
    - 另一个巧妙的优化是histogram做差加速，一个容易观察到的现象：一个叶子的直方图可以由它的父节点的直方图与它兄弟的直方图做差得到。
![](http://ww1.sinaimg.cn/large/9ebd4c2bgy1fw9uxam080j20ey03oq3w.jpg)



#### **GBDT(Gradient Boosting Decison Tree)**
- GBDT中使用的都是回归树，GBDT用来做回归预测，调整后也可以用于分类，设定阈值，大于阈值为正例，反之为负例，可以发现多种有区分性的特征以及特征组合。
- GBDT是把所有树的结论累加起来做最终结论，GBDT的核心就在于，每一棵树学的是之前所有树结论和的残差，这个残差就是把一个加预测值后能得到真实值的累加量。
- 比如A的真实年龄是18岁，但第一棵树的预测年龄是12岁，差了6岁，即残差为6岁。那么在第二棵树里我们把A的年龄设为6岁去学习，如果第二棵树真的能把A分到6岁的叶子节点，那累加两棵树的结论就是A的真实年龄；如果第二棵树的结论是5岁，则A仍然存在1岁的残差，第三棵树里A的年龄就变成1岁，继续学。 Boosting的最大好处在于，每一步的残差计算其实变相地增大了分错instance的权重，而已经分对的instance则都趋向于0。这样后面的树就能越来越专注那些前面被分错的instance。
- 用公式来表示提升树的部分原理 ![](http://ww1.sinaimg.cn/large/9ebd4c2bgy1fwb1vcxifkj20qy0ftq5d.jpg)

- GBDT划分标准默认是friedman_mse可以查看[sklearn 官方文档中GBDT的参数说明 ](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier)

- Gradient Boost与传统的Boost的区别
- 每一次的计算是为了减少上一次的残差(residual)，而为了消除残差，我们可以在残差减少的梯度(Gradient)方向上建立一个新的模型。
- 所以说，在Gradient Boost中，每个新的模型的建立是为了使得之前模型的残差往梯度方向减少。
- Shrinkage（缩减）的思想认为，每次走一小步逐渐逼近结果的效果，要比每次迈一大步很快逼近结果的方式更容易避免过拟合。
- 即它不完全相信每一棵残差树，他认为每棵树只学到了真理的一部分，累加的时候只累加一小部分，每次通过多学几棵树弥补不足。
- 本质上，**Shrinkage为每棵树设置了一个weight，累加时要乘以这个weight，但和Gradient并没有关系**。
- The advantages of GBRT are:

- Natural handling of data of mixed type (= heterogeneous features)
- 可以处理不同性质的属性，数值特征与category特征，
- 数值特征需要进行数据的预处理
- Predictive power
- Robustness to outliers in output space (via robust loss functions)
  
The disadvantages of GBRT are:

- Scalability, due to the sequential nature of boosting it can hardly be parallelized.  
- Boost是一个串行过程，不好并行化，**而且计算复杂度高，同时不太适合高维稀疏特征。**

#### **随机森林**
 - 是随机的方式建立一个森林，森林里面有很多的决策树组成，随机森林的每一决策树质检是没有关联的。在得到随机森林之后，当有一个新的样本输进的时候，就让森林中的每一棵决策树进行判断，判断样本属于哪一类，然后看哪一类被选择最多，就预测这个样本为这一类。
- 随机采样
    - 随机行采样，采用有放回的方式，也就是在采样得到的样本集合中，可能有重复的样本。
    假如输入的样本为N个，那么采样的样本也是N个。这使得在训练的时候，每棵树的输入的样本都不是全部的样本，使得相对不容易出现over-fitting。

    - 随机列采样，从M个feature中，选择m个(m<<M)。
    对采样之后的数据使用完全分裂的方式建立决策树，这样的决策树的某个叶子节点要么无法继续分裂，要么里面所有的样本都是指向的同一分类。
- 随机森林 中同样有剪枝，限制决策树的最大深度，以及最小的样本分裂，最小的节点样本数目，样本分裂节点的信息增益或gini系数必须达到的阈值
- 随机森林用于分类的话，划分标准是entropy或者gini系数
- 随机森林用于回归的话，划分标准是mse(mean squared error)或者mae(mean absolute error)

#### **决策树**

- ID3 信息增益：熵（数据的不确定性程度）的减少；一个属性的信息增益量越大，这个属性作为一棵树的根节点就能使这棵树更简洁。
信息增益=分裂前的熵 – 分裂后的熵
面对类别较少的离散数据时效果较好，但如果面对连续的数据（如体重、身高、年龄、距离等），或者每列数据没有明显的类别之分（最极端的例子的该列所有数据都独一无二），即每个值对应一类样本
- C4.5信息增益比：克服了ID3用信息增益选择属性时偏向选择取值多的属性的不足（**某个属性存在大量的不同值，在划分时将每个值分为一个结点**）

- CART 使用基尼系数进行分类
基尼指数Gini(D)表示集合D的不确定性，基尼指数Gini(D,A)表示经A＝a分割后集合D的不确定性。基尼指数值越大，样本集合的不确定性也就越大，这一点与熵相似。
![](http://ww1.sinaimg.cn/mw690/9ebd4c2bgy1fwa5y85jk5j20cl03smxh.jpg)
![](http://ww1.sinaimg.cn/mw690/9ebd4c2bgy1fwa5v5nbrsj20g9040dge.jpg)
- 分类与回归树（CART）:二叉树形式，分类时：**根据Gini指数选择划分特征**
- 回归时：Los为 **平方损失函数**，最小化均方误差选择划分特征，切分点（值）将数据切分成两部分，用平方误差最小的准则（最小二乘法）求解每个单元上的最优输出值（**每个叶子节点上的预测值为所有样本的平均值**）。
![](http://ww1.sinaimg.cn/mw690/9ebd4c2bgy1fwa63i1g2oj20gg02omxi.jpg)
    用选定的对（j,s）划分区域并决定相应的输出值，每个叶子节点上的预测值为所有样本的平均值：
![](http://ww1.sinaimg.cn/mw690/9ebd4c2bgy1fwa65n9f8uj20ge03rmxu.jpg)

    [可以参看该repository中的另外文章中介绍的回归树](https://github.com/point6013/essay/blob/master/GBDT.md#dt-%E5%9B%9E%E5%BD%92%E6%A0%91-regression-decision-tree)


- 决策树的生成通常使用 **信息增益最大、信息增益比最大或基尼指数最小**作为特征选择的准则。
