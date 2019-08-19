推荐系统中的评价指标真的是五花八门，今天我们就来系统的总结一下，这些指标有的**适用于二分类问题**，有的**适用于对推荐列表topk的评价**。

# 1、精确率、召回率、F1值

我们首先来看一下混淆矩阵，对于二分类问题，真实的样本标签有两类，我们学习器预测的类别有两类，那么根据二者的类别组合可以划分为四组，如下表所示：
![img](img/1.png)
上表即为混淆矩阵，其中，行表示预测的label值，列表示真实label值。TP，FP，FN，TN分别表示如下意思：
- TP（true positive）：表示样本的真实类别为正，最后预测得到的结果也为正；
- FP（false positive）：表示样本的真实类别为负，最后预测得到的结果却为正；
- FN（false negative）：表示样本的真实类别为正，最后预测得到的结果却为负；
- TN（true negative）：表示样本的真实类别为负，最后预测得到的结果也为负.

可以看到，TP和TN是我们预测准确的样本，而FP和FN为我们预测错误的样本。

基于混淆矩阵，我们可以得到如下的评测指标：

## 准确率

准确率表示的是分类正确的样本数占样本总数的比例，假设我们预测了10条样本，有8条的预测正确，那么准确率即为80%。
用混淆矩阵计算的话，准确率可以表示为：

$$
\text {Accuracy}=\frac{T P+T N}{T P+F P+T N+F N}
$$

## 精确率／召回率

**精确率**表示预测结果中，预测为正样本的样本中，正确预测为正样本的概率；
**召回率**表示在原始样本的正样本中，最后被正确预测为正样本的概率；

二者用混淆矩阵计算如下：

$$
\text { Precision }=\frac{T P}{T P+F P}
$$

$$
\text {Recall}=\frac{T P}{T P+F N}
$$

## F1值

为了折中精确率和召回率的结果，我们又引入了F-1 Score，计算公式如下：

$$
F1-score=\frac{2\times recall\times precision}{recall+precision}
$$

# 2、AUC

AUC定义为ROC曲线下方的面积：

ROC曲线的横轴为“假正例率”（False Positive Rate,FPR)，又称为“假阳率”；纵轴为“真正例率”(True Positive Rate,TPR)，又称为“真阳率”，

假阳率，简单通俗来理解就是预测为正样本但是预测错了的可能性，显然，我们不希望该指标太高。
$$
FPR=\frac{FP}{TN+FP}
$$
真阳率，则是代表预测为正样本但是预测对了的可能性，当然，我们希望真阳率越高越好。
$$
TPR=\frac{TP}{TP+FN}
$$
下图就是我们绘制的一张ROC曲线图，曲线下方的面积即为AUC的值：
![img](img/2.png)
**AUC还有另一种解释，就是测试任意给一个正类样本和一个负类样本，正类样本的score有多大的概率大于负类样本的score。**

# 3、Hit Ratio(HR)

**在top-K推荐中，HR是一种常用的衡量召回率的指标**，其计算公式如下：

$$
HR@K=\frac{\text {NumberofHits} @K}{|GT|}
$$

分母是所有的测试集合，分子式每个用户top-K推荐列表中属于测试集合的个数的总和。举个简单的例子，三个用户在测试集中的商品个数分别是10，12，8，模型得到的top-10推荐列表中，分别有6个，5个，4个在测试集中，那么此时HR的值是 (6+5+4)/(10+12+8) = 0.5。

# 4、Mean Average Precision(MAP)

在了解MAP(Mean Average Precision)之前，先来看一下**AP(Average Precision), 即为平均准确率。**

对于AP可以用这种方式理解: 假使当我们使用google搜索某个关键词，返回了10个结果。当然最好的情况是这10个结果都是我们想要的相关信息。但是假如只有部分是相关的，比如5个，那么这5个结果如果被显示的比较靠前也是一个相对不错的结果。但是如果这个5个相关信息从第6个返回结果才开始出现，那么这种情况便是比较差的。这便是**AP所反映的指标，与recall的概念有些类似，不过是“顺序敏感的recall”。**

比如对于用户u, 我们给他推荐一些物品，那么u的平均准确率定义为：

![img](img/3.png)

用一个例子来解释AP的计算过程：

![img](img/4.png)

因此该user的AP为（1 + 0.66 + 0.5） ／ 3 = 0.72

那么**对于MAP(Mean Average Precision)，就很容易知道即为所有用户u的AP再取均值(mean)而已**。那么计算公式如下：

$$
MAP=\frac{\sum_{u \in \mathcal{V}} \operatorname{te} AP_{u}}{\left|\mathcal{V}^{ie}\right|}
$$

# 5、Normalized Discounted Cummulative Gain(NDCG)

对于NDCG，我们需要一步步揭开其神秘的面纱，先从CG说起：

## CG

我们先从**CG(Cummulative Gain)**说起, 直接翻译的话叫做“**累计增益**”。 **在推荐系统中，CG即将每个推荐结果相关性(relevance)的分值累加后作为整个推荐列表(list)的得分。**即

$$
CG_{k}=\sum_{i=1}^{k}rel_{i}
$$

这里，**rel-i 表示处于位置 i 的推荐结果的相关性，k 表示所要考察的推荐列表的大小**。

## DCG

CG的一个缺点是**没有考虑每个推荐结果处于不同位置对整个推荐效果的影响**，例如**我们总是希望相关性高的结果应排在前面**。显然，如果相关性低的结果排在靠前的位置会严重影响用户体验， 所以在**CG的基础上引入位置影响因素**，即**DCG(Discounted Cummulative Gain)**, “Discounted”有打折，折扣的意思，这里指的是**对于排名靠后推荐结果的推荐效果进行“打折处理”**:

$$
DCG_{k}=\sum_{i=1}^{k} \frac{2^{rel_{i}}-1}{\log_{2}(i+1)}
$$

从上面的式子可以得到两个结论：
- 1）**推荐结果的相关性越大，DCG越大。**
- 2）**相关性好的排在推荐列表的前面的话，推荐效果越好，DCG越大。**

## NDCG

**DCG仍然有其局限之处，即不同的推荐列表之间，很难进行横向的评估**。而我们 **<font color= blue>评估一个推荐系统，不可能仅使用一个用户的推荐列表及相应结果进行评估， 而是对整个测试集中的用户及其推荐列表结果进行评估</font>**。 那么 **<font color= red>不同用户的推荐列表的评估分数就需要进行归一化，也即NDCG(Normalized Discounted Cummulative Gain)。</font>**

在介绍NDCG之前，还需要了解一个概念：IDCG. IDCG, 即Ideal DCG， 指推荐系统为某一用户返回的最好推荐结果列表， 即**假设返回结果按照相关性排序， 最相关的结果放在最前面， 此序列的DCG为IDCG**。因此**DCG的值介于 (0,IDCG] ，故NDCG的值介于(0,1]，那么用户u的NDCG@K定义为**：

$$
NDCG_{u}@k=\frac{DCG_{u}@k}{IDCG_{u}}
$$

因此，平均NDCG计算为：

$$
NDCG@k=\frac{\sum_{u\in\mathcal{V}}\operatorname{te}NDCG_{u}@k}{\left|\mathcal{V}^{te}\right|}
$$

## NDCG的完整案例

看了上面的介绍，是不是感觉还是一头雾水，不要紧张，我们通过一个案例来具体介绍一下。

假设在Baidu搜索到一个词，得到5个结果。我们对这些**结果进行3个等级的分区，对应的分值分别是3、2、1，等级越高，表示相关性越高**。
**假设这5个结果的分值分别是3、1、2、3、2。**

因此CG的计算结果为3+1+2+3+2 = 11。DCG的值为6.69，具体见下表：

![img](img/5.png)

理想状况下，我们的IDCG排序结果的相关性应该是3，3，2，2，1，因此IDCG为7.14(具体过程不再给出)，因此NDCG结果为6.69/7.14 = 0.94。

# 6、Mean Reciprocal Rank (MRR)

MRR计算公式如下：

$$
\mathrm{MRR}=\frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\mathrm{rank}_{i}}
$$

其中|Q|是用户的个数，ranki是对于第i个用户，推荐列表中第一个在ground-truth结果中的item所在的排列位置。

举个例子，有三个用户，推荐列表中正例的最小rank值分别为3，2，1，那么MRR=(1 + 0.5 + 0.33) / 3 = 0.61

# 7、ILS

ILS是衡量推荐列表多样性的指标，计算公式如下：

$$
ILS(\mathrm{L})=\frac{\sum_{b_{i} \in \mathrm{L}} \sum_{b_{j} \in \mathrm{L}, b_{j} \neq b_{i}} S\left(b_{i}, b_{j}\right)}{\sum_{b_{i} \in \mathrm{L}} \sum_{b_{j} \in \mathrm{L}, b_{j} \neq b_{i}} 1}
$$

如果S(bi,bj)计算的是i和j两个物品的相似性，如果推荐列表中物品越不相似，ILS越小，那么推荐结果的多样性越好。