# 导入包
import random
import math
import time
from tqdm import tqdm

# 1 通用函数定义

## 定义装饰器，监控运行时间
def timmer(func):
    def wrapper(*args,**kargs):
        start_time = time.time()
        res = func(*args,**kargs)
        stop_time = time.time()
        print('Func {0},run time:{1}'.format(func.__name__,stop_time - start_time))
        return res
    return wrapper

## 1. 数据处理相关
### load data
### split data

class Dataset():
    def __init__(self,fp):
        # fp:data file path
        self.data = self.loadData(fp)
    @timmer
    def loadData(self,fp):
        data = []
        for l in open(fp):
            data.append(tuple(map(int,l.strip().split('::')[:2])))
        return data

    @timmer
    def splitData(self,M,k,seed=1):
        """
        :params:data,加载的所有(user,item)数据条目
        :params:M,划分的数目，最后需要取M折的平均
        :params:k,本次是第几次划分，k-[0,M)
        :params:seed,random的种子数，对于不同的k应该设置成一样的
        :return:train,test
        """
        train,test = [],[]
        random.seed(seed)
        for user,item in self.data:
            if random.randint(0,M-1) == k:
                test.append((user,item))
            else:
                train.append((user,item))
        
        # 处理成字典的形式，user->set(items)
        def convert_dict(data):
            data_dict = {}
            for user,item in data:
                if user not in data_dict:
                    data_dict[user] = set()
                data_dict[user].add(item)
            data_dict = {k: list(data_dict[k]) for k in data_dict}
            return data_dict

        return convert_dict(train),convert_dict(test)

## 2. 评价指标
### Precision
### Recall
### Coverage
### Popularity(Novelty)

class Metric():
    def __init__(self,train,test,GetRecommendation):
        """
        :params:train,训练数据
        :params:test,测试数据
        :params:GetRecommendation,为某个用户获取推荐物品的接口函数
        """
        self.train = train
        self.test = test
        self.GetRecommendation = GetRecommendation
        self.recs = self.getRec()

    # 为test中的每个用户进行推荐
    def getRec(self):
        recs = {}
        for user in self.test:
            rank = self.GetRecommendation(user)
            recs[user] = rank
        return recs

    # 定义精确率指标计算方法
    def precision(self):
        all, hit = 0, 0
        for user in self.test:
            test_items = set(self.test[user])
            rank = self.recs[user]
            for item, score in rank:
                if item in test_items:
                    hit += 1
            all += len(rank)
        return round(hit / all * 100, 2)

    # 定义召回率指标计算方式
    def recall(self):
        all, hit = 0, 0
        for user in self.test:
            test_items = set(self.test[user])
            rank = self.recs[user]
            for item, score in rank:
                if item in test_items:
                    hit += 1
            all += len(test_items)
        return round(hit / all * 100, 2)

    # 定义覆盖度指标计算方式
    def coverage(self):
        all_item,recom_item = set(),set()
        for user in self.test:
            for item in self.train[user]:
                all_item.add(item)
            rank = self.recs[user]
            for item, score in rank:
                recom_item.add(item)
        return round(len(recom_item) / len(all_item) * 100, 2)

    # 定义新颖度指标计算方式
    def popularity(self):
        # 计算物品的流行度
        item_pop = {}
        for user in self.train:
            for item in self.train[user]:
                if item not in item_pop:
                    item_pop[item] = 0
                item_pop[item] += 1
        
        num, pop = 0, 0
        for user in self.test:
            rank = self.recs[user]
            for item, score in rank:
                # 取对数，防止因长尾问题带来的被流行物品所主导
                pop += math.log(1 + item_pop[item])
                num += 1
        return round(pop / num,6)

    def eval(self):
        metrics = {
            'Precision':self.precision(),
            'Recall':self.recall(),
            'Coverage':self.coverage(),
            'Popularity':self.popularity()
        }
        print('Metric:', metrics)
        return metrics
# 2 算法实现
## ItemCF
## ItemIUF
## temCF_Norm

## 1. 基于物品余弦相似度的推荐
def ItemCF(train, K, N):
    '''
    :params: train, 训练数据集
    :params: K, 超参数，设置取TopK相似物品数目
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    '''
    # 计算物品相似度矩阵
    sim = {}
    num = {}
    for user in train:
        items = train[user]
        for i in range(len(items)):
            u = items[i]
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for j in range(len(items)):
                if j == i: continue
                v = items[j]
                if v not in sim[u]:
                    sim[u][v] = 0
                sim[u][v] += 1
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])
    
    # 按照相似度排序
    sorted_item_sim = {k: list(sorted(v.items(), \
                                key=lambda x: x[1], reverse=True)) \
                        for k, v in sim.items()}
    
    # 获取接口函数
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        for item in train[user]:
            for u, _ in sorted_item_sim[item][:K]:
                if u not in seen_items:
                    if u not in items:
                        items[u] = 0
                    items[u] += sim[item][u]
        recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs
    
    return GetRecommendation

## 2. 基于改进的物品余弦相似度的推荐
def ItemIUF(train, K, N):
    pass

## 3. 基于归一化的物品余弦相似度的推荐
def ItemCF_Norm(train, K, N):
    pass

# 3 实验

## ItemCF实验，K=[5, 10, 20, 40, 80, 160]
## ItemIUF实验, K=10
## ItemCF-Norm实验，K=10

## M=8, N=10
class Experiment():
    def __init__(self,M,K,N,fp = '../dataset/ml-1m/ratings.dat',rt = 'ItemCF'):
        """
        :params:M,进行多少次实验
        :params:K,TopK相似物品的个数
        :params:N,TopN推荐物品的个数
        :params:fp,数据文件路径
        :params:rt,推荐算法类型
        """
        self.M = M
        self.K = K
        self.N = N
        self.fp = fp
        self.rt = rt
        self.alg = {
            'ItemCF':ItemCF,
            'ItemIUF':ItemIUF,
            'ItemCF_Norm':ItemCF_Norm
        }

    # 定义单次实验
    @timmer
    def worker(self,train,test):
        """
        :params:train,训练数据集
        :params:test,测试数据集
        :return:各指标的值
        """
        getRecommendation = self.alg[self.rt](train,self.K, self.N)
        metric = Metric(train,test,getRecommendation)
        return metric.eval()

    # 多次实验取平均
    @timmer
    def run(self):
        metrics = {
            'Precision':0,
            'Recall':0,
            'Coverage':0,
            'Popularity':0
        }
        dataset = Dataset(self.fp)
        for ii in range(self.M):
            train,test = dataset.splitData(self.M,ii)
            print('Experiment {}:'.format(ii))
            metric = self.worker(train,test)
            metrics = {
                k:metrics[k] + metric[k] for k in metrics
            }
        metrics = {
            k:metrics[k] / self.M for k in metrics
        }
        print('Average Result (M={}, K={}, N={}):{}'.format(self.M,self.K,self.N,metrics))

# 1. ItemCF实验
M,N = 8,10
for K in [5,10,20,40,80,160]:
    cf_exp = Experiment(M,K,N,rt='ItemCF')
    cf_exp.run()

# # 2. ItemIUF实验
# M, N = 8, 10
# K = 10 # 与书中保持一致
# iuf_exp = Experiment(M, K, N, rt='ItemIUF')
# iuf_exp.run()

# # 3. ItemCF-Norm实验
# M, N = 8, 10
# K = 10 # 与书中保持一致
# norm_exp = Experiment(M, K, N, rt='ItemCF-Norm')
# norm_exp.run()