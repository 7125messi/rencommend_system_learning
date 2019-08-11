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
    @timer
    def loadData(self,fp):
        data = []
        for l in open(fp):
            data.append(tuple(map(int,l.strip().split('::')[:2])))
        return data

    @timer
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
            data_dict = [k:list(data_dict[k]) for k in data_dict]
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
    

# 2 算法实现
## ItemCF
## ItemIUF
## temCF_Norm

## 1. 基于物品余弦相似度的推荐
def ItemCF(train, K, N):
    pass

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
    @timer
    def worker(self,train,test):
        """
        :params:train,训练数据集
        :params:test,测试数据集
        :return:各指标的值
        """
        pass

    # 多次实验取平均
    @timer
    def run(self):
        pass

# 1. ItemCF实验
M,N = 8,10
for K in [5,10,20,40,80,160]:
    cf_exp = Experiment(M,K,N,rt='ItemCF')
    cf_exp.run()

# 2. ItemIUF实验
M, N = 8, 10
K = 10 # 与书中保持一致
iuf_exp = Experiment(M, K, N, rt='ItemIUF')
iuf_exp.run()

# 3. ItemCF-Norm实验
M, N = 8, 10
K = 10 # 与书中保持一致
norm_exp = Experiment(M, K, N, rt='ItemCF-Norm')
norm_exp.run()