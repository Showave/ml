# 评价指标
## TP、FP、TN、FN
### Accuracy、Precision、Recall、F1-Score、ROC和AUC、ROC曲线

## NDCG（Normalized discounted cumulative gain）
> https://cloud.tencent.com/developer/article/1532635

| id            | doc_1 | doc_2 | doc_3 | doc_4 | doc_5 |
| ------------- | ----- | ----- | ----- | ----- | ----- |
| Ground Lable  | 1     | 2     | 0     | 0     | 1     |
| Predict Lable | 2     | 1     | 0     | 1     | 1     |

### CG（cumulative gain），累积获得，指的是网页gain的累加
ground truth：CG=(2^1-1)+(2^2-1)+(2^1-1)+(2^0-1)+(2^0-1)=5 (指数为了**拉开label之间的差异**)

Predict：CG=(2^2-1)+(2^1-1)+(2^0-1)+(2^1-1)+(2^0-1)=5

#### DCG（Discounted Cumulative Gain）DCG对位置做了相应的discount

Predict:  DCG = (2^1-1)/1+(2^2-1)/1.58+ (2^1-1)/2+(2^1-1)/2.32+ (2^0-1)/2.58=1.551

Groundtruth: DCG = (2^2-1)/1+(2^1-1)/1.58+ (2^1-1)/2+(2^0-1)/2.32+ (2^0-1)/2.58=3.148

每个gain后的除数是上图公式的log(i+1)加上位次的数值

#### NDCG 就是被IDEA DCG规格化的值，即DCG/IDCG。整体质量越高的列表NDCG值越大。
NDCG=1.551/3.1486=0.4925

以NDCG为优化目标，保证了搜索引擎在返回结果总体质量好的情况下，把更高质量结果排在更前面。

## MAP（Mean Average Precision）
MAP 是反映系统在全部相关文档上性能的单值指标。系统检索出来的相关文档越靠前(rank越高)，MAP就可能越高

例子：一个主题1有四个相关的doc，分别被排在了1，2，4，7位置。则MAP=(1/1+2/2+3/4+4/7)/4=0.83

## MRR（Mean Reciprocal Rank）
是把标准答案在被评价系统给出结果中的排序取倒数作为它的准确度,**对于top的排序做一个discount,保证头部**。

例子：一个主题1有四个相关的doc，分别被排在了1，2，4，7位置。则MRR=(1/1+1/2+1/4+1/7)/4=0.475。

## HR（Hit Rate）
## 回归指标：MSE、RMSE、MAE、R Squared
## CR用户转化率：用户购买的占点击的比率