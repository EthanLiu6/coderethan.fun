# 03-Bert理解

## 1.预训练

- masked language model 用来提取token向量
- next sentence prediction 用来提取文本向量

![image-20250327181838063](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250327181838063.png)

![image-20250327183133577](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250327183133577.png)

input 开头加cls，拼接第二个文本，中间和结尾加sep，cls可以提取这两个文本的语意连续性或者相关性

## 2. embedding

1. word/token emb有点像word2vocter

- 对除特殊token外的特征向量的15%做随机替换这其中的：
  - 80%做mask
  - 10%随机选一个
  - 10%不变
- 对这15%的做预测，得到mask的预测结果