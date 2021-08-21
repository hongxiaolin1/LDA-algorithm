import numpy as np
import pandas as pd

# ======================= 林应昕 ========================
# ===================== 第2次作业 =======================
# ==================== 2021/04/27 ======================

# 定义 LDA 函数
# 第一类：正例；第二类：反例
# x1：第一类的特征；x2：第二类的特征
# x1 = obs_num * Feature_num = [x11';x12';···;x1N']
# x11(类别为1的第一个样品，是一个列向量) =  Feature_num * 1 = [Feature1;Feature2]
def lda(x1, x2, show_predict = True):

    miu1 = np.mean(x1, axis=0)  # 第一类的均值向量
    miu2 = np.mean(x2, axis=0)  # 第二类的均值向量

    x1_scale = x1 - miu1  # 对特征数据集做中心化
    x2_scale = x2 - miu2

    n1, _ = np.shape(x1_scale)  # 获得每一类的样品数量
    n2, _ = np.shape(x2_scale)

    sigma1 = np.dot(x1_scale.T, x1_scale)/n1  # 估计每一类的协方差矩阵 Σ1 和 Σ2
    sigma2 = np.dot(x2_scale.T, x2_scale)/n2

    Sw = sigma1 + sigma2  # 计算类内散度矩阵 Sw


    w = np.dot(np.linalg.inv(Sw), miu1 - miu2)  # 计算最优的 w

    if show_predict:  # 是否输出判别结果

        y_hat = np.dot(np.r_[x1, x2], w)  # 所有样例降至 1 维后的 预测值
        y1_hat = np.inner(w, miu1)  # 第一类中心点降至 1 维后的 预测值
        y2_hat = np.inner(w, miu2)  # 第二类中心点降至 1 维后的 预测值
        class_raw = np.append(np.ones(n1, dtype=bool), np.zeros(n2, dtype=bool))  # 真实的类别
        distance1 = np.abs(y_hat - y1_hat)  # 全部样例到 中心1 的距离
        distance2 = np.abs(y_hat - y2_hat)  # 全部样例到 中心2 的距离
        class_pre = (distance1 - distance2) <= 0  # 把到中心1近的样例 判别为正例
        sucess_num = np.sum((class_pre == class_raw))  # 成功判别的数量
        sucess_rate = sucess_num / (n1 + n2) * 100  # 成功判别的比例


        result = {'样例编号':np.arange(1, (n1 + n2 + 1) ),'真实的类别':class_raw, 'LDA的判别结果':class_pre}
        result = pd.DataFrame(result)
        result['判断正确与否'] = "否"
        result.loc[(result['真实的类别'] == result['LDA的判别结果']), '判断正确与否'] = "是"

        print('一共有{}个样例被成功分类，占比为{}%\n'.format(sucess_num, sucess_rate))
        print('系数w为：')
        print(w)

        return w, result
    else:
        return w




# 读入数据
data = pd.read_table("3.0α.txt", delimiter=',')
x1 = np.array(data[data['好瓜'] == '是'][['密度', '含糖率']])
x2 = np.array(data[data['好瓜'] == '否'][['密度', '含糖率']])
d = np.array(data[['密度', '含糖率']])

# 调用 LDA 进行判断

w, res = lda(x1, x2)
print('LDA的判别结果：')
print(res)

# 画图
'''import seaborn as sns
import matplotlib.pyplot as plt

sns.set()  # 使用默认配色
sns.pairplot(data[['密度', '含糖率', '好瓜']], hue="好瓜")  # hue：选择分类列
plt.show()'''
