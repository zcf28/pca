import numpy as np


"""
PAC(主成分分析)降维 是一种统计方法， 通过正交变换， 将一组可能存在相关性的变量
转换为一组线性不相关的变量， 转换后的变量成为原来变量的主成分
"""


class PCA(object):
    """
    pca 降维顺序
        1 对特征进行标准化
        2 计算协方差矩阵
        3 计算协方差矩阵的特征值和特征向量
        4 选出 最大的 k 个特征值对于的特征向量， 得到特征向量矩阵  k 为最终的维度
        5 将数据降低到 k 维 得到新的特征矩阵
    """

    def __init__(self, matrix_vector, k):
        self.matrix_vector = matrix_vector
        self.k = k

    def get_standard(self):
        """
        功能描述:  1 对特征进行标准化
        :return:
        """
        mean_matrix_vector = np.mean(self.matrix_vector, axis=0)
        return self.matrix_vector - mean_matrix_vector, mean_matrix_vector

    def get_cov_matrix(self):
        """
        功能描述:  2 计算协方差矩阵
        :return:
        """
        standard_matrix_vector, _ = self.get_standard()
        return np.cov(standard_matrix_vector, rowvar=False)

    def get_cov_value_and_vector(self):
        """
        功能描述:  3 计算协方差矩阵的特征值和特征向量
        :return:
        """
        cov_matrix = self.get_cov_matrix()
        cov_value, cov_vector = np.linalg.eig(cov_matrix)
        return cov_value, cov_vector

    def get_feature_vector_matrix(self):
        """
        功能描述:  4 选出 最大的 k 个特征值对于的特征向量， 得到特征向量矩阵  k 为最终的维度
        :return:
        """
        cov_value, cov_vector = self.get_cov_value_and_vector()
        cov_value_sort = np.argsort(cov_value)
        cov_value_top_k = cov_value_sort[:-(self.k + 1):-1]
        return cov_vector[:, cov_value_top_k]

    def get_result(self):
        """
        功能描述:  5 将数据降低到 k 维 得到新的特征矩阵
        :return:
        """
        feature_vector_matrix = self.get_feature_vector_matrix()
        return np.dot(self.matrix_vector, feature_vector_matrix)

    def get_rec_matrix_vector(self):
        """
        功能描述:  由 主成分矩阵 重构结果
        :return:
        """
        result = self.get_result()
        _, mean_matrix_vector = self.get_standard()
        feature_vector_matrix = self.get_feature_vector_matrix()

        return np.mat(result) * feature_vector_matrix.T + mean_matrix_vector

    def get_info_loss(self):
        """
        功能描述:  主成分的方差 占 总方差 的比例 可反映 信息保留比例
        :return:
        """
        cov_value, _ = self.get_cov_value_and_vector()

        return np.sum(cov_value[:self.k])/np.sum(cov_value)


if __name__ == '__main__':
    np.random.seed(42)
    matrix_vector = np.random.rand(4, 8)

    pca = PCA(matrix_vector=matrix_vector, k=4)

    print(matrix_vector, matrix_vector.shape)

    new_matrix_vector = pca.get_result()
    print(new_matrix_vector, new_matrix_vector.shape)

    rec_matrix_vector = pca.get_rec_matrix_vector()
    print(rec_matrix_vector, rec_matrix_vector.shape)

    print(pca.get_info_loss())
