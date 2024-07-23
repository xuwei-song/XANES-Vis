#!/usr/bin/env python
# coding: utf-8
# %%
# #!/usr/bin/env python
# coding: utf-8

import numpy as np

def read_and_process_dat_file(filename, delimiter=' ', normalization_threshold=6585):
    """
    读取.dat文件，将数据转换为2D数组，进行最大值归一化，并进行阈值截断。

    参数:
    filename (str): .dat文件路径。
    delimiter (str): 文件中的数据分隔符。默认为空格。
    normalization_threshold (float): 用于截断数据的阈值。默认值为6585。

    返回:
    numpy.ndarray: 处理后的2D数组。
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        data_list = [line.strip().split(delimiter) for line in lines if line.strip()]
        data_array = np.array([[float(num) for num in row] for row in data_list])

    max_vals = np.max(data_array[:, 1:], axis=0)
    data_array[:, 1:] = data_array[:, 1:] / max_vals

    data_array = data_array[data_array[:, 0] <= normalization_threshold]

    return data_array

# 示例使用
file1 = 'MnCO3.dat'
file2 = 'Mn5O8.dat'  # 替换为你的.dat文件路径
file3 = 'MnO2.dat'
data = read_and_process_dat_file(file1)
reference_Mn5O8 = read_and_process_dat_file(file2)
reference_MnO2  = read_and_process_dat_file(file3)

# 打印数据形状（行数和列数）
print(f"The data has {data.shape[0]} rows and {data.shape[1]} columns.")
print(f"The reference_Mn5O8 has {reference_Mn5O8.shape[0]} rows and {reference_Mn5O8.shape[1]} columns.")
print(f"The reference_MnO2 has {reference_MnO2.shape[0]} rows and {reference_MnO2.shape[1]} columns.")

# 绘制插值后的曲线
# plt.figure(figsize=(10, 6))
# plt.plot(reference_Mn5O8[:,0], reference_Mn5O8[:,1], label='reference_Mn5O8 Data')
# plt.plot(reference_MnO2[:,0], reference_MnO2[:,1], label='reference_MnO2 Data')
# plt.xlabel('Energy')
# plt.ylabel('Normalized I0')
# plt.legend()
# plt.show()

# %%
def define_references(data):
    """
    定义参考光谱。

    参数:
    data (numpy.ndarray): 处理后的数据。

    返回:
    tuple: 包含两个参考光谱的元组。
    """
    reference_MnCO3 = data[:, :2]  # 取第一张作为reference_MnCO3
    reference_Mn2O3 = np.column_stack((data[:, 0], data[:, -1]))  # 取最后一张作为reference_Mn2O3
    return reference_MnCO3, reference_Mn2O3

reference_MnCO3, reference_Mn2O3 = define_references(data)

# %%
from sklearn.decomposition import PCA

def pca_reduction(X):
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X.T).T
    print(X_reduced.shape)
    return X_reduced

X = data[:, 1:]
X_reduced = pca_reduction(X).T

# %%
from sklearn.cluster import KMeans

def kmeans_clustering(X_reduced, n_clusters=4):
    kmeans = KMeans(n_clusters, random_state=0)
    labels = kmeans.fit_predict(X_reduced)
    return labels

labels = kmeans_clustering(X_reduced)

# %%
import numpy as np
from scipy.optimize import least_squares

def ls_linear_combination_fit(data, references):
    def residuals(coeffs, data, references):
        fit = sum(c * ref for c, ref in zip(coeffs, references))
        penalty = 1e3 * (np.sum(coeffs) - 1)
        return np.append(data - fit, penalty)

    initial_guess = np.ones(len(references)) / len(references)
    bounds = (0, 1)
    result = least_squares(residuals, initial_guess, bounds=bounds, args=(data, references))

    fit = sum(c * ref for c, ref in zip(result.x, references))
    rss = np.sum((data - fit) ** 2)

    return result.x, rss

def process_cluster_means(cluster_means, references):
    coefficients = []
    rss_values = []
    for i in range(len(cluster_means)):
        coeffs, rss = ls_linear_combination_fit(cluster_means[i], references)
        coefficients.append(coeffs)
        rss_values.append(rss)

    coefficients = np.array(coefficients)
    R_values = np.array(rss_values)
    return coefficients, R_values

references = [reference_MnCO3[:,1], reference_MnO2[:,1], reference_Mn2O3[:,1]]
cluster_means = np.array([X[:, labels == label].mean(axis=1) for label in np.unique(labels)])
coefficients, R_values = process_cluster_means(cluster_means, references)

# %%
import matplotlib.pyplot as plt

def plot_results(coefficients, R_values, component_names):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    bar_width = 0.2
    r = np.arange(len(coefficients))  # 聚类数目

    for i in range(coefficients.shape[1]):  # 参考光谱数目
        ax1.bar(r + i * bar_width, coefficients[:, i] * 100, width=bar_width, label=component_names[i])

    ax1.set_xlabel('Clusters', fontweight='bold')
    ax1.set_ylabel('%', fontweight='bold')
    ax1.set_xticks([r + bar_width * (coefficients.shape[1] - 1) / 2 for r in range(len(coefficients))])
    ax1.set_xticklabels([f'Cl. {i+1}' for i in range(len(coefficients))])

    ax2 = ax1.twinx()
    ax2.plot(r, R_values, color='orange', marker='o', label='R')

    ax2.set_ylabel('R', fontweight='bold')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()
    
component_names = ['MnCO3', 'MnO2', 'Mn2O3']
plot_results(coefficients, R_values, component_names)

# %%
