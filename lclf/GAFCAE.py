# %%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Reshape, Conv1DTranspose
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import *


# %%
def autoencoderConv1D(input_shape=(360, 1), filters=[32, 64, 128, 20]):
    input_img = Input(shape=input_shape)
    
    # Encoder
    x = Conv1D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1')(input_img)
    x = Conv1D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(x)
    x = Conv1D(filters[2], 5, strides=2, padding='same', activation='relu', name='conv3')(x)
    
    conv_shape = x.shape[1:]
    flatten_dim = np.prod(conv_shape)
    
    x = Flatten()(x)
    encoded = Dense(units=filters[3], name='embedding')(x)
    
    # Decoder
    x = Dense(units=flatten_dim, activation='relu')(encoded)
    x = Reshape(conv_shape)(x)
    
    x = Conv1DTranspose(filters[1], 5, strides=2, padding='same', activation='relu', name='deconv2')(x)
    x = Conv1DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv1')(x)
    decoded = Conv1DTranspose(1, 5, strides=2, padding='same', name='output')(x)
    
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

# 创建模型
autoencoder, encoder = autoencoderConv1D(input_shape=(360, 1))
autoencoder.summary()

# 编译模型
autoencoder.compile(optimizer='adam', loss='mse')

# %%
file1 = 'MnCO3.dat'  # 替换为你的.dat文件路径
data = read_and_process_dat_file(file1)
data = data[:, 1:]
data_reshaped = data.T.reshape(140, 360, 1)

# 数据集划分为训练集和验证集
X_train, X_val = train_test_split(data_reshaped, test_size=0.2, random_state=42)

# %%
# 训练模型
history = autoencoder.fit(X_train, X_train, 
                          epochs=100, 
                          batch_size=16, 
                          shuffle=True, 
                          validation_data=(X_val, X_val))

# 绘制训练和验证损失值
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# 保存权重
autoencoder.save_weights('conv_ae_weights.weights.h5')

# 加载权重
autoencoder.load_weights('conv_ae_weights.weights.h5')

# %%
import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=tf.float32, shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim),
                                        initializer='glorot_uniform',
                                        name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

ym = encoder.predict(data_reshaped)


# %%
_ = autoencoder.predict(data_reshaped)
# Reshape the data to (140, 360) if needed
_ = _.reshape(-1, 360)
print(_.shape)

# %%
import matplotlib.pyplot as plt
import numpy as np
# Create a new figure
plt.figure(figsize=(10, 8))

# Plot each spectrum
for i in range(data_reshaped.shape[0]):
    plt.plot(_[i], label=f'Spectrum {i+1}')

# Add title and labels
plt.title('Generative Spectra Visualization')
plt.xlabel('Data Points')
plt.ylabel('Amplitude')

# Optionally, add a legend (if you want to identify individual spectra, this can be crowded)
# plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

# Show the plot
plt.show()

# %%
# Create a new figure
plt.figure(figsize=(10, 8))

# Plot each spectrum
for i in range(data_reshaped.shape[0]):
    plt.plot(data_reshaped[i], label=f'Spectrum {i+1}')

# Add title and labels
plt.title('140 Spectra Visualization')
plt.xlabel('Data Points')
plt.ylabel('Amplitude')

# Optionally, add a legend (if you want to identify individual spectra, this can be crowded)
# plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

# Show the plot
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# 计算每个样本的MSE
mse = np.mean((_ - data_reshaped.reshape(_.shape))**2, axis=1)

# 找到差异最大和最小的10个样本的索引
max_diff_indices = np.argsort(mse)[-10:]  # 差异最大的10个样本
min_diff_indices = np.argsort(mse)[:10]   # 差异最小的10个样本

# 可视化差异最大和最小的10个样本
def plot_differences(indices, title):
    plt.figure(figsize=(15, 20))
    for i, idx in enumerate(indices):
        plt.subplot(10, 2, 2 * i + 1)
        plt.plot(_[idx], label='Original')
        plt.plot(data_reshaped[idx].flatten(), label='Reconstructed')
        plt.title(f'Sample {idx} - MSE: {mse[idx]:.4f}')
        plt.legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# 可视化差异最大的10个样本
plot_differences(max_diff_indices, 'Top 10 Samples with Maximum Differences')

# %%
# 可视化差异最小的10个样本
plot_differences(min_diff_indices, 'Top 10 Samples with Minimum Differences')

# %%
## how many clusters
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

sil = []
kmax = 10

# # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters = k,max_iter=10).fit(ym)
    labels = kmeans.labels_
    sil.append(silhouette_score(ym, labels, metric = 'l1'))
    print(str(k)+' - '+ str(sil[k-2]))
#
plt.plot(range(2, kmax+1),sil,'o-')
plt.show

# %%
n_clusters = 5 # obtained by seperately minimizing
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output) # encoder.output是静态张量定义
model = Model(inputs=encoder.input, outputs=clustering_layer)
model.compile(optimizer='adam', loss='kld')

kmeans = KMeans(n_clusters=n_clusters, n_init=20)

# %%
y_pred = kmeans.fit_predict(ym)
y_pred_last = np.copy(y_pred)

model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])


loss = 0
index = 0

update_interval = 14
index_array = np.arange(data_reshaped.shape[0])

# %%
tol = 1 # tolerance threshold to stop training

# %% md
### Start training

# %%
# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T



maxiter = 2000
batch_size = 16
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(data_reshaped, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        print('Iter %d' % (ite))
        print(y_pred)
          # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, data_reshaped.shape[0])]
    loss = model.train_on_batch(x=data_reshaped[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= data_reshaped.shape[0] else 0

model.save_weights('conv_DEC_model_final.weights.h5')

# %% md
### Load the clustering model trained weights

# %%
model.load_weights('conv_DEC_model_final.weights.h5')

# %% md
### Final Evaluation

# %%
## Eval.
q = model.predict(data_reshaped, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
y_pred = q.argmax(1)

import pandas as pd

# 假设 y_pred 和 y_pred_last 是 numpy 数组或可以转换为 DataFrame 的数据结构
data = {
    'y_pred': y_pred.flatten(),  # 如果是多维数组，可以用 flatten() 转换为一维
    'y_pred_last': y_pred_last.flatten()
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 保存为 CSV 文件
df.to_csv("results1000.csv", index=False)

# %%
import numpy as np
import matplotlib.pyplot as plt

num_samples = len(y_pred)
unique_labels = np.unique(y_pred)
num_clusters = len(unique_labels)

# 创建一个映射字典，将原始的簇类标签映射到连续的整数
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
y_pred = np.array([label_mapping[int(label)] for label in y_pred])

# 设置网格的行和列为固定的10x10
grid_rows = 10
grid_cols = 10

# 创建一个映射标签到颜色的字典
def get_color_map(unique_labels):
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    return color_map

color_map = get_color_map(unique_labels)

# 创建方格图
plt.figure(figsize=(10, 10))

# 绘制每个时间序列的方格，根据聚类标签上色
for i in range(num_samples):
    label_color = color_map[y_pred[i]]
    row = i // grid_cols
    col = i % grid_cols
    plt.text(col + 0.5, row + 0.5, str(i + 1), ha='center', va='center', fontsize=8, color='black')
    plt.fill_between([col, col + 1], row, row + 1, color=label_color)

# 绘制未填充的方格，使用透明色
for row in range(grid_rows):
    for col in range(grid_cols):
        if (row * grid_cols + col) >= num_samples:
            plt.fill_between([col, col + 1], row, row + 1, color='none')

# 设置图例
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[l], label=f'Cluster {l}') for l in unique_labels]
plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.2, 1))

# 设置轴标签和标题
plt.title('Clustering Results')
plt.xlabel('Sample Index')
plt.ylabel('Sample Index')
plt.xticks(np.arange(0, grid_cols + 1, 1))
plt.yticks(np.arange(0, grid_rows + 1, 1))

# 显示图形
plt.tight_layout()
plt.show()

# %%
# 将AE.output对应y_pred做cluster_means
# 将_与y_pred拼接
y_pred = y_pred.reshape(140,1)
raw_data = data_reshaped.reshape(140,360)
stack_data = np.hstack((raw_data, y_pred))
print(stack_data.shape)

# 计算每个簇的平均值
cluster_means = []

for label in unique_labels:
    cluster_data = stack_data[stack_data[:, -1] == label]
    cluster_mean = cluster_data[:, :-1].mean(axis=0)
    cluster_means.append(cluster_mean)

cluster_means = np.array(cluster_means)

# 可视化二维图
plt.figure(figsize=(10, 6))

for i, label in enumerate(unique_labels):
    plt.plot(cluster_means[i], linestyle="-.", label=f'Cluster {label}')
# plt.plot(data[:,0], data[:,1], linestyle=":", label="reference_MnCO3")
# plt.plot(reference_MnO2[:,0], reference_MnO2[:,1], linestyle=":", label="reference_MnO2")
# plt.plot(reference_Mn2O3[:,0], reference_Mn2O3[:,1], linestyle=":", label="reference_Mn2O3")
plt.xlabel('Feature Index')
plt.ylabel('Mean Value')
plt.title('Cluster Means')
plt.legend()
plt.show()

# %%
# 存储为 .npy 文件
np.save('cluster_means.npy', cluster_means)

# %%
