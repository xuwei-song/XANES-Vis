# %%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Reshape, Conv1DTranspose
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import *
import pandas as pd


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
ym = encoder.predict(data_reshaped)
np.savetxt('embeddings', ym, delimiter=',')


# %%
