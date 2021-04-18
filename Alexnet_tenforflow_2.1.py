import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np



file_name = 'training_data_2_v2_2.npy'

train_data = np.load(file_name, allow_pickle=True)

train = train_data[0:300]
test = train_data[300:400]

X = np.array([i[0] for i in train]).reshape(-1, 86, 86, 1)
Y = [i[1] for i in train]

for j in range(300):
    if Y[j]==[1,0,0,0,0]:
        Y[j] = 1
    elif Y[j]==[0,1,0,0,0]:
        Y[j] = 2
    elif Y[j] == [0,0,1,0,0]:
        Y[j] = 3
    elif Y[j] == [0,0,0,1,0]:
        Y[j] = 4
    else:
        Y[j] = 5

test_x = np.array([i[0] for i in test]).reshape(-1, 86, 86, 1)

test_y = [i[1] for i in test]
for i in range(100):
    if test_y[i]==[1,0,0,0,0]:
        test_y[i] =1
    elif test_y[i]==[0,1,0,0,0]:
        test_y[i] = 2
    elif test_y[i] == [0,0,1,0,0]:
        test_y[i] = 3
    elif test_y[i] == [0,0,0,1,0]:
        test_y[i] = 4
    else:
        test_y[i] = 5
    # x_train = X[0:100]
    # x_test = Y[0:100]
    #
    # y_train = test_x[0:100]
    # y_test = test_y[0:100]


np.set_printoptions(threshold=np.inf)

x_train = X[0:300]
y_train = Y[0:300]

x_test = test_x[0:100]
y_test = test_y[0:100]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
y_train = tf.cast(y_train, tf.float32)
y_test = tf.cast(y_test, tf.float32)
#
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)




# plt.imshow(x_train[30], cmap='gray')  # 绘制灰度图
# plt.show()
# print(x_train[50])

# 输入是100张86*96的图片，输出是一个一维列表[0,0,0,0,1]。
# 就是数据集有问题，下面应该都没有问题


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=1, epochs=50, validation_data=(x_test, y_test), validation_freq=1)
model.summary()
