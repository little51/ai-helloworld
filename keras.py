from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import matplotlib.pyplot as plt
# gpu使用配置，在windows上使用gpu版的tensorflow时，要配置
def gpu_config():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3 #使用30%的GPU
    session = tf.Session(config=config)
    KTF.set_session(session)
# 准备数据
def prepare_data():
    data_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  #输入
    data_y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) #输出
    train_x, train_y = data_x[:6], data_y[:6]
    test_x, test_y = data_x[6:], data_y[6:]
    # 将数据分成两份，前6个做为训练数据，后4个做为测试数据
    return train_x,train_y,test_x,test_y
# 定义模型
def define_model():
    model = Sequential()
    # y = W * x + b
    model.add(Dense(
        input_dim=1,  # 一维
        units=1,
        use_bias=True, # 需要 b
    ))
    # 使用mse损失函数和sgd优化器
    model.compile(loss='mse', optimizer='sgd')
    return model
# 训练模型
def train_model(model,train_x,train_y,test_x,test_y):
    # 匹配训练
    h = model.fit(train_x, train_y, batch_size=2, epochs=50, initial_epoch=0)
    # 得分
    score = model.evaluate(train_x, train_y, batch_size=2)
    # 验算
    pred_y = model.predict(test_x)
    # 取出结果w和b
    W, b = model.layers[0].get_weights()
    model.save_weights('model1')
    return h,score,pred_y,W,b

gpu_config()
# 准备训练数据和则测试数据
train_x,train_y,test_x,test_y = prepare_data()
# 定义模型
model = define_model()
# 训练模型
h,score,pred_y,W,b = train_model(model,train_x,train_y,test_x,test_y)
# 打印结果
print(score)
print('Weights=', W, '\nbiases=', b)
print(pred_y)
# 绘图
plt.scatter(test_x, test_y)
plt.plot(test_x, pred_y)
plt.plot(h.epoch ,h.history['loss'])
plt.show()