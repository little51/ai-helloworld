import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()

batch_size = 2
learning_rate = 0.01
iterations = 50

# gpu使用配置，在windows上使用gpu版的tensorflow时，要配置
def gpu_config():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 使用30%的GPU
    session = tf.Session(config=config)
    return session

# 准备数据
def prepare_data():
    data_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # 输入
    data_y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # 输出
    train_x, train_y = data_x[:6], data_y[:6]
    test_x, test_y = data_x[6:], data_y[6:]
    # 将数据分成两份，前6个做为训练数据，后4个做为测试数据
    return train_x, train_y, test_x, test_y

# 定义模型
def define_model():
    x_data = tf.placeholder(dtype=tf.float32)
    y_target = tf.placeholder(dtype=tf.float32)
    w = tf.Variable(tf.random_normal(shape=[1, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    model_output = tf.add(tf.matmul(x_data, w), b)
    # 最小二乘法的损失函数
    loss = tf.reduce_mean(tf.square(y_target - model_output))
    # loss = tf.reduce_mean(tf.abs(y_target - model_output))
    # 梯度下降，learning_rate为下降步长，可调
    my_opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = my_opt.minimize(loss)
    return x_data, y_target, loss, train_step, w, b

# 训练模型
def train_model(session, loss, train_step, train_x, train_y, test_x, test_y, w, b):
    init = tf.global_variables_initializer()
    session.run(init)
    loss_vec = []
    for i in range(iterations):
        rand_index = np.random.choice(len(train_x), size=batch_size)
        rand_x = np.transpose([train_x[rand_index]])
        rand_y = np.transpose([train_y[rand_index]])
        session.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss = session.run(
            loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)
        if (i+1) % 2 == 0:
            print('Step #' + str(i+1) + ' w = ' + str(session.run(w)) +
                  ' b = ' + str(session.run(b)) + ' loss = ' + str(temp_loss))
    target_test = session.run(w * test_x + b)
    return loss_vec, target_test


session = gpu_config()
# 准备训练数据和则测试数据
train_x, train_y, test_x, test_y = prepare_data()
x_data, y_target, loss, train_step, w, b = define_model()
loss_vec, target_test = train_model(session, loss, train_step, train_x,
                                    train_y, test_x, test_y, w, b)
# 绘损失值的图
plt.plot(loss_vec, 'k-', label='loss')
plt.xlabel('Generation')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.show()
# 测试模型
print(test_x)
print(target_test)
