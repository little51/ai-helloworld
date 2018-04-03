# 写给程序员的机器学习入门

## 1、数学门槛

说到机器学习的数学基础，线性代数、微积分、概率论是起步，几乎所有讲AI的书籍，上来就是一通数学公式，这说明这个领域的门槛还是相当高的。从事管理信息系统开发的程序员，整天与js、SQL打交道，可以写出逆天的js，也可以写出多层嵌套的SQL，但一遇到数学，别说高等数学，大部分人把小学以上的数学早就还给老师了，对于机器学习，只好从入门到放弃。

机器学习的开发需要一定的数学知识，但在AI应用领域，其实对数学的要求并不高，对于初学者，并不一定开始就琢磨数学计算，应从机器学习的基本原理入手，由浅入深，随着研究的深入，再回忆一下数学也可以。

阅读机器学习入门书籍，遇到一些 公式也别太着急，没那么复杂，稍微变换一下思路，如下面这个公式：

![https://github.com/little51/ai-helloworld/blob/master/loss.png]()

描述损失函数，实际上就累加，写成下面这段代码大家就看明白了。在编写tensorflow代码时，x、y的数据（可以理解成数组，专业术语叫张量）可以一句带入，不用写循环。如果x是多维，样本量大，则这样的计算就非常多，而且全是浮点运算，所以需要GPU。

```java
loss = 0.0 ;
for(int i=0;i++;i<n){
    loss += abs(w * x[i] - b + y[i]);
}
```

##  2、基本原理

### （1）思路

tensorflow等入门书，将Hello World!用图像识别来描述，难度还是相当高的，首先，得先明白tensor的概念，点阵灰度图像的张量表示也得花点功夫理解，还要学习占位符，线性回归等，门槛比较高。比较好理解的学习思路是：先搞明白机器学习的最简单用法，不要一下引入太多概念，而且例子相对完整，然后在此基础上，再深入学习。

### （2）基本原理

机器学习的基本原理：知道输入和结果，反推出输入与结果的函数关系，再将函数套到新的输入上，验算结果偏差。比如监督学习里垃圾短信判断，找出训练文本和标签（0或1，是否是垃圾短信）的函数关系，再用测试文本代入函数，看结果是0还是1。

### （3）线性回归

Hello world最好是以找函数关系举例，如输入x有10种情况，x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]，输出y的对应值为 y = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]，问x和y的关系是什么。因为样例比较简单，大家很快就看出来关系是：y = x + 1。以机器学习的术语来说，这是一个线性关系，可以用线性回归解决。线性关系可以描述为：y = w * x + b，那这个关系就是 y = 1 * x + 1,其中w = 1,b = 1。如果通过机器学习，能算出w = 1,b = 1，那损失是0，是理想的状态，在现实情况中，x和y不可能完全满足线性关系，总会有噪音。如果不是线性关系，可以用更高次的方程：y = u * x^2^  + w * x + b。

那机器学习怎找关系的，不会是穷举吧，当然不是，这就要用到如梯度下降算法，不过初学时，不用关心这个，就知道机器能算出就行了。

### （4）深入学习

对于满足线性关系的领域，无非是x,y的每个元素变成了更多的维度，写程序的方式没有变化，不管几维，都用张量（tensor）表示，至于工具怎计算张量，不用关心。对于不满足线性关系的领域，再学习其他算法。

## 3、工具选型及安装

tensorflow最为流行，但比较底层，pytorch支持动态图，也比较好理解，但在windows上不好安装。入门可以选keras，keras可以当成是tensorflow的API，在keras搞明白机器学习的基本原理后，有必要的话再转到tensorflow，这样会轻松很多。关于keras的安装，网上文章很多，这个对于程序员没难度，不详细讲了。

## 4、机器学习的Hello world!

线性回归的hello world源程序如下，使用“python 文件名”运行。

```python
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
```

## 5、Hello world的tensorflow版本

掌握了keras版本的Hello world后，可以用tensorflow实现相同的功能。tensorflow 比keras更底层一些，需要明确计算图、损失函数等，y = w*x + b要显式申明。

在此基础上，逐步掌握以下概念

* 张量
* 占位符
* 线性关系
* 损失函数
* 梯度下降

```python
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
```

