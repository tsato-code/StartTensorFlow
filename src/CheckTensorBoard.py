# 必要なライブラリのインポート
import tensorflow as tf
import numpy as np

# 変数の定義
dim = 5
x = tf.placeholder(tf.float32, [None, dim + 1])
w = tf.Variable(tf.zeros([dim+1,1]))
y = tf.matmul(x,w)
t = tf.placeholder(tf.float32, [None, 1])
sess = tf.Session()

# 損失関数と学習メソッドの定義
loss = tf.reduce_sum(tf.square(y - t))
train_step = tf.train.AdamOptimizer().minimize(loss)

# TensorBoardで追跡する変数を定義
with tf.name_scope('summary'):
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)
    
# セッションの初期化と入力データの準備
sess.run(tf.global_variables_initializer())

train_t = np.array([5.2, 5.7, 8.6, 14.9, 18.2, 20.4,25.5, 26.4, 22.8, 17.5, 11.1, 6.6])
train_t = train_t.reshape([12,1])
train_x = np.zeros([12, dim+1])
for row, month in enumerate(range(1, 13)):
    for col, n in enumerate(range(0, dim+1)):
        train_x[row][col] = month**n

# 学習
i = 0
for _ in range(100000):
    i += 1
    sess.run(train_step, feed_dict={x: train_x, t: train_t})
    if i % 10000 == 0:
        loss_val = sess.run(loss, feed_dict={x: train_x, t: train_t})
        print('Step: %d, Loss: %f' % (i, loss_val))

get_ipython().run_cell_magic('sh', '', 'tensorboard --logdir=./logs')