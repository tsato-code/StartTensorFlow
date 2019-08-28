#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

# データフローグラフの定義
a = tf.constant(2, name="a")
b = tf.constant(-3, name="b")
c = a + b

# まとめて計算を実行
with tf.Session() as sess:
    print(sess.run(c))


# In[2]:


# c は Tensor 型のインスタンス
print(c)


# In[3]:


# 計算グラフの確認
graph = tf.get_default_graph()
print(graph.as_graph_def())


# In[4]:


# まとめて実行
d = a - b
with tf.Session() as sess:
    print(sess.run([c, d]))


# In[5]:


# 変数と定数の違い
a = tf.Variable(1, name="a")
b = tf.constant(4, name="b")
c = tf.assign(a, a+b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("1回目: [c, a] =", sess.run([c, a]))
    print("2回目: [c, a] =", sess.run([c, a]))


# In[6]:


# プレースホルダー
# 実行時に値を指定する
a = tf.placeholder(dtype=tf.int32, name="a")
b = tf.constant(4, name="b")
c = a + b

with tf.Session() as sess:
    # 変数の初期化
    sess.run(tf.global_variables_initializer())
    print("a + b =", sess.run(c, feed_dict={a: 1}))


# In[7]:


# 演算
a = tf.constant(2, name="a")
b = tf.constant(-4, name="b")
c = tf.add(a, b)
d = tf.multiply(a, b)

with tf.Session() as sess:
    print("a + b =", sess.run(c))
    print("a * b =", sess.run(d))


# In[8]:


# ベクトル演算
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
c = a + b

with tf.Session() as sess:
    print("a + b =", sess.run(c))


# In[9]:


# 行列演算
a = tf.constant([[1, 2, 3]])
b = tf.constant([[1, 2],[-1, -2],[-1, 1]])
c = tf.matmul(a, b)

print(a.shape)
print(b.shape)
print(c.shape)

with tf.Session() as sess:
    print("a =", sess.run(a))
    print("b =", sess.run(b))
    print("c =", sess.run(c))


# In[10]:


a = tf.Variable(1)
b = tf.assign(a, a + 1)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))
    saver.save(sess, "../models/20190828/model.ckpt")


# In[11]:


# 最適化
x = tf.Variable(0., name="x")
func = (x-1)**2
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_step = optimizer.minimize(func)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        sess.run(train_step)
    print("x =", sess.run(x))

