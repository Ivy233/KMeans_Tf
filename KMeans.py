from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA

# 导入数据
mnist = input_data.read_data_sets("Kmeans/data/", one_hot=True)
train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.labels

# 集合设置
accuracy_set = []
# 参数设置
times = 100  # 训练100次
num_labels = 10  # 最终结果归类
n = 24  # PCA降维
k = 800  # KMeans训练的聚类中心数量，之后映射到数字即可

# PCA降维
pca = PCA(n_components=n)
reduced_train_images = pca.fit_transform(train_images)
print(len(reduced_train_images), len(reduced_train_images[0]),
      pca.n_components_)

# tf输入输出
tf_in = tf.placeholder(tf.float32, shape=[None, pca.n_components_])  # 输入
tf_out = tf.placeholder(tf.float32, shape=[None, num_labels])  # 输出

# KMeans，注意这是归属在tf下的，产生就会自动加入tf网络
# 距离度量的方式采用余弦距离（余弦相似度）
tf_kmeans = KMeans(
    inputs=tf_in,
    num_clusters=k,
    distance_metric='cosine'  # 对于没有归一化的使用cos距离比较好，关掉也可以测试
)
(tf_incluster_dis, tf_cluster_id, tf_dis, tf_kmeans_initialized, tf_init_op,
 tf_train_op) = tf_kmeans.training_graph()  # 1.13
tf_cluster_id = tf_cluster_id[0]  # 你可以考虑删除，不过会有很神奇的结果
tf_avg_dis = tf.reduce_mean(tf_dis)
tf_init_vars = tf.global_variables_initializer()

# 准备开始计算
sess = tf.Session()
# 初始化网络
sess.run(tf_init_vars, feed_dict={tf_in: reduced_train_images})
sess.run(tf_init_op,
         feed_dict={tf_in: reduced_train_images})  # init_op在training_graph的输出中

# 训练
for i in range(times):
    _, dis, get_idx = sess.run([tf_train_op, tf_avg_dis, tf_cluster_id],
                               feed_dict={tf_in: reduced_train_images})
    # 统计与映射
    counts = np.zeros(shape=(k, num_labels))
    for j in range(len(get_idx)):
        counts[get_idx[j]] += mnist.train.labels[j]
    f_cluster_label = [np.argmax(c) for c in counts]  # 从聚类中心到最终标签的映射
    f_cluster_label = tf.convert_to_tensor(f_cluster_label)

    # 构建验证网络
    tf_predicted_labels = tf.nn.embedding_lookup(f_cluster_label,
                                                 tf_cluster_id)  # 构建查询
    tf_correct = tf.equal(tf_predicted_labels,
                          tf.cast(tf.argmax(tf_out, 1), tf.int32))  # 计算是否正确
    tf_accuracy = tf.reduce_mean(tf.cast(tf_correct, tf.float32))

    # 测试
    reduced_test_images = pca.transform(test_images)
    accuracy, predicted_answers = sess.run([tf_accuracy, tf_predicted_labels],
                                           feed_dict={
                                               tf_in: reduced_test_images,
                                               tf_out: test_labels
                                           })
    accuracy_set.append(accuracy)
    if i % 10 == 0:
        print("Step %i, Avg Distance: %f, Accuracy: %f" % (i, dis, accuracy))
print(accuracy_set)
