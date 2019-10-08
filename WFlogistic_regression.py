from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.python import pywrap_tensorflow
# Dependency imports
from absl import flags
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
tfd = tfp.distributions

flags.DEFINE_float("learning_rate",
                   default=0.001,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=10000,
                     help="Number of training steps to run.")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join("./logistic_regression"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("decay_step",default=20000,help="the total step of decay")
flags.DEFINE_float("min_learning_rate",default=1e-6,help="the minimum learning rate")
flags.DEFINE_float("max_gradient_norm",default=3.0,help="the maximum gradient")
FLAGS = flags.FLAGS

def load_data():
    trainset = pd.read_csv("./fine_data/final_final_train.csv")
    testset = pd.read_csv("./fine_data/final_final_test.csv")
    labels_train = trainset['hsi_Close_RDP1']
    labels_test = testset['hsi_Close_RDP1']
    train_data = trainset.drop(columns=['hsi_Close_RDP1'])
    test_data = testset.drop(columns=['hsi_Close_RDP1'])
    return train_data,test_data,labels_train,labels_test

def load_data_WF(total_sample,inSample,outOfSample,mark):
    # total_sample = pd.read_csv("./fine_data/total_data.csv")
    test_labels = total_sample["hsi_Close_RDP1"].loc[mark:mark+outOfSample]
    test_sample = total_sample.drop(columns=['hsi_Close_RDP1']).loc[mark:mark+outOfSample]
    train_labels = total_sample["hsi_Close_RDP1"].loc[mark+outOfSample:mark+outOfSample+inSample]
    train_sample = total_sample.drop(columns=['hsi_Close_RDP1']).loc[mark+outOfSample:mark+outOfSample+inSample]
    return test_labels,test_sample,train_labels,train_sample
def build_input_pipeline(x, y, batch_size):
  training_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(x.values, tf.float32),tf.cast(y.values, tf.int32)))
  training_batches = training_dataset.repeat().batch(batch_size)
  training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_batches)
  batch_features, batch_labels = training_iterator.get_next()
  return batch_features, batch_labels


def main(argv):
    print("main")
    # for j in range(2):
    batchsize = 128
    inSample = 180
    outOfSample = 30

    # for i in range(int((len(total_sample) - (inSample + outOfSample))/inSample)):

    global_step = tf.Variable(0, trainable=False, name='global_step')

    local_flag = tf.placeholder(dtype=tf.string)
    print(local_flag)
    print(tf.convert_to_tensor('train'))
    print(tf.equal(local_flag,tf.convert_to_tensor('train'))==True)
    if local_flag == tf.cast("train",tf.string):
        features = tf.placeholder(dtype=tf.float32, shape=[batchsize,None],name="train_features")
        labels = tf.placeholder(dtype=tf.int32,shape=[batchsize,],name="train_labels")
        # features, labels = build_input_pipeline(train_sample, train_labels, 128)
    else:
        features = tf.placeholder(dtype=tf.float32, shape=[outOfSample, None], name="test_features")
        labels = tf.placeholder(dtype=tf.int32, shape=[outOfSample, None], name="test_labels")
        # features, labels = tf.cast(test_sample.values, tf.float32),tf.cast(test_labels.values, tf.int32)
    # Define a logistic regression model as a Bernoulli distribution
    # parameterized by logits from a single linear layer. We use the Flipout
    # Monte Carlo estimator for the layer: this enables lower variance
    # stochastic gradients than naive reparameterization.
    with tf.compat.v1.name_scope("logistic_regression", values=[features]):
        layer = tfp.layers.DenseFlipout(
            units=1,
            activation=None,
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
            bias_posterior_fn=tfp.layers.default_mean_field_normal_fn())
        logits = layer(features)
        labels_distribution = tfd.Bernoulli(logits=logits)
    # Compute the -ELBO as the loss, averaged over the batch size.
    if local_flag == tf.cast("train",tf.string):
        neg_log_likelihood = -tf.reduce_mean(
          input_tensor=labels_distribution.log_prob(labels))
        kl = sum(layer.losses) / len(inSample)
        elbo_loss = neg_log_likelihood + kl
    else:
          # Build metrics for evaluation. Predictions are formed from a single forward
          # pass of the probabilistic layers. They are cheap but noisy predictions.
          tf_label = tf.placeholder(dtype=tf.int32)
          tf_prediction = tf.placeholder(dtype=tf.int32)
          predictions = tf.cast(tf.nn.sigmoid(logits) > 0.5, dtype=tf.int32)
          tf_precision,tf_precision_update = tf.metrics.precision(tf_label,tf_prediction)
          tf_accuracy,tf_accuracy_update = tf.metrics.accuracy(tf_label,tf_prediction)
    if local_flag == tf.cast("train",tf.string):
      with tf.compat.v1.name_scope("train"):
        learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, global_step,
                                                    FLAGS.decay_step, FLAGS.min_learning_rate, power=0.5)
        FLAGS.learning_rate = learning_rate
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(elbo_loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, FLAGS.max_gradient_norm
        )
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        train_op = optimizer.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=global_step
        )

    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                     tf.compat.v1.local_variables_initializer())

    with tf.compat.v1.Session() as sess:
        print("sess")
        # Fit the model to data.
        sess.run(init_op)
        # if local_flag == "train":
        total_sample = pd.read_csv("./fine_data/total_data.csv")
        mark = 0
        test_labels, test_sample, train_labels, train_sample = load_data_WF(total_sample, inSample, outOfSample, mark)

        for step in range(FLAGS.max_steps):
          features_train, labels_train = build_input_pipeline(train_sample, train_labels, batchsize)
          loss_value,learningRate,_= sess.run([elbo_loss,FLAGS.learning_rate,train_op],{features:features_train,labels:labels_train,local_flag:'train'})
          if step % 100 == 0:
            # loss_value,learning_rate = sess.run([elbo_loss,FLAGS.learning_rate])
            print("Step: {:>3d} Loss: {:.3f} Learning_rate: {:.6f}".format(
                step, loss_value,learningRate))
        saver = tf.train.Saver()
        saver.save(sess, "./model/MyModelWF{}.ckpt".format(0))

        # else:
        # np.set_printoptions(threshold=np.inf)
        # saver = tf.train.Saver()
        # saver.restore(sess, "./model/MyModelWF{}.ckpt".format(0))
        # pre = sess.run(predictions)
        # labels_test = sess.run(labels)
        # labels_test = np.array(labels_test).reshape(len(labels_test),1)
        # for i in range(labels_test.shape[0]):
        #     sess.run(tf_precision_update, {tf_label: labels_test[i], tf_prediction: pre[i]})
        # print("tf_precision:")
        # print(sess.run(tf_precision))
        # for i in range(labels_test.shape[0]):
        #     sess.run(tf_accuracy_update, {tf_label: labels_test[i], tf_prediction: pre[i]})
        # print("tf_accuracy:")
        # print(sess.run(tf_accuracy))
    # local_flag = "test"


if __name__ == "__main__":
    tf.compat.v1.app.run()
