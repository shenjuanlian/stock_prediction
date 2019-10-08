from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
from matplotlib import pyplot as plt
tfd = tfp.distributions

flags.DEFINE_float("learning_rate",
                   default=0.001,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=100000,
                     help="Number of training steps to run.")

flags.DEFINE_integer("decay_step",default=100000,help="the total step of decay")
flags.DEFINE_float("min_learning_rate",default=1e-6,help="the minimum learning rate")
flags.DEFINE_float("max_gradient_norm",default=3.0,help="the maximum gradient")
FLAGS = flags.FLAGS

def load_data(inSample,outSample,pointer=0):
    data = pd.read_csv("./data_new/norm_final_data_dis.csv")
    train = data.loc[pointer:pointer+inSample]
    test = data.loc[pointer+inSample:pointer+inSample+outSample]
    labels_train = train['hsi_label']
    labels_test = test['hsi_label']
    train_data = train.drop(columns=['hsi_label'])
    test_data = test.drop(columns=['hsi_label'])
    return train_data,test_data,labels_train,labels_test


def build_input_pipeline(x, y, batch_size):
  training_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(x.values, tf.float32),tf.cast(y.values, tf.int32)))
  training_batches = training_dataset.repeat().batch(batch_size)
  training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_batches)
  batch_features, batch_labels = training_iterator.get_next()
  return batch_features, batch_labels


def main(argv):


  global_step = tf.Variable(0, trainable=False, name='global_step')


  train_data,test_data,labels_train,labels_test = load_data(2687,500)
  if argv[0] == "train":
    features, labels = build_input_pipeline(train_data, labels_train, 128)
  else:
    features, labels = tf.cast(test_data.values, tf.float32),tf.cast(labels_test.values, tf.int32)

  with tf.compat.v1.name_scope("logistic_regression", values=[features]):
    layer = tfp.layers.DenseFlipout(
        units=1,
        activation=None,
        kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
        bias_posterior_fn=tfp.layers.default_mean_field_normal_fn())
    logits = layer(features)
    labels_distribution = tfd.Bernoulli(logits=logits)
  # Compute the -ELBO as the loss, averaged over the batch size.
  if argv[0] == "train":
      neg_log_likelihood = -tf.reduce_mean(
          input_tensor=labels_distribution.log_prob(labels))
      kl = sum(layer.losses) / len(train_data)
      elbo_loss = neg_log_likelihood + kl
  else:
      tf_label = tf.placeholder(dtype=tf.int32)
      tf_prediction = tf.placeholder(dtype=tf.int32)
      probability = tf.nn.sigmoid(logits)
      predictions = tf.cast(tf.nn.sigmoid(logits) > 0.5, dtype=tf.int32)
      tf_precision,tf_precision_update = tf.metrics.precision(tf_label,tf_prediction)
      tf_accuracy,tf_accuracy_update = tf.metrics.accuracy(tf_label,tf_prediction)
  if argv[0] == "train":
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
    # Fit the model to data.
    sess.run(init_op)
    if argv[0] == "train":
        # _logits = sess.run(logits)
        # print(_logits)
        # print(_logits.shape)
        for step in range(FLAGS.max_steps):
          _= sess.run(train_op)
          if step % 100 == 0:
            loss_value,learning_rate = sess.run([elbo_loss,FLAGS.learning_rate])
            print("Step: {:>3d} Loss: {:.3f} Learning_rate: {:.6f}".format(
                step, loss_value,learning_rate))
        saver = tf.train.Saver()
        saver.save(sess, "./model/MyModel.ckpt")

    else:
        np.set_printoptions(threshold=np.inf)
        saver = tf.train.Saver()
        saver.restore(sess, './model/MyModel.ckpt')
        pre, _probability,labels_test = sess.run([predictions,probability,labels])
        labels_test = np.array(labels_test).reshape(len(labels_test),1)
        flag = []
        print(_probability[0:5])
        print(pre[0:5])

        for i in range(labels_test.shape[0]):
            if labels_test[i] == 1 and pre[i] == 1:
                flag.append('red')
            elif labels_test[i] == 0 and pre[i] == 0:
                flag.append('blue')
            else:
                flag.append('yellow')
            sess.run(tf_precision_update, {tf_label: labels_test[i], tf_prediction: pre[i]})
        plt.xlabel("sample index")
        plt.ylabel("probability of Price increased")
        plt.scatter(range(501),_probability,c=flag)
        plt.show()
        print("tf_precision:")
        print(sess.run(tf_precision))
        for i in range(labels_test.shape[0]):
            sess.run(tf_accuracy_update, {tf_label: labels_test[i], tf_prediction: pre[i]})
        print("tf_accuracy:")
        print(sess.run(tf_accuracy))



if __name__ == "__main__":
  tf.compat.v1.app.run(argv=["test"])

# [[-0.04524215]
#  [ 0.5557304 ]
#  [-0.46326762]
#  [-0.14334992]
#  [ 0.3364817 ]]