# MLP
#
# This example: - Trains a neural network. The Adam optimizer is employed. Its parameters may need tuning. - Monitors
#  the learning using a validation data set. - Stores the network with the lowest validation error in a file. At the
# end, this network is loaded and evaluated. - Uses mini-batches. These batches are generated using the self-defined
# "Batcher" object. A version using proper queues is provided by "MLPMiniBatchWeedTensorFlowTensorBoardQueue.py"


import numpy as np
import tensorflow as tf


# Splits data into mini-batches
# Batches are not randomized/shuffled, shuffling the data in mini-batch learning typically improves the performance
class Batcher:
    """Splits data into mini-batches"""

    def __init__(self, data, batch_size):
        self.data = data
        self.batchSize = batch_size
        self.batchStartIndex = 0
        self.batchStopIndex = 0
        self.noData = self.data.data.shape[0]

    def next_batch(self):
        self.batchStartIndex = self.batchStopIndex % self.noData
        self.batchStopIndex = min(self.batchStartIndex + self.batchSize, self.noData)
        return self.data.data[self.batchStartIndex:self.batchStopIndex], self.data.target[
                                                                         self.batchStartIndex:self.batchStopIndex]


# Flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summary_dir', '../tmp/MLPMiniLog', 'directory to put the summary data')
flags.DEFINE_string('data_dir', '../data', 'directory with data')
flags.DEFINE_integer('maxIter', 20000, 'number of iterations')
flags.DEFINE_integer('batchSize', 128, 'batch size')
flags.DEFINE_integer('noHidden1', 64, 'size of first hidden layer')
flags.DEFINE_integer('noHidden2', 32, 'size of second hidden layer')
flags.DEFINE_float('lr', 0.0001, 'initial learning rate')

# Read data
dataTrain = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=FLAGS.data_dir + '/LSDA2017GalaxiesTrain.csv',
    target_dtype=np.float32,
    features_dtype=np.float32,
    target_column=-1)
dataTest = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=FLAGS.data_dir + '/LSDA2017GalaxiesTest.csv',
    target_dtype=np.float32,
    features_dtype=np.float32,
    target_column=-1)
dataValidate = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=FLAGS.data_dir + '/LSDA2017GalaxiesValidate.csv',
    target_dtype=np.float32,
    features_dtype=np.float32,
    target_column=-1)

print("The observed variance of redshifts in the training data was: %.6f", np.var(dataTrain.data))

# Number of training data points
noTrain = dataTrain.data.shape[0]
print("Number of training data points:", noTrain)

# Input dimension
inDim = dataTrain.data.shape[1]

# Create graph
sess = tf.Session()

# Initialize placeholders
x_data = tf.placeholder(shape=[None, inDim], dtype=tf.float32, name='input')
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='target')


# Define variables
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # restrict to +/- 2*stddev
    return tf.Variable(initial, name='weights')


def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='bias')


# Define model
with tf.name_scope('layer1') as scope:
    W_1 = weight_variable([inDim, FLAGS.noHidden1])
    b_1 = bias_variable([FLAGS.noHidden1])
    y_1 = tf.nn.sigmoid(tf.matmul(x_data, W_1) + b_1)

with tf.name_scope('layer2') as scope:
    W_2 = weight_variable([FLAGS.noHidden1, FLAGS.noHidden2])
    b_2 = bias_variable([FLAGS.noHidden2])
    y_2 = tf.nn.sigmoid(tf.matmul(y_1, W_2) + b_2)

with tf.name_scope('layer3') as scope:
    W_3 = weight_variable([FLAGS.noHidden2, 1])
    b_3 = bias_variable([1])
    model_output = tf.matmul(y_2, W_3) + b_3

# Declare loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target),
                      name='mean_cross-entropy')
tf.summary.scalar('cross-entropy', loss)

# Declare optimizer
my_opt = tf.train.AdamOptimizer(FLAGS.lr)
train_step = my_opt.minimize(loss)

# Map model output to mean squared error
with tf.name_scope('predictions') as scope:
    prediction = tf.sigmoid(model_output)
    prediction_sq_diff = tf.squared_difference(prediction, y_target, name='squared_difference')
mse = tf.reduce_mean(prediction_sq_diff, name='mean_squared_error')
tf.summary.scalar('mse', mse)

# Logging
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train')
test_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/test')
validate_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/validate')
writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
saver = tf.train.Saver()  # for storing the best network

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Best validation mse seen so far
bestValidation = 0.0

# Mini-batches for training
batcher = Batcher(dataTrain, FLAGS.batchSize)

# Training loop
for i in range(FLAGS.maxIter):
    xTrain, yTrain = batcher.next_batch()
    sess.run(train_step, feed_dict={x_data: xTrain, y_target: np.transpose([yTrain])})
    summary = sess.run(merged, feed_dict={x_data: xTrain, y_target: np.transpose([yTrain])})
    train_writer.add_summary(summary, i)
    if (i + 1) % 100 == 0:
        print("Iteration:", i + 1, "/", FLAGS.maxIter)
        summary = sess.run(merged, feed_dict={x_data: dataTest.data, y_target: np.transpose([dataTest.target])})
        test_writer.add_summary(summary, i)
        currentValidation, summary = sess.run([mse, merged], feed_dict={x_data: dataValidate.data,
                                                                        y_target: np.transpose(
                                                                            [dataValidate.target])})
        validate_writer.add_summary(summary, i)
        if currentValidation < bestValidation or bestValidation == 0.0:
            bestValidation = currentValidation
            saver.save(sess=sess, save_path=FLAGS.summary_dir + '/bestNetwork')
            print("\tbetter network stored, ", currentValidation, "<", bestValidation)

# Print values after last training step
print("final training mse:",
      sess.run(mse, feed_dict={x_data: dataTrain.data, y_target: np.transpose([dataTrain.target])}),
      "final test mse: ",
      sess.run(mse, feed_dict={x_data: dataTest.data, y_target: np.transpose([dataTest.target])}),
      "final validation mse: ",
      sess.run(mse, feed_dict={x_data: dataValidate.data, y_target: np.transpose([dataValidate.target])}))

# Load the network with th elowest validation error
saver.restore(sess=sess, save_path=FLAGS.summary_dir + '/bestNetwork')
print("best training mse:",
      sess.run(mse, feed_dict={x_data: dataTrain.data, y_target: np.transpose([dataTrain.target])}),
      "best test mse: ",
      sess.run(mse, feed_dict={x_data: dataTest.data, y_target: np.transpose([dataTest.target])}),
      "best validation mse: ",
      sess.run(mse, feed_dict={x_data: dataValidate.data, y_target: np.transpose([dataValidate.target])}))
