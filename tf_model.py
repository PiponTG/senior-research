### TODO ###
# find new dataset
### write input function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tqdm import tqdm

# creates a tensorflow model
class model_generator():
	def __init__(self, n_layers=3, n_nodes_hl=100, 
				 batch_size=100, n_classes=10):
		#defining size of model
		self.n_layers = n_layers
		self.n_nodes_hl = n_nodes_hl
		self.n_classes = n_classes

		#defining data processing
		self.batch_size = batch_size
		self.mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

		# height x width
		self.x = tf.placeholder('float', [None, 784])
		self.y = tf.placeholder('float')

		print('L: ', self.n_layers, ' N: ', self.n_nodes_hl)
		self.accuracy = self.train_neural_network()

	def neural_network_model(self, data):
		HLA = [] #hidden layer array
		LA = []  #layer array

		#setting weights and biases for hidden layers
		HLA.append({'weights':tf.Variable(tf.random_normal([784, self.n_nodes_hl])),'biases':tf.Variable(tf.random_normal([self.n_nodes_hl]))})
		for i in range(1, (self.n_layers - 1)):
			HLA.append({'weights':tf.Variable(tf.random_normal([self.n_nodes_hl, self.n_nodes_hl])),'biases':tf.Variable(tf.random_normal([self.n_nodes_hl]))}) 
		HLA.append({'weights':tf.Variable(tf.random_normal([self.n_nodes_hl, self.n_classes])),'biases':tf.Variable(tf.random_normal([self.n_classes]))})
		
		#putting data through layers
		LA.append(tf.add(tf.matmul(data, HLA[0]['weights']), HLA[0]['biases']))
		LA[0] = tf.nn.relu(LA[0])
		for i in range(1, (self.n_layers - 1)):
			LA.append(tf.add(tf.matmul((LA[i - 1]), (HLA[i]['weights'])), HLA[i]['biases']))
			LA[i] = tf.nn.relu(LA[i])
		return (tf.add(tf.matmul((LA[self.n_layers - 2]), (HLA[self.n_layers - 1]['weights'])), HLA[self.n_layers - 1]['biases']))
		
	def train_neural_network(self):
		prediction = self.neural_network_model(self.x)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=self.y ))
		optimizer = tf.train.AdamOptimizer().minimize(cost)

		# 1 feedforward + backprop
		hm_epochs = 10

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			for epoch in tqdm(range(hm_epochs)):
				epoch_loss = 0
				for _ in range(int(self.mnist.train.num_examples / self.batch_size)):
					e_x, e_y = self.mnist.train.next_batch(self.batch_size)
					_, c = sess.run([optimizer, cost], feed_dict = {self.x: e_x, self.y : e_y})
					epoch_loss += c
				#print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)

			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y , 1))

			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
			return accuracy.eval({self.x: self.mnist.test.images, self.y : self.mnist.test.labels})
			sess.close()
			
	def get_accuracy(self):
		return self.accuracy


