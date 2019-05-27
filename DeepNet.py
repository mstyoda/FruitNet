import tensorflow as tf
import numpy as np

class DeepNet(object):
	def __init__(self, config):
		super(DeepNet, self).__init__()
		self.parameters = []
		self.config = config
		num_class = 10
		self.X = tf.placeholder(tf.float32, [-1,32,32,3])
		self.Y = tf.placeholder(tf.float32, [-1, num_class])
		self.build()

	def build(self):
		config = self.config
		last = self.X
		img_size = 32
		channels = 3
		for _ in config:
			if _ == 'M':
				last = self.max_pool(last)
				img_size /= 2
			else:
				this_channels = _
				#_ represents the number of the convolution channels in this layer
				last = self.conv(last, channels, this_channels)
				channels = this_channels

		out_dim = img_size * img_size * channels
		last = tf.reshape(last, [-1, out_dim])

		#Add Fully Connected Layers
		fc_size = 256
		last = self.fc(last, out_dim, fc_size)
		last = self.fc(last, fc_size, fc_size)

	def max_pool(self, x):
		return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	
	def conv(self, x):
		np = len(self.parameters)
		conv_w = tf.get_variable(str(np), shape=[1,3,3,1])