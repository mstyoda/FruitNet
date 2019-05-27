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
        self.lr = tf.placeholder(tf.float32, [])
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
        self.pred = self.fc(fc_size, num_class)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=Y,
            logits=self.pred))
    
    def max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    def conv(self, x):
        nameid = len(self.parameters)
        # str(np) is the id of this convolution kernel
        conv_w = tf.get_variable(name=str(nameid), shape=[1,3,3,1], initializer=tf.truncated_normal_initializer(mean=0., stddev=5e-2));
        self.parameters.append(conv_w)
        return tf.nn.relu(tf.nn.conv2d(x, conv_w, strides=[1,1,1,1], padding='SAME'))
    
    def fc(self, x, in_dim, out_dim):
        nameid = len(self.parameters)
        w = tf.get_variable(name=str(nameid), shape=[in_dim, out_dim], initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None))
        return tf.nn.relu(tf.matmul(x,w))

if __name__ == '__main__':
    deepNet = DeepNet()
