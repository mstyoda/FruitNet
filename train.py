# import tensorflow as tf
import numpy as np
import random
# from DeepNet import DeepNet
class Args(object):
	"""docstring for Args"""
	def __init__(self):
		super(Args, self).__init__()
		# batch size
		self.b = 100
		# learning rate
		self.lr = 1e-2
		# momentum
		self.momentum = 0.9
		# network
		self.config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M']
		# the number of epochs
		self.max_epoch = 100
		# 
		self.log_path = "RUN/log.out"

def getData(path='DATASET'):
	tr_x = np.load('%s/train.datas' % (path)).astype(np.float32)
	_tr_y = np.load('%s/train.labels' % (path)).astype(np.int64)
	n_train = _tr_y.shape[0]
	tr_y = np.zeros([n_train, 10])
	tr_y[np.arange(n_train), _tr_y] = 1.

	te_x = np.load('%s/test.datas' % (path)).astype(np.float32)
	_te_y = np.load('%s/test.labels' % (path)).astype(np.int64)
	n_test = _te_y.shape[0]
	te_y = np.zeros([n_test, 10])
	te_y[np.arange(n_test), _te_y] = 1.

	# print tr_x.shape, tr_y.shape, te_x.shape, te_y.shape
	# print type(tr_x), type(tr_y), type(te_x), type(te_y)
	# print tr_x.dtype, tr_y.dtype, te_x.dtype, te_y.dtype
	means = [tr_x[:,:,:,d].mean() for d in range(0,3)]
	stds = [tr_x[:,:,:,d].std() for d in range(0,3)]

	for d in range(0,3):
		tr_x[:,:,:,d] = (tr_x[:,:,:,d] - means[d]) / stds[d]
		te_x[:,:,:,d] = (te_x[:,:,:,d] - means[d]) / stds[d]
	# print means, stds
	return (tr_x, tr_y, te_x, te_y)

def getAcc(model, xs, ys, sess):
	preds = sess.run(model.pred, feed_dict={
		model.X : xs,
		model.Y : ys
		})
	n = ys.shape[0]
	preds = np.argmax(preds, axis=1)
	ylabels = np.argmax(ys, axis=1)
	return float(np.sum(preds == ylabels)) / float(n)

def train(datas, model, args):
	tr_x, tr_y, te_x, te_y = datas
	step = 0
	n = tr_x.shape[0]
	range_n = np.arange(n)
	epoch_size = n // args.b;
	max_step = args.max_epoch * epoch_size
	lr = args.lr

	train_step = tf.train.MomentumOptimizer(model.lr, args.momentum).minimize(model.loss)

	sess = tf.Session()
	
	open(args.log_path,'w').write('')
	for step in range(max_step):
		idx = random.sample(range_n, b)
		bx = tr_x[idx]
		by = tr_y[idx]
		sess.run(train_step, feed_dict={
			model.lr : lr,
			model.X : bx,
			model.Y : by
			})
		
		if step % epoch_size == 0:
			tr_acc = getAcc(model, bx, by, sess)
			idx = random.sample(np.arange(te_x.shape[0]), b)
			bx = te_x[idx]
			by = te_y[idx]
			te_acc = getAcc(model, bx, by, sess)
			step_log_str = 'step: %d  tr_acc: %d%, te_acc: %d%' % (step, tr_acc, te_acc)
			open(args.log_path, 'a+').write(step_log_str + '\n')
			print step_log_str
	sess.close()

if __name__ == "__main__":
	datas = getData()
	args = Args()
	model = DeepNet(args.config)
	train(datas, model, args)