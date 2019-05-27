import numpy as np
import cv2
import os
import random

fruits = ['apple', 'banana', 'cherryTomato', 'kiwiFruit', 'mango', 
'orange', 'peach', 'pear', 'pitaya', 'strawberry']

def data_augment(img):
	w, h, c = img.shape
	d = int(min(w, h) * 0.8)
	_img = np.zeros((d, d, c)).astype(img.dtype)
	si = np.random.randint(0, w - d)
	sj = np.random.randint(0, h - d)
	_img += img[si:si+d, sj:sj+d, :]
	
	if np.random.choice([True, False]):
		_img = cv2.flip(_img, 1)
	if np.random.choice([True, False]):
		_img = cv2.flip(_img, 0)
	_img = cv2.resize(_img, (32, 32))
	# cv2.imshow('dfsd', _img)
	# cv2.waitKey()
	return _img

def addDatas(fruit, label):
	imgs = []
	labels = []
	for file in os.listdir(fruit):
		img = cv2.imread(os.path.join(fruit, file), cv2.IMREAD_COLOR)
		imgs.append(img)
		labels.append(label)
	return imgs, labels

imgs = []
labels = []
for i, fruit in enumerate(fruits):
	_imgs, _labels = addDatas(fruit, i)
	imgs += _imgs
	labels += _labels

n = len(imgs)

idx = random.sample(np.arange(n), n)
imgs = [imgs[idx[i]] for i in range(0,n)]
labels = [labels[idx[i]] for i in range(0,n)]

n_test = n / 10
test_imgs, test_labels = imgs[:n_test], labels[:n_test] 
train_imgs, train_labels = imgs[n_test:], labels[n_test:]

datas, labels = [], []
for i in range(0, n - n_test):
	print 'train: ',i
	for _ in range(0, 30):
		datas.append(data_augment(train_imgs[i]))
		labels.append(train_labels[i])

datas = np.array(datas)
labels = np.array(labels)
datas.dump('train.datas')
labels.dump('train.labels')

print 'train: ', datas.shape, labels.shape

datas, labels = [], []
for i in range(0, n_test):
	print 'test: ', i
	for _ in range(0, 30):
		datas.append(data_augment(test_imgs[i]))
		labels.append(test_labels[i])

datas = np.array(datas)
labels = np.array(labels)
datas.dump('test.datas')
labels.dump('test.labels')

print 'test: ', datas.shape, labels.shape