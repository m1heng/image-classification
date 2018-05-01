from imageloader import ImageDataSet
from perceptron_y import Perceptron
from util import *
import numpy as np 


def test_imageloader():
	data = ImageDataSet(28,28)
	data.loadImageData("digitdata/trainingimages", 1)

	matrix = np.array(data.images[0])
	print(matrix)
	print(data.number)

def test_percep():
    a = np.zeros([20+1, ])
    print(a)

def Magic(x,w):
    x = np.array(x).flatten()
    print(x.shape)
    print(w.shape)
    return np.dot(x,np.transpose(w))

if __name__ == '__main__':
    data = ImageDataSet(28,28)
    data.loadImageData("digitdata/trainingimages", -1)
    data.loadLabelData("digitdata/traininglabels", data.number)
    testdata = ImageDataSet(28,28)
    testdata.loadImageData("digitdata/testimages", -1)
    testdata.loadLabelData("digitdata/testlabels", testdata.number)
    pc = Perceptron(28*28,['0','1','2','3','4','5','6','7','8','9'])
    pc.train(data.images, data.labels, 2)

    x = pc.classify(testdata.images)

    a = Accuracy(x, testdata.labels)

    print(a*100)
