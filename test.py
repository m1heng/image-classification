from imageloader import ImageDataSet
from perceptron_y import Perceptron
from util import *
from simpleNN import NeuralNetwork
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

def ops():
    data = ImageDataSet(70,60)
    data.loadImageData("facedata/facedatatrain", -1)
    print(data.number)
    data.loadLabelData("facedata/facedatatrainlabels", data.number)
    testdata = ImageDataSet(70,60)
    testdata.loadImageData("facedata/facedatatest", -1)
    testdata.loadLabelData("facedata/facedatatestlabels", testdata.number)
    pc = Perceptron(70*60,['0', '1'])
    pc.train(data.images, data.labels, 1)

    x = pc.classify(testdata.images)

    a = Accuracy(x, testdata.labels)

    print(a*100)


def nntest():
    data = ImageDataSet(28,28)
    data.loadImageData("digitdata/trainingimages", 3)
    data.loadLabelData("digitdata/traininglabels", data.number)
    print(data.number)
    nn = NeuralNetwork(28*28, 15, ['0', '1','2','3','4','5','6','7','8','9'])
    nn.train(data.images, data.labels, 1)

    testdata = ImageDataSet(28,28)
    testdata.loadImageData("digitdata/testimages", 1)
    testdata.loadLabelData("digitdata/testlabels", testdata.number)

    x = nn.classify(testdata.images)
    #print(testdata.labels)
    #print(x)

    a = Accuracy(x, testdata.labels)

    print()
    print("Accuracy is : %d" % (a*100))


if __name__ == '__main__':
    nntest()
