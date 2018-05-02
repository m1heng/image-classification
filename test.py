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
    data.loadImageData("digitdata/trainingimages", 20)
    data.loadLabelData("digitdata/traininglabels", data.number)
    #print(   np.array(data.images[0]) / 2  )
    nn = NeuralNetwork(28*28, 15, ['0', '1','2','3','4','5','6','7','8','9'])
    nn.train(data.images, data.labels, 50)

    
    
    testdata = ImageDataSet(28,28)
    testdata.loadImageData("digitdata/testimages", 3)
    testdata.loadLabelData("digitdata/testlabels", testdata.number)


    x = nn.classify(testdata.images)
    print(testdata.labels)
    print(x)
    
  

    a = Accuracy(x, testdata.labels)

    print()
    print("Accuracy is : %d" % (a*100))

def lay():
    data = ImageDataSet(28,28)
    data.loadImageData("digitdata/trainingimages", -1)
    data.loadLabelData("digitdata/traininglabels", data.number)
    nn = NeuralNetwork( (28*28, 28,14 ,10), ['0', '1','2','3','4','5','6','7','8','9'])
    testdata = ImageDataSet(28,28)
    testdata.loadImageData("digitdata/testimages", -1)
    testdata.loadLabelData("digitdata/testlabels", testdata.number)

    for _ in range(50):
        print("Time:%d"%_)
        nn.train(data.images, data.labels, 2, 0.5)    
        x = nn.classify(testdata.images)
        a = Accuracy(x, testdata.labels)
        print()
        print("Accuracy is : %d" % (a*100))
    #print(testdata.labels)
    #print(x)

def layface():
    data = ImageDataSet(70,60)
    data.loadImageData("facedata/facedatatrain", -1)
    data.loadLabelData("facedata/facedatatrainlabels", data.number)
    nn = NeuralNetwork( ( 70*60 ,10,2 ), ['0', '1'])
    testdata = ImageDataSet(70,60)
    testdata.loadImageData("facedata/facedatatest", -1)
    testdata.loadLabelData("facedata/facedatatestlabels", testdata.number)

    for _ in range(50):
        print("Time:%d"%_)
        nn.train(data.images, data.labels, 2, 0.5)    
        x = nn.classify(testdata.images)
        a = Accuracy(x, testdata.labels)
        print()
        print("Accuracy is : %d" % (a*100))
    #print(testdata.labels)
    #print(x)
    
  

    

    
    
def nptest():
    a = np.zeros((7,1))
    a[3,0] = 1
    print(np.argmax(a))

    #print(np.random.random((8,4)))

if __name__ == '__main__':
    ops()
