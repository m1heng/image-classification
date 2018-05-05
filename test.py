from imageloader import ImageDataSet
from perceptron import Perceptron
from util import *
from simpleNN import NeuralNetwork
from naivebayes import NaiveBayes
import numpy as np 
import matplotlib.pyplot as plt

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
    
    testdata = ImageDataSet(28,28)
    testdata.loadImageData("digitdata/testimages", -1)
    testdata.loadLabelData("digitdata/testlabels", testdata.number)
    data = ImageDataSet(28,28)
    data.loadImageData("digitdata/trainingimages", -1)
    data.loadLabelData("digitdata/traininglabels", data.number)
    for t in range(20,101,20):
        images, labels = data.shuffleout(t)
        al = []
        il = []
        pc = Perceptron(28*28,['0', '1','2','3','4','5','6','7','8','9'])
        for i in range(100):
            pc.train(images,labels, 1,0.8)
            x = pc.classify(testdata.images)
            a = Accuracy(x, testdata.labels)
            al.append(a*100)
            il.append(i+1)
            print(a*100)
        plt.plot(il, al, label="size=%d"%(t*0.01*data.number) )

    leg = plt.legend( ncol=2, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.xlim([1,100])
    plt.xlabel("trainning time")
    plt.ylabel("accuracy")

    plt.show()


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
        nn.train(data.images, data.labels, 1, 0.3)    
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
    nn = NeuralNetwork( ( 70*60,  30,25,25,2 ), ['0', '1'])
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

def dataloader_face(n_train = -1, n_test = -1):
    data = ImageDataSet(70,60, labeldomain=['0', '1'])
    data.loadImageData("facedata/facedatatrain", n_train)
    data.loadLabelData("facedata/facedatatrainlabels", data.number)
    testdata = ImageDataSet(70,60)
    testdata.loadImageData("facedata/facedatatest", n_test)
    testdata.loadLabelData("facedata/facedatatestlabels", testdata.number)   
    return data, testdata 

def dataloader_digit(n_train = -1, n_test = -1):
    data = ImageDataSet(28,28, labeldomain=['0', '1','2','3','4','5','6','7','8','9'])
    data.loadImageData("digitdata/trainingimages", n_train)
    data.loadLabelData("digitdata/traininglabels", data.number)
    testdata = ImageDataSet(28,28)
    testdata.loadImageData("digitdata/testimages", n_test)
    testdata.loadLabelData("digitdata/testlabels", testdata.number)
    return data, testdata

def test_preceptorn(traindata, testdata):
    print("Initializing Perceptron")
    times = int(input("max trian time for one data set: "))
    ratio = float(input("learning ratio: "))
    totalnumber = traindata.number
    #first - try with ordered datas 
    for p in range(10,101,10):
        images, labels = traindata.orderedout(p)
        al = []
        il = []
        pc = Perceptron(traindata.width*traindata.height, traindata.labeldomain)
        for i in range(times):
            pc.train(images, labels, 1, ratio)
            x = pc.classify(testdata.images)
            a = Accuracy(x, testdata.labels)
            al.append(a*100)
            il.append(i+1)
            print(a*100)
        plt.plot(il, al, label="size=%d"%(p*0.01*totalnumber) )

    leg = plt.legend( ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.xlim([1,times])
    plt.xlabel("trainning time")
    plt.ylabel("accuracy")
    plt.show()

def test_nueralnetwork(traindata, testdata):
    times = int(input("max trainning time for one data set :"))
    ratio = float(input("learning ratio: "))
    totalnumber = traindata.number
    #first - try with ordered datas 
    for p in range(10,101,10):
        images, labels = traindata.orderedout(p)
        al = []
        il = []
        nn = NeuralNetwork((traindata.width*traindata.height,15,15, len(traindata.labeldomain)), traindata.labeldomain)
        for i in range(times):
            print("Train %d time"%i)
            nn.train(images, labels, 1, ratio)
            x = nn.classify(testdata.images)
            a = Accuracy(x, testdata.labels)
            al.append(a*100)
            il.append(i+1)
            print(a*100)
        plt.plot(il, al, label="size=%d"%(p*0.01*totalnumber) )

    leg = plt.legend( ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.xlim([1,times])
    plt.xlabel("trainning time")
    plt.ylabel("accuracy")
    plt.show()

def test_nueralnetwork_w(traindata, testdata, times, ratio, file):
    totalnumber = traindata.number
    #first - try with ordered datas 
    j = []
    for p in range(10,101,10):
        images, labels = traindata.orderedout(p)
        al = []
        il = []
        nn = NeuralNetwork((traindata.width*traindata.height, 15,15, len(traindata.labeldomain)), traindata.labeldomain)
        for i in range(times):
            print("Train %d time"%i)
            nn.train(images, labels, 1, ratio)
            x = nn.classify(testdata.images)
            a = Accuracy(x, testdata.labels)
            al.append(a*100)
            il.append(i+1)
            print(a*100)
        j.append((al, il))
    import json
    with open(file, 'w') as f:
        json.dump(j, f)



def test_naivebayes(traindata, testdata):
    #raw pixel feature
    feature_domians = [[i for i in np.arange(0,1.1,0.5)] for _ in range(traindata.width * traindata.height)]
    for p in range(10, 101, 10):
        print("Training with %d"%int(p * traindata.number * 0.01))
        nb = NaiveBayes(feature_domians, traindata.labeldomain , 1)
        images, labels = traindata.orderedout(p)
        nb.train(images, labels)
        x = nb.classify(testdata.images)
        a = Accuracy(x, testdata.labels)
        print(a)


def main():
    print("Image Classification Program, coded by H M L for CS440 final project")
    while True:
        train = None
        test  = None
        while True:
            print("Please pick a data set to be use: ")
            print("1. Digit, 2. Face")
            choice = input("(enter 1 or 2 or 3, or Q to exit): ")
            if choice == '1':
                print("Digit data picked, loading data")
                train, test = dataloader_digit()
                break
            elif choice == '2':
                print("Face data picked, loading data")
                train, test = dataloader_face()
                break
            elif choice == 'Q':
                print("Have a nice day, lol")
                return 
            else:
                print("Wrong input XD, try again")

        while True:
            print("Please pick one algorithum from following:")
            print("1. Naive Bayes Classifier")
            print("2. Perceptorn")
            print("3. Neural Network")
            algorithum = input("(enter 1 or 2 or 3, or Q to exit): ")
            if algorithum == '1':
                test_naivebayes(train, test)
            elif algorithum == '2':
                test_preceptorn(train, test)
            elif algorithum == '3':
                test_nueralnetwork(train, test)
            elif algorithum == 'Q':
                return 

            

if __name__ == '__main__':
    main()
    #test_preceptorn(train, test, 10, 1)
    #test_nueralnetwork(train, test, 20, 0.5)
    #test_naivebayes(train, test)
