import numpy as np 

class NaiveBayes(object):
    """docstring for NaiveBayes"""
    def __init__(self, width, height, labels):
        self.labels = labels
        self.learned = np.zeros((len(labels), width, height))
        self.count = np.zeros(len(labels))
        self.total = 0

    def train(self, images, labels):
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            index = labels.index(label)
            self.learned[index] = self.learned[index] + image
            self.count[index] += 1
            self.total += 1


    def classify(self, images):
        o = []
        for image in images:
            temp = 0
            d = self.labels[0]
            for l in self.labels:
                


def calculateP(matrix, ):
    pass

