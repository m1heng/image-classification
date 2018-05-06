import os
import random
class ImageDataSet:
    """
    Object to hold data of images and label
    height -> image height
    width  -> image width
    number -> number of image in object

    """
    def __init__(self, height, width, labeldomain=None):
        self.height = height
        self.width = width
        self.number = 0
        if labeldomain != None:
            self.labeldomain = labeldomain


    def loadDataSet(self, imagepath, labelpath, n):
        self.loadImageData(imagepath, n)
        self.loadLabelData(labelpath, n)


    
    '''
    if n = -1, read all image in file and record number of image loaded 
    '''
    def loadImageData(self, imagefilename, n):
        self.images = []
        if n != -1:
            self.number = n
            with open(imagefilename, 'r') as f:
                for x in range(n):
                    one_image = []
                    for y in range(self.height):
                        line = f.readline()
                        if len(line) != self.width + 1:
                            raise Expection("Inavalid ImageData")
                        one_image.append(list(map(charToint,list(line)[:-1])))
                    self.images.append(one_image)
                f.close()
        else:
            number_count = 0
            with open(imagefilename, 'r') as f:
                one_image = []
                count = 0
                for line in f:
                    one_image.append(list(map(charToint,list(line)[:-1])))
                    count += 1
                    if count == self.height:
                        self.images.append(one_image)
                        one_image = []
                        number_count += 1
                        count = 0
                self.number = number_count
                f.close()
    def loadLabelData(self, labelfilename, n):
        if n != self.number:
            print("Unequal number of image and label, will cause problem")
        self.labels = []
        count = 0
        with open(labelfilename, 'r') as f:
            for line in f:
                self.labels.append(list(line)[0])
                count += 1
                if count == n:
                    break
            f.close()
        if count != n:
            print("load fewer label than input")


    '''
    Shuffle all pairs of image and label 
    '''
    def shuffle(self):
        temp = list(zip(self.images, self.labels))
        random.shuffle(temp)
        self.images , self.labels = zip(*temp)

    '''
    Return first n percent of data 
    '''
    def orderedout(self, precentage):
        if precentage > 1:
            precentage = precentage * 0.01
        last = int(self.number*precentage -1)
        return self.images[:last], self.labels[:last]


    '''
    return random n precent of data, without change the order of origin data
    '''
    def shuffleout(self, precentage):
        if precentage > 1:
            precentage = precentage * 0.01
        temp = list(zip(self.images, self.labels))
        random.shuffle(temp)
        last = int(self.number*precentage -1)
        temp = temp[:last]
        images , labels = zip(*temp)

        return images,labels



def charToint(char):
    if char == ' ':
        return 0
    elif char == '+':
        return 0.5
    elif char == '#':
        return 1





