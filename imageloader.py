import os
class ImageDataSet:
    """
    Object to hold data of images and label
    height -> image height
    width  -> image width
    number -> number of image in object

    """
    def __init__(self, height, width):
        self.height = height
        self.width = width
    
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


def charToint(char):
    if char == ' ':
        return 0
    elif char == '+':
        return 1
    elif char == '#':
        return 2





