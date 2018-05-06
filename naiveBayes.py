import numpy as np 

class NaiveBayes(object):
    """
    feature domains should be a list of domain of each feature.
    E.g. there are three features for this model, f1,f2,f3
    domain for f1,f2,f3 = [0,1,2]   
    feature_domains would be [ [0,1,2], [0,1,2], [0,1,2]]
    with this, algorithum is more adjustable to varity of feature 
    distrubution 
    labels should be be a list of labels, e.g. ['1','2','3' ...]
    leaened is a list of matrixs, where each matrix is coresponding 
    to a feature with same index in feature_domains
    inside each matrix, first axis is array to each domain value of
    that feature, and inside the array, is the count of each label
    in general, in learned, feature -> feature_value -> label to get 
    the count
    """
    def __init__(self, feature_domains, labels, k):
        self.labels = labels
        self.learned = []
        self.feature_domains = feature_domains
        for one_feature in feature_domains:
            self.learned.append(np.zeros((len(one_feature), len(labels))))
        self.label_count = np.zeros(len(labels))
        self.total = 0
        self.k     = k

    def train(self, features_list, labels):
        #loop to go over input
        for i in range(len(features_list)):
            features = features_list[i]
            features = np.array(features).flatten()
            label = labels[i]
            label_index = self.labels.index(label)
            self.label_count[label_index] += 1
            self.total += 1
            for f in range(len(self.feature_domains)):
                feature_value = features[f]
                feature_index = self.feature_domains[f].index(feature_value)
                self.learned[f][feature_index][label_index] += 1


    def classify(self, features_list):
        o = []
        for features in features_list:
            dist = self.calculateP(np.array(features).flatten())
            o.append(self.labels[np.argmax(dist)])
        return o

    def calculateP(self, features):
        dist = []
        for i in range(len(self.labels)):
            label_count = self.label_count[i]
            temp = np.log(label_count/self.total)
            for f in range(len(features)):
                feature_index = self.feature_domains[f].index(features[f])
                temp += np.log((self.learned[f][feature_index][i] + self.k)/ (label_count + self.k*len(self.learned[f]))) 
            dist.append(temp)
        return dist


