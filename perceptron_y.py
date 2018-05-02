import numpy as np

class Perceptron():
	def __init__(self, feature_number, labels):
		self.name = "Perceptron"
		self.weights = np.zeros((len(labels), feature_number))
		self.w0 = np.zeros(len(labels))
		self.labels = labels

	def train(self,feature_list, label_list , max_train_time):
		print("Start Perceptron trainning")
		vectors = feature_list
		for time in range(max_train_time):
			print("Training time: %d" % time, end='\r')
			for i in range(len(feature_list)):
				feature = feature_list[i]
				feature = np.array(feature).flatten()
				label = label_list[i]
				for l in range(len(self.labels)):
					label_c = self.labels[l]
					weight = self.weights[l]
					f = Magic(feature, weight) 
					if f < 0 and label == label_c:
						for j in range(len(weight)):
							weight[j] = weight[j] + feature[j]
					elif f >=0 and label != label_c:
						for j in range(len(weight)):
							weight[j] = weight[j] - feature[j]

	def classify(self, datalist):
		result = []
		for image in datalist:
			data = np.array(image).flatten()
			temp = float("-inf")
			index = 0
			for l in range(len(self.labels)):
				weight = self.weights[l]
				m = Magic(data, weight) 
				if m >= temp:
					temp = m
					index = l
			result.append(self.labels[index])
		return result



def Magic(x,w):
	return np.dot(x, w)

def Phi(x):
	return x