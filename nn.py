import numpy as np

class NeuralNetwork():
	"""docstring for NeuralNetwork"""
	def __init__(self, number_input, number_hidden, labels):
		self.input_num = number_input
		self.neuron_input  = [InputNeuron() for _ in range(number_input)]
		self.neuron_hidden = [Neuron('hidden layer',self.neuron_input, 0.5, 0.5) for _ in range(number_hidden)]
		self.neuron_output = [Neuron('output layer',self.neuron_hidden, 0.5, 0.2) for _ in range(len(labels))]
		self.labels = labels

	def train_one_time(self, data, l):
		#feed in data forward
		for inputn, number in zip(self.neuron_input, data):
			inputn.put(number)

		for hn in self.neuron_hidden:
			hn.forward()

		for i in range(len(self.labels)):
			on = self.neuron_output[i]
			ol = self.labels[i]
			result = on.forward()
			target = 1.0 if ol == l else 0.0
			on.recvfeed(-(target - result))
			on.updatedelta()
			on.updateweights()
			on.feedback()

		for nh in self.neuron_hidden:
			nh.updatedelta()
			nh.updateweights()

	def train(self, data_list, label_list, train_time):
		print("NeuralNetwork Start training process")
		for x in range(train_time):
			for i in range(len(data_list)):
				print("NeuralNetwork now training the %d data" %i, end='\r')
				self.train_one_time(np.array(data_list[i]).flatten(), label_list[i])

	def classify_one_time(self, data):
		for inputn, number in zip(self.neuron_input, data):
			inputn.put(number)

		for hn in self.neuron_hidden:
			a = hn.forward()
		temp = float('-inf')
		index = 0

		
		for i in range(len(self.neuron_output)):

			re = self.neuron_output[i].forward()
			#print(self.labels[i], re)
			if temp < re:
				index = i 
				temp = re

		return self.labels[index]

	def classify(self, data_list):
		out = []
		for data in data_list:
			out.append(self.classify_one_time(np.array(data).flatten()))

		return out


class InputNeuron(object):
	def __init__(self):
		self.output = 0

	def put(self, inputdata):
		self.output = inputdata /1000


class Neuron(object):
	"""docstring for HiddenNeural"""
	def __init__(self, name, pre_Nerons, bais, ratio):
		self.name      = name
		self.pre_neron = pre_Nerons
		self.weights   = np.random.random(len(pre_Nerons))
		self.bais      = bais
		self.delta     = float("-inf")
		self.output    = float("-inf")
		self.upratio   = ratio


	def forward(self):
		inputs = [i.output for i in self.pre_neron]
		self.output = SomeFunction(np.dot(inputs, self.weights))
		#print(self.name, self.output)
		self.back_delta = 0
		return self.output

	def recvfeed(self, number):
		self.back_delta += number

	def feedback(self):
		for n,w in zip(self.pre_neron, self.weights):
			n.recvfeed(self.delta * w)

	def updatedelta(self):
		self.delta = self.back_delta * self.output *(1- self.output)

	def updateweights(self):
		for i in range(len(self.weights)):
			self.weights[i] = self.weights[i] - self.upratio*self.delta*self.pre_neron[i].output


		

def SomeFunction(a):
	return 1.0/(1.0 + np.exp(-a))