import numpy as np

class NeuralNetwork():
	def __init__(self, structure, labels):
		self.input_number = structure[0]
		self.layers = []
		self.labels = labels
		i = self.input_number
		for s in structure[1:-1]:
			self.layers.append(HiddenLayer('Hidden Layer', s, i))
			i = s
		self.layers.append(HiddenLayer('Output Layer', structure[-1], i))

	def forward(self, data):
		out = data
		for l in self.layers:
			out = l.forward(out)

		
		return out

	def desired_output(self, label):
		out = np.zeros(len(self.labels)).reshape((-1,1))
		out[self.labels.index(label),0] = 1.0
		return out

	def stochastic_gradient_descent(self, data, label, ratio):

		x = self.forward(data)

		C_bais = [None for _ in range(len(self.layers))]
		C_weights = [None for _ in range(len(self.layers))]

		outputlayer = self.layers[-1]
		target      = self.desired_output(label)

		delta = (outputlayer.out - target) * Sigmoid_prime(outputlayer.raw)

		C_bais[-1] = delta
		C_weights[-1] = np.dot(delta, self.layers[-2].out.transpose())

		for i in range(2,len(self.layers)):
			layer = self.layers[-i]
			delta = np.dot(self.layers[-i+1].weights.transpose(), delta) * Sigmoid_prime(layer.raw)
			C_bais[-i] = delta
			C_weights[-i] = np.dot(delta, self.layers[-i-1].out.transpose())

		delta = np.dot(self.layers[1].weights.transpose(), delta) * Sigmoid_prime(self.layers[0].raw)
		C_bais[0] = delta
		C_weights[0] = np.dot(delta, data.transpose())


		#update
		for i in range(len(C_bais)):
			self.layers[i].weights = self.layers[i].weights - ratio * C_weights[i]
			self.layers[i].bais    = self.layers[i].bais    - ratio * C_bais[i]

	def train(self, data_list, label_list ,time, ratio):
		for t in range(time):
			print("Starting Training, Time : %d th" %(t+1))
			for i in range(len(data_list)):
				print("Training data number %d " %(i+1), end='\r')
				self.stochastic_gradient_descent(np.reshape(data_list[i], (-1, 1)), label_list[i], ratio)

	def classify_one_time(self, data):
		a = self.forward(data)
		return self.labels[np.argmax(a)]

	def classify(self, data_list):
		o = []
		for data in data_list:
			o.append(self.classify_one_time(np.reshape(data, (-1, 1))))
		return o


class HiddenLayer():
	def __init__(self, name , neron_number, input_number):
		self.name = name
		self.weights = np.zeros((neron_number, input_number))
		self.bais = np.random.random(neron_number).reshape((-1,1))

	def forward(self, inputs):
		self.raw = np.dot(self.weights, inputs) + self.bais
		self.out = Sigmoid(self.raw)
		return self.out
	


def Sigmoid(x):
	return 1.0/(1.0 + np.exp(-x)) 

def Sigmoid_prime(x):
	return Sigmoid(x)*(1.0 - Sigmoid(x))