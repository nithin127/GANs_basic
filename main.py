import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation

def targetDistribution(n = 10):
	_mean = 5
	_std_deviation = 1.2
	return np.random.normal(_mean,_std_deviation,n)

class generator():
	def __init__ (self, hidden_units = 10):
		# We are defining a 2 layered model for G
		self.model = Sequential([Dense(hidden_units, batch_input_shape = (None, 1), activation = 'softmax'),							
								Dense(1)])	
		self.model.compile(loss = 'mse', optimizer = 'sgd')
	def eval(self, x):
		return self.model(x)

class discriminator():
	def __init__ (self, hidden_units = 20):
		# We are defining a 2 layered model for G
		self.model = Sequential([Dense(hidden_units, batch_input_size = (None, 1), activation = 'tanh'),
							Dense(hidden_units, activation = 'tanh'),
							Dense(hidden_units, activation = 'tanh'),
							Dense(1, activation = 'tanh')])
		self.model.compile(loss = 'mse', optimizer = 'sgd')

G = generator()
D = discriminator()