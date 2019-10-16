import tensorflow as tf
import numpy as np
from tf import keras
from tf.keras import layers
from keras.layers import Input, Dense
from keras.models import Model

class RLSeekerCriticAgent:
	def __init__(self):
		state_input = Input(shape = (2,), name = 'State_In')
		action_input = Input(shape = (2,), name = 'Action_In')

		sa_concat = layers.concatenate([state_input,action_input])
		quadLayer = LinQuadLayer()(sa_concat)
		fc_linear = layers.Dense(1)(sa_concat)
		fc_quad = layers.Dense(1)(quadLayer)
		addLayer = layers.Add()([fc_linear, fc_quad])

		reward_estimator = Model(inputs = [state_input, action_input], outputs = addLayer)









class LinQuadLayer(layers.Layer):
	def __init__(self):
		super(LinQuadLayer,self).__init__()
		self.numOutputs = 1

	def call(self,inputs):
		if inputs.shape[0] < 1:
			raise ValueError("Cannot have layer input size less than 1")
		u,v = tf.meshgrid(inputs,inputs)
		quadLayer = u * v;
		r,c = quadLayer.shape
		quadLayer = (u * v).numpy();
		op = np.zeros([self.__triangle(inputs.shape[0])])
		for i in range(r):
			for j in range(i):
				op[i * inputs.shape[0] + j] = quadLayer[i][j]

		return tf.convert_to_tensor(op)

	def __triangle(self,N):
		return (N**2 + N)/2





