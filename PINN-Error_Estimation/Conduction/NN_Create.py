
# Import
import tensorflow as tf
import numpy as np
from scipy import optimize as opt
from time import time
import math

from General_Functions import *

def Set_NN_Model(Case_Info, CD):
    [Load_NN, Loaded_NN_Name] = Case_Info.Load_Loaded_NN_Info()
    if Load_NN:
        NN_Model = tf.keras.models.load_model(Loaded_NN_Name)
    else:
        All_Neurons = Case_Info.Load_NN_Size()
        NN_Model = PINN_NeuralNet(CD.n_Output, [CD.Overall_lb, CD.Overall_ub], All_Neurons)
        NN_Model.build(input_shape=(None, CD.n_Input))
    CD.GE.Set_Model(NN_Model)

class PINN_NeuralNet(tf.keras.Model):
    def __init__(self, N_Output, Domain, All_Neurons,
                 activation='tanh',
                 kernel_initializer='glorot_normal',
                 **kwargs):
        super().__init__(**kwargs)

        self.NN_Layers = len(All_Neurons)
        self.lb = tf.constant(Domain[0])
        self.ub = tf.constant(Domain[1])

        # Define NN architecture
        self.scale = tf.keras.layers.Lambda(
            lambda x: 2.0 * (x - self.lb ) /(self.ub - self.lb) - 1.0)
        self.hidden = [tf.keras.layers.Dense(All_Neurons[_],
                                             activation=tf.keras.activations.get(activation),
                                             kernel_initializer=kernel_initializer)
                       for _ in range(self.NN_Layers)]
        self.out = tf.keras.layers.Dense(N_Output)

    def call(self, X):
        Z = self.scale(X)
        for i in range(self.NN_Layers):
            Z = self.hidden[i](Z)
        return self.out(Z)


