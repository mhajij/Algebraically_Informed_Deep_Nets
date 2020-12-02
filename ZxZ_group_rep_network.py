# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:09:49 2020

@author: Mustafa Hajij
"""

import math
import numpy as np


from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, Flatten, Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import InputLayer


from tensorflow.python.ops import math_ops
import tensorflow as tf

import os
os.system('cls')


class slice_layer(tf.keras.layers.Layer):
    def __init__(self,a=0,b=0):
        
        super(slice_layer, self).__init__()

        self._a=a
        self._b=b
    

    def call(self, inputs):
                
        return tf.slice(inputs,[self._a,self._b],[-1,1]) 


class slice_layerA(tf.keras.layers.Layer):
    def __init__(self,a=0,b=0,c=0,d=0):
        
        super(slice_layerA, self).__init__()

        self._a=a
        self._b=b
        self._c=c
        self._d=d    

    def call(self, inputs):
                
        return tf.slice(inputs,[self._a,self._b],[self._c,self._d]) 




def a_operator(input_dim=2,activation_function='linear',bias=True):


    inputs = Input(shape=(input_dim,))

    x=Dense(2*input_dim+2, use_bias=bias,activation=activation_function)(inputs)
    x=Dense(2*input_dim+2, use_bias=bias,activation=activation_function)(x)

    x=Dense(100,use_bias=bias, activation=activation_function)(x)

    x=Dense(50, use_bias=bias,activation=activation_function)(x)

    if activation_function!='linear':
        last='sigmoid'
    else:
        last=activation_function    
    predictions=Dense(input_dim,use_bias=bias, activation=last ,name='final_output')(x)



    model = Model(inputs=inputs, outputs=predictions)
    
    return model




def ZxZ_group_rep_net(a_op,b_op,input_shape=2):
        
    
    input_tensor_1=Input(shape=(input_shape,))
        
    



    R_output=b_op(input_tensor_1)     
    side_1_equation_1=a_op(R_output)
    

    R_output=a_op(input_tensor_1)    
    side_2_equation_1=b_op(R_output)

    
    output_tensor=Concatenate()([side_1_equation_1,side_2_equation_1])    
    M=Model(inputs=[input_tensor_1],outputs=output_tensor) 
    
    return M


def ZxZ_group_rep_loss(input_dim=2):
    
    def loss(y_true,y_pred):
        
        equation_1_out=tf.slice(y_pred,[0,0],[-1,input_dim])
        equation_2_out=tf.slice(y_pred,[0,input_dim],[-1,input_dim])
        
        A=K.mean(math_ops.square(equation_1_out - equation_2_out), axis=-1) 
                    
        return A
    
    return loss



