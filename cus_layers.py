# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:37:47 2020

@author: Mustafa Hajij
"""

import math
import numpy as np

from tensorflow  import keras


from tensorflow.python.ops import math_ops
import tensorflow as tf

  
import tensorflow

import os
os.system('cls')
class slice_layer(tensorflow.keras.layers.Layer):
    def __init__(self,a=0,b=0,**kwargs):
        
        super(slice_layer, self).__init__(**kwargs)

        self._a=a
        self._b=b
    

    def call(self, inputs):
                
        return tf.slice(inputs,[self._a,self._b],[-1,1]) 


class slice_layerA(tensorflow.keras.layers.Layer):
    def __init__(self,a=0,b=0,c=0,d=0,**kwargs):
        
        super(slice_layerA, self).__init__(**kwargs)

        self._a=a
        self._b=b
        self._c=c
        self._d=d    

    def call(self, inputs):
                
        return tf.slice(inputs,[self._a,self._b],[self._c,self._d]) 

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            '_a': self._a,
            '_b': self._b,
            '_c': self._c,
            '_d': self._d,
        })
        return config



class matrix_tensor(tensorflow.keras.layers.Layer):
    def __init__(self):
        
        super(matrix_tensor, self).__init__()


    def call(self,a,b):
                
        return tf.tensordot( a, b, axes=0)




class eye(tensorflow.keras.layers.Layer):
    def __init__(self,size):
        
        super(eye, self).__init__()
        self.size=size

    def call(self):
                
        return tf.keras.backend.eye(self.size )


      
  