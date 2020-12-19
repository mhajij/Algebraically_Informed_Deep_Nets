# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 08:32:09 2020

@author: Mustafa Hajij
"""

from tensorflow  import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Lambda


def Id(x):
    return x 

def identity(dim=2):
    
    inputs = Input(shape=(dim,))
    
    out = Lambda(Id, name="identity")(inputs)
    
    model=Model(inputs,out)
    
    return model



def operator(input_dim=2,activation_function='linear',bias=True):


    inputs = Input(shape=(input_dim,))

    x=Dense(2*input_dim+2, use_bias=bias,activation=activation_function)(inputs)
    x=Dense(2*input_dim+2, use_bias=bias,activation=activation_function)(x)

    x=Dense(100,use_bias=bias, activation=activation_function)(x)

    x=Dense(50, use_bias=bias,activation=activation_function)(x)
  
    predictions=Dense(input_dim,use_bias=bias, activation='linear' ,name='final_output')(x)



    model = Model(inputs=inputs, outputs=predictions)
    
    return model


def train_net(model,x_data,y_data,lossfunction,callbacks_list,lr,batch_size,epochs):
    
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss = lossfunction  )
            
    model.fit(x_data, y_data,  batch_size=batch_size, epochs=epochs, shuffle = True,  verbose=1,callbacks=callbacks_list)  