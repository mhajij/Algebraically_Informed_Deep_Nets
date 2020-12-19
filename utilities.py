# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 08:32:09 2020

@author: Mustafa Hajij
"""

from tensorflow  import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import ModelCheckpoint


def Id(x):
    return x 

def identity(dim=2):
    
    inputs = Input(shape=(dim,))
    
    out = Lambda(Id, name="identity")(inputs)
    
    model=Model(inputs,out)
    
    return model



def operator(input_dim=2,activation_function='linear',bias=True):
    
    """
    this is a network that can used to represent element in a given presentation. The choice is the one made in the paper.    
    """


    inputs = Input(shape=(input_dim,))

    x=Dense(2*input_dim+2, use_bias=bias,activation=activation_function)(inputs)
    x=Dense(2*input_dim+2, use_bias=bias,activation=activation_function)(x)

    x=Dense(100,use_bias=bias, activation=activation_function)(x)

    x=Dense(50, use_bias=bias,activation=activation_function)(x)
  
    predictions=Dense(input_dim,use_bias=bias, activation='linear' ,name='final_output')(x)



    model = Model(inputs=inputs, outputs=predictions)
    
    return model


def get_n_operators(dim,activation_function,bias,n_of_operators):
    
    out=[operator(input_dim=dim,activation_function=activation_function,bias=bias) for i in range(0,n_of_operators)]

    return out   


def train_net(model,x_data,y_data,model_name,lossfunction,lr,batch_size,epochs,):
 
    
    checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    
    callbacks_list = [checkpoint]   
    
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss = lossfunction  )           
    
    model.fit(x_data, y_data,  batch_size=batch_size, epochs=epochs, shuffle = True,  verbose=1,callbacks=callbacks_list)  
    
    return 1
    
def get_relation_tensor(modelpath,model,data):
    
    model.load_weights(modelpath)

    return model.predict(data)
    
    