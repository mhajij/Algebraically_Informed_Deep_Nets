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
from tensorflow.keras.layers import Concatenate

import cus_layers as cl


def Id(x):
    
    """
    simple identity function
    
    """
    
    return x 

def identity(dim=2):
    
    """
    A network of the form id : R^n->R^n that gives identity(x)=x. Namely the identity network. 
    
    Purpose : Used for in mathematical functional equations that have identity functions.
    Examples are the Yang-Baxter Equation : (IdXR)*(RXId)*(IdXR)=(RXId)*(IdXR)*(IdXR) 
    
    """    
    
    inputs = Input(shape=(dim,))
    
    out = Lambda(Id, name="identity")(inputs)
    
    model=Model(inputs,out)
    
    return model


def generator(input_dim=2,activation_function='linear',bias=True):
    
    """
    Purpose: 
    -------    
        
        This is a network that can used to represent element in a given presentation. The choice is the one made in the paper.    
 
       
    Arguments: 
    ---------    
        dim : integer dimension of the input
        activation_function : the activation function used in network.
        bias: determines if the network has bias. 
        * The choice of activation_function and bias determines the type of the representation.
         - When bias=False and activation_function='linear' the represenation is linear.
         - When bias=True and activation_function='linear' the represenation is linear.
         - When activation_function is not linear, the represenation is not linear.

    Return: 
    -------    
        A neural network with the following properties :
        (1) input dimension=output dimension
        (2) used to represent an algebraic element.
          
    """


    inputs = Input(shape=(input_dim,))

    x=Dense(2*input_dim+2, use_bias=bias,activation=activation_function)(inputs)
    x=Dense(2*input_dim+2, use_bias=bias,activation=activation_function)(x)

    x=Dense(100,use_bias=bias, activation=activation_function)(x)

    x=Dense(50, use_bias=bias,activation=activation_function)(x)
  
    predictions=Dense(input_dim,use_bias=bias, activation='linear' ,name='final_output')(x)



    model = Model(inputs=inputs, outputs=predictions)
    
    return model


def tensor_generator_with_identity(generator, inputs, gen_position=2, total_dimension=3,input_dim=2 ):
    
    
    """
    Purpose:
    --------    
        If we have a map gen=(R^n)^2->(R^n)^2 this function generates a function of the form 
         
        generator_i=id^{i-1} X generator X id^{m-i+1}, and U is a the two hooks shape curve in the TL generator Ui.
         
        where id: (R^n)->(R^n)      
        
        then for a subset of the domain  (inputs), it returns generator_i(input) 
        

    
    Parameters:
    -----------    
        Ugen : Keras model (R^n)^2->(R^n)^2, used to define the generator in some algebraic structure
        
               Ui=id^{i-1} X Ugen X id^{m-i+1} 
               
        inputs: input tensor
        
        gen_position: Integer,  this is the index "i" in generator_i
        
        total_dimension : Integer, this determines how many tensors in id^{i-1} X generator X id^{m-i+1}.
        
        input_dim: Integer, this is the dimension n on which the function Ugen : network (R^n)^2->(R^n)^2 is defined. 
        
    Returns:
    --------      
       The tensors list generator_i(input), where generator_i=id^{i-1} X generator X id^{m-i+1}
        

    """

    con_list=[]
        
    generator_input=inputs[gen_position-1:gen_position+1] 
    
    tenosor=Concatenate()(generator_input)
    
    out_gen=generator(tenosor)
        
    
    
    for i in range(0,gen_position-1):
        con_list.append(inputs[i])
    
    con_list.append(out_gen)
        
    for i in range(gen_position+1,total_dimension):        
        con_list.append(inputs[i])
                
            
    con_final=Concatenate()(con_list)  
    
    out_tensors_list=[]
        
    for i in range(0,total_dimension):       
        x_i=cl.SliceLayer(0,i*input_dim,-1,input_dim)(con_final)

        out_tensors_list.append(x_i)
    
    return out_tensors_list 




def get_n_operators(dim,activation_function,bias,n_of_operators):
    
    """
    Parameters: 
    ----------  
        dim : integer dimension of the input
        activation_function : the activation function used in network.
        bias: determines if the network has bias. 
        n_of_operators : number of neural network.
        
    Returns:
    --------    
        a list of n_of_operators networks. 
    """
    
    out=[generator(input_dim=dim,activation_function=activation_function,bias=bias) for i in range(0,n_of_operators)]

    return out   


def train_net(model,x_data,y_data,model_name,lossfunction,lr,batch_size,epochs):
    
    """
    Parameters: 
    ----------            
        model : keras model
        x_data : training X data
        y_data : training Y data
        model_name : name of the file where the model is going to be saved.
        lr: learning rate
        batch_size : batch size
        epochs : number of epochs
    
    """     
    
    checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    
    callbacks_list = [checkpoint]   
    
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss = lossfunction  )           
    
    model.fit(x_data, y_data,  batch_size=batch_size, epochs=epochs, shuffle = True,  verbose=1,callbacks_list=callbacks_list)  
    
    return 
    
def get_relation_tensor(modelpath,model,data):
    
    """
    Parameters:
    ----------    
        modelpath: folder where the model weights are located.
        model : keras model
        data : the data that we want to infer the model on.
    Returns:
    -------    
        the prediction of the input model on the input data.
            
    """
    
    model.load_weights(modelpath)

    return model.predict(data)
    
    