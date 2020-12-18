# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 08:32:09 2020

@author: Mustafa Hajij
"""

from tensorflow  import keras



def train_net(model,x_data,y_data,lossfunction,callbacks_list,lr,batch_size,epochs):
    
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss = lossfunction  )
            
    model.fit(x_data, y_data,  batch_size=batch_size, epochs=epochs, shuffle = True,  verbose=1,callbacks=callbacks_list)  