# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:01:54 2020

@author: Mustafa Hajij
"""


import knot_invariants_deep_net as ki

from tensorflow.keras.callbacks import ModelCheckpoint


from tensorflow  import keras

import tensorflow as tf
import argparse

import numpy as np


      
        
    
    
    

if __name__ == '__main__':

    
    
    parser = argparse.ArgumentParser()

    # testing or training argument

    parser.add_argument('-m', '--mode',type=str,default='training',required=True,help='Specify if you want to train a network or testing it.')   



    #network arguments
    
    #(1) dim of the rep
    parser.add_argument('-dim', '--iden_dimension',type=int,default=2,help='domain dimension of the identity net.')

  
 
    #__________________________________________________________

    # training arguments
    #(1) number of epochs
    
    parser.add_argument('-e', '--epoch',type=int,default=2000,help='Number of epochs.')
    
    #(2) learning rate
    
    parser.add_argument('-lr', '--learning_rate',type=float,required=False,default=0.002,help='learning rate.')

    #(3) batchsize

    parser.add_argument('-b', '--batch_size',type=float,required=False,default=2000,help='batch_size.')

    #___________________________________________________________



    args = parser.parse_args()
    
    
    
    if args.mode=='training':
        

        model=ki.RT_training_net(id_dim=args.iden_dimension,loop_constant=2)
        
        file='RT_data_dim='+str(args.iden_dimension**3)+'.npy'
        
        data1=np.load(file)[:50000]


        model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate), loss = ki.RT_loss(args.iden_dimension))

        model_name="knot_model_dim="+str(args.iden_dimension)+".h5"
 
        checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True, mode='min')

        callbacks_list = [checkpoint]
        
        history=model.fit(data1, data1[:,:args.iden_dimension*args.iden_dimension],
                  batch_size=args.batch_size,
                  epochs=args.epoch,
                  shuffle = True,
                  verbose=1,callbacks=callbacks_list)  


        
        
    else:
        if args.iden_dimension not in [2,4]:
            raise ValueError("dim must be 2 or 4.")
    
        
        file='RT_data_dim='+str(args.iden_dimension**3)+'.npy'
        
        model=ki.RT_training_net(id_dim=args.iden_dimension,loop_constant=2)
        
        data1=np.load(file)
                
        model.compile(optimizer=keras.optimizers.Adam(lr=args.learning_rate), loss = ki.RT_loss(args.iden_dimension))

        model_name="knot_model_dim="+str(args.iden_dimension)+".h5"
      
        model.load_weights(model_name)
        
        d=args.iden_dimension

        R,R_inv,n,u=ki.get_weights(model,args.iden_dimension)

        id_dim=args.iden_dimension        
        
        eye=np.eye(args.iden_dimension)
        
        R1=ki.tensor(R,eye,(id_dim**3,id_dim**3))
        R2=ki.tensor(eye,R,(id_dim**3,id_dim**3))


        R1_inv=ki.tensor(R_inv,eye,(id_dim**3,id_dim**3))
        R2_inv=ki.tensor(eye,R_inv,(id_dim**3,id_dim**3))
        
        N1=ki.tensor(n,eye,(id_dim,id_dim**3))
        N2=ki.tensor(eye,n,(id_dim,id_dim**3))


        U1=ki.tensor(u,eye,(id_dim**3,id_dim))
        U2=ki.tensor(eye,u,(id_dim**3,id_dim))
                


        print("(n X id ) (id X u )    : ")
        
        print( np.linalg.norm( np.dot(N2,U1) -np.eye(id_dim) ))        



        print("(n X id ) (id X u ) relation 2   : ")
        
        print( np.linalg.norm( np.dot(N1,U2) -np.eye(id_dim) ))  
        
        print("testing R3 : ")
        
        print( np.linalg.norm( np.dot(np.dot(R1,R2),R1)-np.dot(np.dot(R2,R1),R2)))
      
        
        
        print( "testing idXn Rxid =nXid id X R " )
        
        print( np.linalg.norm( np.dot(N2,R1)-np.dot(N1,R2) ))

        


        
        print("testing the relation: R times R_inv =id  R2 move")
        
        print( np.linalg.norm( np.dot(R,R_inv) -np.eye(args.iden_dimension*args.iden_dimension) ))

        
        print("testing the relation: R_inv times R =id  R2 move")
        
        print( np.linalg.norm( np.dot(R_inv,R) -np.eye(args.iden_dimension*args.iden_dimension) ))





        print("testing the relation:  R1 move using n and R")
        
        print( np.linalg.norm( np.dot(n,R) -n ))


        
        print("R is : ")
        print(R)

        print("R inv is : ")
        
        print(R_inv)

        print("n is : ")
        
        print(n)
        print("u is : ")
        
        print(u)
        
        
