# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:01:54 2020

@author: Mustafa Hajij
"""

import Temperley_Lieb_algebra_rep_network as tlnet 
import braid_group_rep_network as bgnet
import symmetric_group_rep_network as symgnet
import ZxZ_group_rep_network as ZSnet
import utilities as ut

import math
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow  import keras
import argparse

import numpy as np
import os.path
import os

def model_string_gen(model_name):
    
    name=model_name+str(args.bias)+"_activation="+  str(args.network_generator_activation)+"_"+str(args.generator_dimension)+"_to_"+str(args.generator_dimension)+"_delta="+str(args.delta)+"_.h5"
    
    return name
    
weight_folder='weights/'   
 
data_folder='data/'    

    

if __name__ == '__main__':

    
    
    parser = argparse.ArgumentParser()

    # testing or training argument

    parser.add_argument('-m', '--mode',type=str,default='training',required=True,help='Specify if you want to train a network or testing it.')   

    # type of structure, options : TL, braid_group, symmetric_group, ZxZ

    parser.add_argument('-st', '--structure',type=str,default="TL",required=True,help='Which structure you want to train. Options are : "TL" , "braid_group", "ZxZ_group" and "symmetric_group". Any other option will generate an error.')
    
    
    #type, dim of network arguments : linear, affine, nonlinear. Activation determines the type.
    
    #(1) dim of the rep
    parser.add_argument('-dim', '--generator_dimension',type=int,default=2,help='domain dimension of the gen network.')

    #(2) determines the type of the of the rep
  
    parser.add_argument('-a', '--network_generator_activation',type=str,default='linear',required=False, help='Activation of the type. Options are : linear, tanh. Uniformly chosen for all layers when linear activation are chosen and sigmoid otherwise..')
   
    #(3) determines the type of the of the rep
    
    parser.add_argument('-bias', '--bias',type=str,default=False,required=False, help='Use bias in your network.')
   
    # TL algebra argument : delta.
    #* determines delta for the TL algebra. Not effective when training a group.
    
    
    parser.add_argument('-delta', '--delta',type=float,required=False,default=1,help='delta, the TL parameter.')

    #__________________________________________________________

    # training arguments
    #(1) number of epochs
    
    parser.add_argument('-e', '--epoch',type=int,default=2,help='Number of epochs.')
    
    #(2) learning rate
    
    parser.add_argument('-lr', '--learning_rate',type=float,required=False,default=0.002,help='learning rate.')

    #(3) batchsize

    parser.add_argument('-b', '--batch_size',type=float,required=False,default=2000,help='batch size')

    #___________________________________________________________




    args = parser.parse_args()
    
    if args.generator_dimension in [2,4,6]:
        
        d=args.generator_dimension//2

        data1=np.load(data_folder+str(d)+'d_data_1.npy')
        data2=np.load(data_folder+str(d)+'d_data_2.npy')
        data3=np.load(data_folder+str(d)+'d_data_3.npy')
         
    else:
    
        raise ValueError("dimensions are constrained to 2 , 4 or 6")

    
    if args.mode=='training':
        
        print("choosing the training mode. ")
        
        
        
        if args.structure=='TL':
                
                
            if args.delta==0:
                
                raise ValueError("delta, a parameter for TL algebra, must be nonzero.")


            print("training the TL algebra generator.")
          
            print("generator function : R^"+str(args.generator_dimension) +"-> R^"+str(args.generator_dimension) ) 
            
            dim=args.generator_dimension
            
            
            Ugen=tlnet.U_generator(dim=args.generator_dimension,bias= args.bias, activation_function=args.network_generator_activation)
            

            M=tlnet.TL_algebra_net(Ugen,delta =args.delta  ,input_dim=dim//2)
            
            
            model_name=model_string_gen("TL_algebra_relations_trainer_use_bias=")
            
  
            
            checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
    
            
            ut.train_net(M,[data1,data2,data3],data1,tlnet.TL_loss_wrapper(dim//2),callbacks_list,args.learning_rate,args.batch_size,args.epoch)


            print("saving the model.." )

            model_name_U_gen=model_string_gen("TL_algebra_generator_use_bias=")

            
            Ugen.save(model_name_U_gen) 


            print("model saved.")
            
        elif args.structure=='braid_group': 
            
                          
            
            print("training the braid group generators")


            dim=args.generator_dimension
            
                        
            R_oP1=bgnet.R_operator(dim=args.generator_dimension,activation_function=args.network_generator_activation,bias=args.bias )
            
            R_oP2=bgnet.R_operator(dim=args.generator_dimension,activation_function=args.network_generator_activation,bias=args.bias)
            
            
            M=bgnet.braid_group_rep_net(R_oP1,R_oP2,input_shape=dim//2)

            model_name1=model_string_gen("braid_group_sigma_generator_use_bias=")
            model_name2=model_string_gen("braid_group_sigma_inverse_generator_use_bias=")
            model_name3=model_string_gen("braid_group_relations_trainer_use_bias=")

            """
            if os.path.exists(model_name3):
                print("---------------------******---------------------------")
                print("model weights found, loading weights from harddrive")
                print("---------------------******---------------------------")
            
                M.load_weights(weight_folder+model_name3)
            """   
            
            checkpoint = ModelCheckpoint(model_name3, monitor='loss', verbose=1, save_best_only=True, mode='min')
            
            callbacks_list = [checkpoint]
    
             
            
            
            data_out=np.hstack([data1,data2])

            ut.train_net(M,[data1,data2,data3],data_out,bgnet.braid_group_rep_loss(dim//2),callbacks_list,args.learning_rate,args.batch_size,args.epoch)
            
            

            
            #M.save()
            print("saving the models.." )

            R_oP1.save(model_name1) 
            R_oP2.save(model_name2) 
 
            print("model saved.")


        elif args.structure=='ZxZ_group': 
            
            if args.generator_dimension in list(range(2,11)):
                
                
                data1=np.load(data_folder+str(args.generator_dimension)+'d_data.npy')
            
            else:        
                
                raise ValueError("dimensions are constrained between 2 and 10")




            dim=args.generator_dimension
            
                        
            A_oP=ZSnet.a_operator(args.generator_dimension,activation_function=args.network_generator_activation,bias=args.bias )
            
            B_oP=ZSnet.a_operator(args.generator_dimension,activation_function=args.network_generator_activation,bias=args.bias)
            
            
            M=ZSnet.ZxZ_group_rep_net(A_oP,B_oP,input_shape=dim)


            model_name=model_string_gen("ZxZ_group_relations_trainer_use_bias=")
            aname=model_string_gen("ZxZ_group_a_generator_use_bias=")
            bname=model_string_gen("ZxZ_group_b_generator_use_bias=")



            
            checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
            
            callbacks_list = [checkpoint]
    
             
            
            ut.train_net(M,data1,data1,ZSnet.ZxZ_group_rep_loss(dim),callbacks_list,args.learning_rate,args.batch_size,args.epoch)

            
            
            

            print("saving the models.." )

            A_oP.save(aname) 
            
            B_oP.save(bname) 
 
            print("model saved.")


            
        elif args.structure=='symmetric_group': 
                          
            print("training the symmetric_group generator")        

            dim=args.generator_dimension
         
            R_oP1=symgnet.sigma_operator(args.generator_dimension,activation_function=args.network_generator_activation ,bias=args.bias)
            
            
            
            
            M=symgnet.symmetric_group_rep_net(R_oP1,input_shape=dim//2)

            model_name=model_string_gen("symmetric_group_relations_trainer_use_bias=")
            model_name_generator=model_string_gen("symmetric_group_generator_use_bias=")

           
            checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
            
            callbacks_list = [checkpoint]
               
            
            
            data_out=np.hstack([data1,data2])
            
            ut.train_net(M,[data1,data2,data3], data_out, symgnet.symmetric_group_rep_loss(dim//2),callbacks_list,args.learning_rate,args.batch_size,args.epoch)
            
 
            print("saving the model..")
            
            
            R_oP1.save(model_name_generator) 
            
                        
            print("model saved.")
            
        else:
            
            raise ValueError("structures must be : TL, braid_group, symmetric_group, or ZxZ_group")

       

    
    elif args.mode=='testing':
        print("testing the relations..")
        if args.structure=='TL':
                
                
            if args.delta==0:
                
                raise ValueError("delta must be nonzero.")


          
            print("generator function : R^"+str(args.generator_dimension) +"-> R^"+str(args.generator_dimension) ) 
            
            dim=args.generator_dimension

            Ugen=tlnet.U_generator(dim=args.generator_dimension,bias=args.bias,activation_function=args.network_generator_activation)
            
            M=tlnet.TL_algebra_net(Ugen,delta =args.delta  ,input_dim=dim//2)

            model_name=model_string_gen("TL_algebra_relations_trainer_use_bias=")

            
            M.load_weights(weight_folder+model_name)
            
            d=dim//2
            
            relations_tensor=M.predict([data1[:500],data2[:500],data3[:500]])  
            relations_tensor=relations_tensor[:100]
            
            print("testing the relation:  U_i*U_{i-1}*U_i=U_i")
            
            print(np.linalg.norm(relations_tensor[:,0:3*d]-relations_tensor[:,3*d:6*d]))
            
            print("testing the relation:  U_i*U_{i+1}*U_i=U_i")
            
            print(np.linalg.norm(relations_tensor[:,6*d:9*d]-relations_tensor[:,9*d:12*d]))

            print("testing the relation: U_{i}^2=\delta*U_{i} ")
            
            print(np.linalg.norm(relations_tensor[:,12*d:14*d]-relations_tensor[:,14*d:]))
        

        elif args.structure=='braid_group': 
          
            print("testing the braid group generators")


            dim=args.generator_dimension
            
                        
            R_oP1=bgnet.R_operator(dim=args.generator_dimension,activation_function=args.network_generator_activation, bias=args.bias)
            
            R_oP2=bgnet.R_operator(dim=args.generator_dimension,activation_function=args.network_generator_activation,bias=args.bias)
            
            
            M=bgnet.braid_group_rep_net(R_oP1,R_oP2,input_shape=dim//2)
            
            
            data1=data1[:100]
            data2=data2[:100]
            data3=data3[:100]
            
            
            data_out=np.hstack([data1,data2])

            print("loading the models.." )

            model_name1=model_string_gen("braid_group_sigma_generator_use_bias=")
            model_name2=model_string_gen("braid_group_sigma_inverse_generator_use_bias=")
            model_name3=model_string_gen("braid_group_relations_trainer_use_bias=")



 

            M.load_weights(weight_folder+model_name3)
            
            d=dim//2
 
            relations_tensor=M.predict([data1,data2,data3])
            relations_tensor=relations_tensor[:100]


            print("testing the relation:  R3 ")

            print(np.linalg.norm(relations_tensor[:,0:3*d]-relations_tensor[:,3*d:6*d]))
            
            print("testing the relation:  R2 ")
            
            print(np.linalg.norm(relations_tensor[:,6*d:8*d]-data_out))

            print("testing the relation:  R2 ")
            
            print(np.linalg.norm(relations_tensor[:,8*d:]-data_out))
            
        elif args.structure=='symmetric_group': 
                          

            dim=args.generator_dimension

            
            R_oP1=symgnet.sigma_operator(args.generator_dimension,activation_function=args.network_generator_activation,bias=args.bias)
            
            
            
            
            M=symgnet.symmetric_group_rep_net(R_oP1,input_shape=dim//2)

            model_name=model_string_gen("symmetric_group_relations_trainer_use_bias=")
            model_name_generator=model_string_gen("symmetric_group_generator_use_bias=")

             
            R_oP1.load_weights(weight_folder+model_name_generator)
            M.load_weights(weight_folder+model_name)            
            
            
            data_out=np.hstack([data1,data2])
            
            
            d=dim//2


            relations_tensor=M.predict([data1,data2,data3])
            
            relations_tensor=relations_tensor[:100]

            print("testing the relation:  R3 ")

            print(np.linalg.norm(relations_tensor[:,0:3*d]-relations_tensor[:,3*d:6*d]))
            
            print("testing the relation:  R2 ")
            
            print(np.linalg.norm(relations_tensor[:,6*d:]-data_out))



        elif args.structure=='ZxZ_group': 

            if args.generator_dimension in list(range(2,11)):
                
                
                data1=np.load(data_folder+args.generator_dimension+'d_data.npy')

            else:
                
                raise ValueError("dimensions are constrained between 2 and 10")
                          

            dim=args.generator_dimension

            A_oP=ZSnet.a_operator(args.generator_dimension,activation_function=args.network_generator_activation,  bias=args.bias)
            
            B_oP=ZSnet.a_operator(args.generator_dimension, activation_function=args.network_generator_activation, bias=args.bias)
            
            
            M=ZSnet.ZxZ_group_rep_net(A_oP,B_oP,input_shape=dim)


            model_name=model_string_gen("ZxZ_group_relations_trainer_use_bias=")
            aname=model_string_gen("ZxZ_group_a_generator_use_bias=")
            bname=model_string_gen("ZxZ_group_b_generator_use_bias=")
 

           
            A_oP.load_weights(weight_folder+aname) 
            B_oP.load_weights(weight_folder+bname) 
 
            M.load_weights(weight_folder+model_name)

           
            


            relations_tensor=M.predict(data1)
            
            print("testing the relation:  ")

            print( np.linalg.norm( relations_tensor[:,:dim][:100]-relations_tensor[:,dim:][:100]))
            
        else:
            
            raise ValueError("structures must be : TL, braid_group, symmetric_group, or ZxZ_group")




    else:
        raise ValueError("Mode options are either training or testing.")
        
 
    