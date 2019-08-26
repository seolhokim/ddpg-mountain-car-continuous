from keras import layers
import tensorflow as tf
from keras import backend as K
from keras.models import Model
import numpy as np
from keras import optimizers

class Actor:
    def __init__(self,input_dim,output_dim, tau,gamma):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau = tau
        self.gamma = gamma
        self.model = self.__make_model()

        self.__make_loss_function()
                
    def __make_model(self):
        input_layer = layers.Input(shape=(self.input_dim,))
        x = layers.BatchNormalization()(input_layer)
        x = layers.Dense(32,activation = 'relu')(x)
        x = layers.Dense(32,activation = 'relu')(x)
        x = layers.Dense(32,activation = 'relu')(x)
        x = layers.Dense(self.output_dim,activation = 'tanh')(x)
        
        model = Model(inputs = input_layer, outputs = x)
        return model
    
    def __make_loss_function(self):   
        q_gradient = K.placeholder(shape = (None,self.output_dim),\
                                  name = 'q_gradient_placeholder')
        
        loss =  tf.gradients(self.model.output,self.model.trainable_weights, -q_gradient) #grad_ys임 값 고정위해
        
        #loss = -K.mean(a_gradient_when_q)
        
        optimizer = optimizers.Adam(lr=0.0001)
        updates = optimizer.get_updates(loss = loss, params = self.model.trainable_weights)
        
        self.update_function = K.function(inputs = [self.model.input,q_gradient],\
                                          outputs = [],\
                                          updates = updates)


    def get_action(self,state):
        return self.model.predict(state)
    def train(self,state,grads):
        self.update_function([state,grads])
    
    def soft_update(self,target):
        weights = np.array(self.model.get_weights())
        target_weights = np.array(target.get_weights())
        target_weights = self.tau * weights + (1 - self.tau) * target_weights
        target.set_weights(target_weights)
        return target