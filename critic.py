from keras import layers
import tensorflow as tf
from keras import backend as K
from keras.models import Model
import numpy as np
from keras import optimizers

class Critic:
    def __init__(self,input_dim,output_dim,tau,gamma):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau = tau
        self.gamma = gamma
        
        self.model = self.__make_model()
        self.__build_loss_function()
        self.q_gradient = K.function(inputs = self.model.input,\
                                    outputs = tf.gradients(self.model.output,self.model.input[1]))
        
    def __make_model(self):
        state_input_layer = layers.Input(shape=(self.input_dim,))
        action_input_layer = layers.Input(shape=(self.output_dim,))
        
        state_x = layers.BatchNormalization()(state_input_layer)
        action_x = layers.BatchNormalization()(action_input_layer)
        
        state_x = layers.Dense(32,activation = 'relu')(state_x)
        action_x = layers.Dense(8,activation = 'relu')(action_x)
        
        x = layers.Concatenate()([state_x,action_x])
        x = layers.Dense(32,activation = 'relu')(x)
        x = layers.Dense(1,activation = 'linear')(x)
        
        model = Model(inputs = [state_input_layer,action_input_layer], outputs = x)
        return model        
    
    def __build_loss_function(self):
        critic_output = self.model.output
        reward_placeholder = K.placeholder(shape = (None,self.output_dim),\
                                          name = 'reward')
        
        critic_loss = K.mean(K.square(reward_placeholder - critic_output)) 
        
        critic_optimizer = optimizers.Adam(lr = 0.001)
        critic_updates = critic_optimizer.get_updates(params = self.model.trainable_weights,\
                                                     loss = critic_loss)
        self.update_function = K.function(inputs = [self.model.input[0],\
                                                    self.model.input[1],\
                                                     reward_placeholder],\
                                           outputs = [], updates = critic_updates)
    def train(self,state,action,reward):
        self.update_function([state,action,reward])
              
    def get_gradient(self,state,action):
        return self.q_gradient([state,action])
    def soft_update(self,target):
        weights = np.array(self.model.get_weights())
        target_weights = np.array(target.get_weights())
    
        target_weights = self.tau * weights + (1 - self.tau) * target_weights
        target.set_weights(target_weights)
        return target