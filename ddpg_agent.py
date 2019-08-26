from keras import layers
import tensorflow as tf
from keras import backend as K
from keras.models import Model
import numpy as np
from keras import optimizers
import copy

import gym

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
        #tf.gradients(self.model.output,self.model.input[1])[0]
        
    def __make_model(self):
        state_input_layer = layers.Input(shape=(self.input_dim,))
        action_input_layer = layers.Input(shape=(self.output_dim,))
        
        state_x = layers.BatchNormalization()(state_input_layer)
        action_x = layers.BatchNormalization()(action_input_layer)
        #action_x = layers.GaussianNoise(1.0)(action_x)#action noise랑 다르지만 
        
        state_x = layers.Dense(32,activation = 'relu')(state_x)
        action_x = layers.Dense(8,activation = 'relu')(action_x)
        
        x = layers.concatenate([state_x,action_x])
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
    
        #for layer in range(len(weights)):
        #    target_weights[layer] = self.tau * weights[layer] + (1 - self.tau) * target_weights[layer]
        target_weights = self.tau * weights + (1 - self.tau) * target_weights
        target.set_weights(target_weights)
        return target
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
        #x = layers.GaussianNoise(1.0)(x)
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
        
        '''
                optimizer = optimizers.Adam(lr = 0.0001)
        updates = optimizer.get_updates(params = self.model.trainable_weights,\
                                       loss =grads)
        self.update_function = K.function(inputs = [self.model.input,\
                                          self.q_gradient],\
                                         outputs = [],\
                                         updates = updates)
        '''

    def get_action(self,state):
        return self.model.predict(state)
    def train(self,state,grads):
        self.update_function([state,grads])
    
    def soft_update(self,target):
        weights = np.array(self.model.get_weights())
        target_weights = np.array(target.get_weights())
    
        #for layer in range(len(weights)):
        #    target_weights[layer] = self.tau * weights[layer] + (1 - self.tau) * target_weights[layer]
        target_weights = self.tau * weights + (1 - self.tau) * target_weights
        target.set_weights(target_weights)
        return target

import random
from collections import namedtuple, deque 
class ReplayBuffer():
    def __init__(self, maxlen=1208*4*10, batch_size=1208*4):
        self.memory = deque()
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",field_names=["state", "action", "reward", "next_state", "done"])
        
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        random.shuffle(self.memory)
        random.shuffle(self.memory)
        random.shuffle(self.memory)
        return [self.memory.popleft() for x in range(self.batch_size)]
    
    def __len__(self):
        return len(self.memory)
def _compute_discounted_R(R, discount_rate=.999):
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    #for t in reversed(range(len(R))):
    for t in reversed(range(len(R))):
        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add
    discounted_r -= discounted_r.mean() 
    discounted_r /= discounted_r.std()

    return discounted_r

def compute_discounted_R(record,discounted_rate = 0.999):
    reward_list = [x[2] for x in record]
    reward_list = _compute_discounted_R(reward_list)
    for i in range(len(record)):
        record[i][2] = reward_list[i]
    return record

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
class Agent:
    def __init__(self,input_dim,output_dim, tau = 0.001, gamma =0.99):
        '''
        TODO:
        -1.Critic, target Critic
        -2.Policy, target Policy
        3.Buffer(outside)
        -4.Put batchNormalization in the Network(at the first layer) 
        5.Put the noise into action placeholder
        6.loss is derivative of critic * derivative of policy
        
        Input:
        1. tau(for target network update)
        2. gamma(for reward)
        3. layer info(input,hidden, output)
        '''
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau = tau
        self.gamma = gamma
        
        self.main_critic = Critic(input_dim,output_dim,tau,gamma)
        self.target_critic = Critic(input_dim,output_dim,tau,gamma)
    
        self.main_actor = Actor(input_dim,output_dim,tau,gamma)
        self.target_actor = Actor(input_dim,output_dim,tau,gamma)
    
        self.target_critic.model.set_weights(self.main_critic.model.get_weights())
        self.target_actor.model.set_weights(self.main_actor.model.get_weights())
        
        self.memory = ReplayBuffer()

    def get_action(self,state):
        return self.main_actor.get_action(state)

    def train(self):
        while (len(self.memory)) > 1208*4:
            data = self.memory.sample()
            states = np.vstack([e.state for e in data if e is not None])
            actions = np.array([e.action for e in data if e is not None]).astype(np.float32).reshape(-1, self.output_dim)
            rewards = np.array([e.reward for e in data if e is not None]).astype(np.float32).reshape(-1, 1)
            dones = np.array([e.done for e in data if e is not None]).astype(np.uint8).reshape(-1, 1)
            next_states = np.vstack([e.next_state for e in data if e is not None])
            
            
            actions_next = self.target_actor.model.predict_on_batch(next_states)
            #actions_next = self.target_actor.predict_on_batch(next_states)
            Q_targets_next = self.target_critic.model.predict_on_batch([next_states, actions_next])

            Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)

            
            self.main_critic.train(states,actions,Q_targets)
            #self.main_critic.train(states,actions,rewards)
            action_gradients = np.reshape(self.main_critic.get_gradient(states,actions), \
                                         (-1, self.output_dim))

            self.main_actor.train(states,action_gradients)

            self.target_actor.model = self.main_actor.soft_update(self.target_actor.model)
            self.target_critic.model = self.main_critic.soft_update(self.target_critic.model)

action_size = 1
exploration_mu = 0
exploration_theta = 0.15
exploration_sigma = 0.25
noise = OUNoise(action_size, exploration_mu, exploration_theta, exploration_sigma)
env = gym.make("MountainCarContinuous-v0")

for iterate in range(10000):
    if iterate % 100 == 0:
        print('saved')
        agent.main_critic.model.save_weights("./well_trained_main_critic_"+str(iterate)+".h5")
        agent.main_actor.model.save_weights("./well_trained_main_actor_"+str(iterate)+".h5") 
    print('iterate : ',iterate)
    record = []
    done = False
    frame = env.reset()
    ep_reward = 0
    while done != True:
        #env.render()
        state = frame.reshape(1,-1)
        state = (state - env.observation_space.low) / \
                (env.observation_space.high - env.observation_space.low)
        
        action = agent.get_action(state)
        action = np.clip((action +(noise.sample()* (1 - 0.0001 * iterate))), -1, 1)
        next_frame, reward, done, _ = env.step(action)
        
        
        if reward < 99:
            reward_t = -1.
        else:
            reward_t = 100.
        record.append([state,action,reward_t,next_frame.reshape(1,-1),done])
        ep_reward += reward_t
        frame = next_frame
        #print('state : ', state, ', action :', action, ', reward_t : ',reward_t,', reward : ', reward,', done : ',done,\
        #     ', ep_reward : ',ep_reward)
        if ep_reward < - 999:
            done = True
        if done :
            #print("asd")
            #record = compute_discounted_R(record)
            
            list(map(lambda x : agent.memory.add(x[0],x[1],x[2],x[3],x[4]), record))
            #break
        if len(agent.memory)> 1208*8*5:
            print('trained_start')
            agent.train()
            print('trained_well')
    else:
        print("ep_reward:", ep_reward)
        
        continue
    break

env.close()