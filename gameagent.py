import numpy as np
from critic import Critic
from actor import Actor
from replaybuffer import ReplayBuffer

class Agent:
    def __init__(self,input_dim,output_dim, tau = 0.001, gamma =0.99,train_batch_size = 512):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau = tau
        self.gamma = gamma
        self.train_batch_size = train_batch_size
        self.main_critic = Critic(input_dim,output_dim,tau,gamma)
        self.target_critic = Critic(input_dim,output_dim,tau,gamma)
    
        self.main_actor = Actor(input_dim,output_dim,tau,gamma)
        self.target_actor = Actor(input_dim,output_dim,tau,gamma)
    
        self.target_critic.model.set_weights(self.main_critic.model.get_weights())
        self.target_actor.model.set_weights(self.main_actor.model.get_weights())
        
        self.memory = ReplayBuffer(batch_size = train_batch_size)

    def get_action(self,state):
        return self.main_actor.get_action(state)

    def train(self):
        while (len(self.memory)) > self.train_batch_size:
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
            action_gradients = np.reshape(self.main_critic.get_gradient(states,actions), \
                                         (-1, self.output_dim))

            self.main_actor.train(states,action_gradients)

            self.target_actor.model = self.main_actor.soft_update(self.target_actor.model)
            self.target_critic.model = self.main_critic.soft_update(self.target_critic.model)