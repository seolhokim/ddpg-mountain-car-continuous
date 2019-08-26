from gameagent import Agent
import gym
from ounoise import OUNoise
import numpy as np


action_size = 1
exploration_mu = 0
exploration_theta = 0.15
exploration_sigma = 0.25
noise = OUNoise(action_size, exploration_mu, exploration_theta, exploration_sigma)

def _compute_discounted_R(R, discount_rate=.999):
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
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

import gym
env = gym.make("MountainCarContinuous-v0")

agent = Agent(2,1)

for iterate in range(5301,10000):
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
        action = np.clip((action +(noise.sample())), -1, 1)
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