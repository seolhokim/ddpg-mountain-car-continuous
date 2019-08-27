from gameagent import Agent
import gym
from ounoise import OUNoise
import numpy as np
import matplotlib.pyplot as plt

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

def run_process(iteration, double_mode = False, train = True, render = False, train_batch_size = 128,verbose = False,reward_normalization = False):
    save_point = 1
    if train : 
        save_point = iteration // 10
    for iterate in range(iteration):
        if train & (iterate % save_point == 0):
            print('saved')
            agent.main_critic.model.save_weights("./well_trained_main_critic_"+str(iterate)+".h5")
            agent.main_actor.model.save_weights("./well_trained_main_actor_"+str(iterate)+".h5") 
        print('iterate : ',iterate)
        if double_mode :
            run_episode(train,render, train_batch_size, verbose, reward_normalization)
            run_episode(False,render, train_batch_size, verbose, reward_normalization)
        else:
            run_episode(train,render,train_batch_size,verbose,reward_normalization)

def run_episode(train = True, render = False, train_batch_size = 128,verbose = False,reward_normalization = False):
    record = []
    done = False
    frame = env.reset()
    ep_reward = 0
    while done != True:
        if render:
            env.render()
        state = frame.reshape(1,-1)
        state = (state - env.observation_space.low) / \
                (env.observation_space.high - env.observation_space.low)

        action = agent.get_action(state)
        if train : 
            action = np.clip((action +(noise.sample())), -1, 1)
        else :
            action = np.clip(action, -1,1)
        next_frame, reward, done, _ = env.step(action)
        record.append([state,action,reward,next_frame.reshape(1,-1),done])
        ep_reward += reward
        frame = next_frame
        if verbose :
            print('state : ', state, ', action :', action, ', reward : ',reward,', reward : ', reward,', done : ',done,\
                ', ep_reward : ',ep_reward)
        if done & train :
            if reward_normalization : 
                record = compute_discounted_R(record)
            list(map(lambda x : agent.memory.add(x[0],x[1],x[2],x[3],x[4]), record))
        if (len(agent.memory)>10000)& train:
            print('trained_start')
            agent.train()
            print('trained_well')
    print("ep_reward:", ep_reward)
    if train:
        episode_reward_lst.append(ep_reward)
    else:
        test_episode_reward_lst.append(ep_reward)

def main():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs, (default: 100)')
    parser.add_argument('--e', type=str, default='MountainCarContinuous-v0', help='environment name, (default: MountainCarContinuous-v0)')
    parser.add_argument('--d', type=bool, default=False, help='train and test alternately. (default : False)')
    parser.add_argument('--t', type=bool, default=True, help="True if training, False if test. (default: True)")
    parser.add_argument('--r', type=bool, default=False, help='rendering the game environment. (default : False)')
    parser.add_argument('--b', type=int, default=128, help='train batch size. (default : 128)')
    parser.add_argument('--v', type=bool, default=False, help='verbose mode. (default : False)')
    parser.add_argument('--n', type=bool, default=True, help='reward normalization. (default : True)')
    
    args = parser.parse_args()
    configuration(args.e,args.b,args.epochs,args.d,args.t,args.r,args.v.args.n)
    
def configuration(environment,batch_size,epochs,double_mode,train,render,verbose,reward_normalization):
    env = gym.make(environment)
    episode_reward_lst = []
    test_episode_reward_lst = []
    agent = Agent(env.observation_space.shape[0],env.action_space.shape[0],train_batch_size = batch_size)
    agent.main_actor.model.summary()
    agent.main_critic.model.summary()

    run_process(epochs,double_mode=double_mode, train=train,render = render, train_batch_size=batch_size,\
                verbose = verbose,reward_normalization=reward_normalization)

def __name__== '__main__':
    main()
