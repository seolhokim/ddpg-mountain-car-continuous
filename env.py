from gameagent import Agent
import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse



epsilon = 0.999
epsilon_decaying = 0.99995
def run_episode(train = True, render = False, train_batch_size = 640,verbose = False):
    global epsilon
    global epsilon_decaying
    epsilon *= epsilon_decaying
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
        if np.random.random() < epsilon:
            action = np.clip(agent.get_action(state) + (np.random.normal()*epsilon),-1,1)
        else:
            action = agent.get_action(state)
        next_frame, reward, done, _ = env.step(action)
        if reward <100 :
            reward = -1.
        else :
            reward = 100.
        agent.memory.add(state,action,reward,next_frame.reshape(1,-1),done)
        ep_reward += reward
        frame = next_frame
        if verbose :
            print('state : ', state, ', action :', action, ', reward : ',reward,', reward : ', reward,', done : ',done,\
                ', ep_reward : ',ep_reward)
    if train:
        print('trained_start')
        agent.train()
        print('trained_well')
    print("ep_reward:", ep_reward)

    episode_reward_lst.append(ep_reward)


def run_training(iteration,render = False, train_batch_size = 640,verbose = False, save_point = 100):
    for iterate in range(1,iteration+1):
        print('iterate : ',iterate)
        if iterate % 5 == 0:
            run_episode(train = True, render = False, train_batch_size=640,verbose=False)
        else:
            run_episode(train = False, render = False, train_batch_size=640,verbose=False)
        if iterate % save_point == 0:
            agent.main_critic.model.save_weights("./well_trained_main_critic_"+str(iterate)+".h5")
            agent.target_critic.model.save_weights("./well_trained_target_critic_"+str(iterate)+".h5")
            agent.main_actor.model.save_weights("./well_trained_main_actor_"+str(iterate)+".h5") 
            agent.target_actor.model.save_weights("./well_trained_target_actor_"+str(iterate)+".h5")
            
def run_test(render = False,verbose = False):
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
        next_frame, reward, done, _ = env.step(action)
        if reward <100 :
            reward = -1.
        else :
            reward = 100.
        ep_reward += reward
        frame = next_frame
        if verbose :
            print('state : ', state, ', action :', action, ', reward : ',reward,', reward : ', reward,', done : ',done,\
                ', ep_reward : ',ep_reward)
    
def main():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs, (default: 100)')
    parser.add_argument('--e', type=str, default='MountainCarContinuous-v0', help='environment name, (default: MountainCarContinuous-v0)')
    #parser.add_argument('--d', type=bool, default=False, help='train and test alternately. (default : False)')
    parser.add_argument('--t', type=bool, default=True, help="True if training, False if test. (default: True)")
    parser.add_argument('--r', type=bool, default=False, help='rendering the game environment. (default : False)')
    parser.add_argument('--b', type=int, default=128, help='train batch size. (default : 128)')
    parser.add_argument('--v', type=bool, default=False, help='verbose mode. (default : False)')
    #parser.add_argument('--n', type=bool, default=True, help='reward normalization. (default : True)')
    parser.add_argument('--sp', type=int, default=True, help='save point. epochs // sp. (default : 100)')

    args = parser.parse_args()
    configuration(args.e,args.b,args.epochs,args.t,args.r,args.v,args.sp)
    
def configuration(environment,batch_size,epochs,train,render,verbose,save_point):
    global agent
    global env
    global episode_reward_lst
    global test_episode_reward_lst
    env = gym.make(environment)
    episode_reward_lst = []
    agent = Agent(env.observation_space.shape[0],env.action_space.shape[0],train_batch_size = batch_size)
    
    agent.main_actor.model.summary()
    agent.main_critic.model.summary()
    if train : 
        run_training(epochs,render = render, train_batch_size=batch_size,\
                verbose = verbose,save_point = save_point)
    else :
        run_test(render = render , verbose = verbose)
if __name__ == '__main__':
    main()
