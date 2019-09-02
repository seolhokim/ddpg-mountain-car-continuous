# Continuous control with deep reinforcement learning
- Implement DDPG ( Deep Deterministic Policy Gradient)

![3dball](https://github.com/seolhokim/ddpg_mountaincar_keras/blob/master/asset/3dball.PNG {: width="100" height="100"})

![mountaincarcontinuous](https://github.com/seolhokim/ddpg_mountaincar_keras/blob/master/asset/mountaincarcontinuous.PNG {: width="100" height="100"})

![pendulum](https://github.com/seolhokim/ddpg_mountaincar_keras/blob/master/asset/pendulum.PNG {: width="100" height="100"})

## Experiments

| Game | Epochs | Training Time | Model Parameters |
| :---: | :---: | :---: | :---: |
| MountainCarContinuous-v0 | 1000 | 30 min | 299,032(total)
| Pendulum-v0 | 1000 | 30 min | 299,536(total)
| 3DBall | willbeupdated | willbeupdated | willbeupdated

## Todo
  - solve the problem that if epochs are over 200, then the action is converged in wrong direction.
  - more games have to be tested.
  - parser

## Update (2019.08.27)
1. Save error and notation fixed
2. argparser added

## Update (2019.08.30)
1. replaybuffer.py's sampling method is changed.
2. new test result added.
3. pendulum-v0 is now testing.

### Plot
#### MountainCarContinuous-v0
##### 2019.08.27
![img](https://github.com/seolhokim/ddpg_mountaincar_keras/blob/master/asset/mountaincar.png)
 - As epochs over 200, all(train and test) models are diverged.
   * i tried to adjust batch size, learning-rate, activation function, model size, noise size but it is not cleared.
##### 2019.08.30
![img](https://github.com/seolhokim/ddpg_mountaincar_keras/blob/master/asset/mountaincar_08_30.PNG)
  - it doesn't converged at all.
    * i tried almost same model maded by another people, it looks same i guess ![here](https://github.com/piotrplata/keras-ddpg), but 
      it looks converged. but my model didn't converged.
##### 2019.08.30
![img](https://github.com/seolhokim/ddpg_mountaincar_keras/blob/master/asset/mountaincar_09_02.PNG)
  - i changed the training rate in Critic model at 0.001 to 0.0001(i have tried some points.)
     * it shows that model can be trained well by adjusting the learning rate. i gain the idea from TRPO and PPO that the change of model of parameters is handled carefully.
  
## Run

~~~
python main.py
~~~
- If you want to change hyper-parameters, you can check "python main.py --help"

Options:
- '--epochs', type=int, default=100, help='number of epochs, (default: 100)'
- '--e', type=str, default='MountainCarContinuous-v0', help='environment name, (default: MountainCarContinuous-v0)'
#- '--d', type=bool, default=False, help='train and test alternately. (default : False)'
- '--t', type=bool, default=True, help="True if training, False if test. (default: True)"
- '--r', type=bool, default=False, help='rendering the game environment. (default : False)'
- '--b', type=int, default=128, help='train batch size. (default : 128)'
- '--v', type=bool, default=False, help='verbose mode. (default : False)'
#- '--n', type=bool, default=True, help='reward normalization. (default : True)'
- '--sp', type=int, default=True, help='save point. epochs // sp. (default : 100)'

## Reference

## Version

