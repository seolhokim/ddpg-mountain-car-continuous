# Continuous control with deep reinforcement learning
- Implement DDPG ( Deep Deterministic Policy Gradient)

## Experiments

| Game | Epochs | Training Time | Model Parameters |
| :---: | :---: | :---: | :---: |
| MountainCarContinuous-v0 | 50 | 5 min | 7,486(total)

## Todo
  - solve the problem that if epochs are over 200, then the action is converged in wrong direction.
  - more games have to be tested.
  - parser
  
### Plot
#### MountainCarContinuous-v0
![img](https://github.com/seolhokim/ddpg_mountaincar_keras/blob/master/asset/mountaincar.png)
 - As epochs over 200, all(train and test) models are diverged.
  * i tried to adjust batch size, learning-rate, activation function, model size, noise size but it is not cleared.
