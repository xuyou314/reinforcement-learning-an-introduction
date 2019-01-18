#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ACTION_BACK = 0
ACTION_END = 1

# behavior policy
def behavior_policy():
    return np.random.binomial(1, 0.5)

# target policy
def target_policy():
    return ACTION_BACK

# one turn
def play():
    # track the action for importance ratio
    trajectory = []
    while True:
        action = behavior_policy()
        trajectory.append(action)
        if action == ACTION_END:
            return 0, trajectory
        if np.random.binomial(1, 0.9) == 0:
            return 1, trajectory

def figure_5_4():
    runs = 10
    episodes = 1000000
    for run in range(runs):
        rewards = []
        for episode in range(0, episodes):
            reward, trajectory = play()
            if trajectory[-1] == ACTION_END:
                rho = 0
            else:
                rho = 1.0 / pow(0.5, len(trajectory))
            rewards.append(rho * reward)
        rewards = np.add.accumulate(rewards)
        estimations = np.asarray(rewards) / np.arange(1, episodes + 1)
        plt.plot(estimations)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Ordinary Importance Sampling')
    plt.xscale('log')

    plt.savefig('../images/figure_5_4.png')
    plt.close()
def test_every_visit():
    runs = 1000
    episodes = 50000
    values=[]
    for run in range(runs):
        reward_sum =0
        ct=0
        for episode in range(0, episodes):
            reward, trajectory = play()
            ct+=len(trajectory)
            if trajectory[-1]==0:
                reward_sum=reward_sum+np.power(2,len(trajectory)+1)-2
        values.append(reward_sum/ct)
        print(np.mean(values))
def test_first_visit():
    runs,episodes=1000,100000
    values=[]
    for run in range(runs):
        sum=0
        for episode in range(episodes):
            reward,trajectory=play()
            if trajectory[-1]==0:
                sum=sum+np.power(2,len(trajectory))
        values.append(sum/episodes)
        print(np.mean(values))
if __name__ == '__main__':
    #figure_5_4()
     #test_every_visit()
     test_first_visit()
