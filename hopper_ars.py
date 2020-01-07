from ARS import *
import gym
# import matplotlib.pyplot as plt
import pickle
import os

seeds = [1, 2, 3, 4, 5]

for i in seeds:
    hp = HP(env=gym.make('Hopper-v2'), noise=0.025, nb_steps=2000, episode_length=1000, num_deltas=8, num_best_deltas=4,
            learning_rate=0.01,seed=i)
    trainer = ARSTrainer(hp=hp, monitor_dir=None, input_size=11, output_size=3)
    reward = trainer.train()
    pickle.dump(reward, open('hopper_ars', 'wb'))
