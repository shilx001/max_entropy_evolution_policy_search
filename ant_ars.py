from ARS import *
import gym
# import matplotlib.pyplot as plt
import pickle
import os

seeds = [1, 2, 3, 4, 5]

for i in seeds:
    hp = HP(env=gym.make('Ant-v2'), noise=0.015, nb_steps=1000, episode_length=1000, num_deltas=60, num_best_deltas=20,
            learning_rate=0.01)
    trainer = ARSTrainer(hp=hp, monitor_dir=None, input_size=111, output_size=8)
    reward = trainer.train()
    pickle.dump(reward, open('ant_ars', 'wb'))
