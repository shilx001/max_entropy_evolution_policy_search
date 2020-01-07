from ARS import *
import gym
# import matplotlib.pyplot as plt
import pickle
import os

seeds = [1, 2, 3, 4, 5]

for i in seeds:
    hp = HP(env=gym.make('HalfCheetah-v2'), noise=0.03, nb_steps=3000, episode_length=1000, num_deltas=32,
            num_best_deltas=4,
            learning_rate=0.03)
    trainer = ARSTrainer(hp=hp, monitor_dir=None, input_size=17, output_size=6)
    reward = trainer.train()
    pickle.dump(reward, open('halfcheetah_ars', 'wb'))
