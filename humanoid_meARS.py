from max_ent_ARS import *
import pickle
import gym

env = 'Humanoid-v2'
seeds = [1, 2, 3, 4, 5]
observation_dim = gym.make(env).observation_space.shape[0]
action_dim = gym.make(env).action_space.shape[0]

for i in range(5):
    hp = HP(env=gym.make(env), noise=0.075, nb_steps=1000, episode_length=1000, num_deltas=230,
            num_best_deltas=230, learning_rate=0.02, seed=seeds[i],
            n_policy_sampling=1, policy_variance=0.02, alpha=0.1)
    trainer = ARSTrainer(hp=hp, monitor_dir=None, input_size=observation_dim, output_size=action_dim)
    reward = trainer.train()
    pickle.dump(reward, open(env + '_mears_+seeds_' + str(seeds[i]), 'wb'))
