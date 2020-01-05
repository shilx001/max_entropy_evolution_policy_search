import os
import numpy as np
import gym
from gym import wrappers
import pickle
import datetime


class HP:
    # Hyperparameters
    def __init__(self,
                 nb_steps=1000,
                 episode_length=1000,
                 learning_rate=0.02,
                 num_deltas=4,
                 num_best_deltas=2,
                 policy_variance=0.01,
                 n_policy_sampling=8,
                 noise=0.03,
                 seed=1,
                 env=None,
                 record_every=50,
                 alpha=0.1):
        self.nb_steps = nb_steps  # rollout number
        self.episode_length = episode_length  # 每个episode的长度
        self.learning_rate = learning_rate
        self.num_deltas = num_deltas  # delta的个数
        self.num_best_deltas = num_best_deltas  # 最优delta的个数
        self.policy_variance = policy_variance
        self.n_policy_sampling = n_policy_sampling
        assert self.num_best_deltas <= self.num_deltas
        self.noise = noise  # 噪音的range
        self.seed = seed  # random seed
        self.env = env
        self.alpha = alpha
        self.record_every = record_every


class Normalizer:
    # Normalizes the input observations
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):  # observe a space, dynamic implementation of calculate mean of variance
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)  # 方差，清除了小于1e-2的

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


class Policy:
    def __init__(self, input_size, output_size, hp):
        self.theta = np.zeros((output_size, input_size))
        self.hp = hp
        self.sigma = self.hp.policy_variance

    def evaluate(self, input, delta=None, direction=None):
        # input: 输入state
        if direction is None:  # 如果没标明方向则返回theta*input(矩阵乘)
            return self.theta.dot(input)
        elif direction == "+":
            return (self.theta + self.hp.noise * delta).dot(input)  # 如果是+方向则返回
        elif direction == "-":
            return (self.theta - self.hp.noise * delta).dot(input)

    def evaluate_random_policy(self, input_state, epsilon, random_number, direction=None):
        # 根据当前的epsilon计算均值。
        if direction is None:
            pass
        elif direction == '+':
            return (self.theta + epsilon * self.hp.noise + random_number * self.hp.policy_variance).dot(
                input_state)
        elif direction == '-':
            return (self.theta - epsilon * self.hp.noise + random_number * self.hp.policy_variance).dot(
                input_state)

    def evaluate_policy(self, input_state):
        return self.theta.dot(input_state)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.hp.num_deltas)]  # 针对每个采样正态分布采样

    def update(self, rollouts, sigma_rewards):  # 针对rollout中的值对policy进行update
        # sigma_rewards is the standard deviation of the rewards
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, delta in rollouts:
            step += (r_pos - r_neg) * delta
        self.theta += self.hp.learning_rate / (self.hp.num_best_deltas * sigma_rewards) * step


class ARSTrainer:
    def __init__(self,
                 hp=None,
                 input_size=None,
                 output_size=None,
                 normalizer=None,
                 policy=None,
                 monitor_dir=None):

        self.hp = hp or HP()
        np.random.seed(self.hp.seed)
        self.env = self.hp.env
        self.env.seed(self.hp.seed)
        if monitor_dir is not None:
            should_record = lambda i: self.record_video
            self.env = wrappers.Monitor(self.env, monitor_dir, video_callable=should_record, force=True)
        self.hp.episode_length = self.hp.episode_length
        self.input_size = input_size or self.env.observation_space.shape[0]
        self.output_size = output_size or self.env.action_space.shape[0]
        self.normalizer = normalizer or Normalizer(self.input_size)
        self.policy = policy or Policy(self.input_size, self.output_size, self.hp)
        self.record_video = False

    def explore_policy(self, epsilon, direction=None, with_noise=True):
        done = False
        num_plays = 0
        sum_rewards = 0
        for i in range(self.hp.n_policy_sampling):
            random_number = np.random.randn(*self.policy.theta.shape)
            obs = self.env.reset()
            while not done and num_plays < self.hp.episode_length:
                self.normalizer.observe(obs)
                entropy = np.log(
                    self.hp.policy_variance ** 2 * np.sum(np.square(self.normalizer.normalize(obs))) + 1e-20)
                obs = self.normalizer.normalize(obs)
                action = self.policy.evaluate_random_policy(obs, epsilon, random_number, direction=direction)
                obs, reward, done, _ = self.env.step(action)
                if with_noise:
                    reward += np.random.randn()
                num_plays += 1
                sum_rewards += reward + self.hp.alpha * entropy
        return sum_rewards / self.hp.n_policy_sampling

    def evaluate_policy(self):
        # 根据policy返回一个值
        obs = self.env.reset()
        done = False
        num_plays = 0
        sum_rewards = 0
        while not done and num_plays < self.hp.episode_length:
            obs = self.normalizer.normalize(obs)
            action = self.policy.evaluate_policy(obs)
            obs, reward, done, _ = self.env.step(action)
            num_plays += 1
            sum_rewards += reward
        return sum_rewards

    def train(self):
        return_reward = []
        for step in range(self.hp.nb_steps):
            start_time = datetime.datetime.now()
            # initialize the random noise deltas and the positive/negative rewards
            deltas = self.policy.sample_deltas()
            positive_rewards = [0] * self.hp.num_deltas
            negative_rewards = [0] * self.hp.num_deltas

            # play an episode each with positive deltas and negative deltas, collect rewards
            for k in range(self.hp.num_deltas):
                positive_rewards[k] = self.explore_policy(deltas[k], direction="+")
                negative_rewards[k] = self.explore_policy(deltas[k], direction="-")

            # Compute the standard deviation of all rewards
            sigma_rewards = np.array(positive_rewards + negative_rewards).std()

            # Sort the rollouts by the max(r_pos, r_neg) and select the deltas with best rewards
            scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.hp.num_best_deltas]
            rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

            # Update the policy
            self.policy.update(rollouts, sigma_rewards)

            # Only record video during evaluation, every n steps
            if step % self.hp.record_every == 0:
                self.record_video = True
            # Play an episode with the new weights and print the score
            reward_evaluation = self.evaluate_policy()
            print('Step: ', step, 'Reward: ', reward_evaluation)
            return_reward.append(reward_evaluation)
            self.record_video = False
            print('Running time: ', (datetime.datetime.now() - start_time).seconds)
        return return_reward.copy()


env = 'HalfCheetah-v2'
seeds = [1, 2, 3, 4, 5]
observation_dim = gym.make(env).observation_space.shape[0]
action_dim = gym.make(env).action_space.shape[0]

for i in range(5):
    hp = HP(env=gym.make(env), noise=0.03, nb_steps=1000, episode_length=1000, num_deltas=16,
            num_best_deltas=8, learning_rate=0.01, seed=seeds[i],
            n_policy_sampling=1, policy_variance=0.01, alpha=0.1)
    trainer = ARSTrainer(hp=hp, monitor_dir=None, input_size=observation_dim, output_size=action_dim)
    reward = trainer.train()
    pickle.dump(reward, open(env + '_mears_analysis_seeds_' + str(seeds[i]), 'wb'))
