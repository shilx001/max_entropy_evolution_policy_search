import gym
import ES
import numpy as np


class CMA:
    def __init__(self, env='Hopper-v2', seed=1, population_size=10, step_size_init=1, sigma_init=1, weight_decay=0.01,
                 episode_length=1000, total_episode=1000):
        self.env = gym.make(env)
        self.seed = seed
        self.population_size = population_size
        self.step_size_init = step_size_init
        self.sigma_init = sigma_init
        self.weight_decay = weight_decay
        self.episode_length = episode_length
        self.total_episode = total_episode
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.step = 0

    def fitness(self, policy_mat):
        obs = self.env.reset()
        total_reward = 0
        for step in range(self.episode_length):
            action = np.clip(obs.dot(np.reshape(policy_mat, [self.state_dim, self.action_dim])), -1, 1)
            next_obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            obs = next_obs
            if done:
                break
        self.step += step
        return total_reward

    def train(self):
        cma = ES.sepCMAES(num_params=self.state_dim * self.action_dim, pop_size=self.population_size, mu_init=None,
                          sigma_init=self.sigma_init, step_size_init=self.step_size_init,
                          weight_decay=self.weight_decay)
        reward_list = []
        step_list = []
        for episode in range(self.total_episode):
            solutions = cma.ask(pop_size=self.population_size)
            scores = [self.fitness(solutions[i]) for i in range(self.population_size)]
            pop = cma.tell(solutions, scores)
            top_fitness = max([scores[i] for i in pop])
            reward_list.append(top_fitness)
            step_list.append(self.step)
            print('#####')
            print('Episode: ', episode)
            print('Total step: ', self.step)
            print('Max fitness:', top_fitness)
        return reward_list,step_list

