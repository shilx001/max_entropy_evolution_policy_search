import cma_es
import pickle

env = 'HalfCheetah-v2'
seed = [1, 2, 3, 4, 5]
population_size = 20
sigma_init = 1
step_size_init = 1
weight_decay = 0.01
total_episode = 2000

for i in seed:
    agent = cma_es.CMA(env=env, population_size=population_size, sigma_init=sigma_init,
                       step_size_init=step_size_init,
                       weight_decay=weight_decay,
                       total_episode=total_episode,seed=i)
    reward, _ = agent.train()
    pickle.dump(env + '_cma_seeds_' + str(i))
