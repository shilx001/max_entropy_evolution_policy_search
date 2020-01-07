import cma_es
import pickle

env = 'Swimmer-v2'
seed = [1, 2, 3, 4, 5]
population_size = 10
sigma_init = 0.5
step_size_init = 0.1
weight_decay = 0.01
total_episode = 2000

for i in seed:
    agent = cma_es.CMA(env=env, population_size=population_size, sigma_init=sigma_init,
                       step_size_init=step_size_init,
                       weight_decay=weight_decay,
                       total_episode=total_episode)
    reward, _ = agent.train()
    pickle.dump(env + '_cma_seeds_' + str(i))
