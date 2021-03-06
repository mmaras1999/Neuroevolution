from games.pong_gym_v2 import PongGame
from lib.cma_es import CMA_ES_Active
from lib.lm_ma_es import LM_MA_ES
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import save_obj, load_obj, calc_weight_count
from multiprocessing import Process, Manager
import numpy as np
import sys
import os

_processes = 4
topology = [ (3, sigmoid), (1, sigmoid)]

class pongThread(Process):
    def __init__(self, id, output, population, input_size, topology):
        super().__init__()
        self.id = id
        self.population = population
        self.input_size = input_size
        self.topology = topology
        self.output = output

    def run(self):
        # print('running process {0}'.format(self.id))
        game = PongGame()
        self.output[self.id] = -np.array([game.play(
            FixedTopologyNeuralNetwork(
                self.input_size, self.topology, ind)
                ,move_fun=PongGame.make_move_rand) for ind in self.population])

cma_es = CMA_ES_Active(np.zeros(calc_weight_count(6, topology)), 1.0, popsize=32)
#cma_es = load_obj(110, 'models/cmaes_pong_v22')

generation = 110

while True:
    generation += 1
    print('Generation:', generation)
    population = cma_es.sample()

    chunks = np.array_split(population, _processes)
    results = Manager().list([None] * _processes)
    threads = [pongThread(i, results, chunks[i], 6, topology) for i in range(_processes)]

    for th in threads:
        th.start()
    
    for th in threads:
        th.join()

    f_eval = np.concatenate(tuple(results))
    print(f_eval)
    print(f_eval.mean(), f_eval.min())
    cma_es.update(population, f_eval)
 

    if generation % 10 == 0:
        save_obj(cma_es, generation, 'models/pong_gym_v2/cmaes_pong_v22')
        
#popsize 32
#v11 -> pong_v2 deterministic  -> topology = [(3, sigmoid), (1, sigmoid)]
#v12 -> pong_v2, random -> topology = [(3, sigmoid), (1, sigmoid)]

#popsize 16 (deafullt for cmaes)
#v13 -> same as v11
#v14 -> same as v12

#popsize 32 topology = [(6, sigmoid), (3, sigmoid), (1, sigmoid)] #v15 #v16

#popsize 32 normal pong [(3, sigmoid), (1, sigmoid)]]
#v17, v19 #21
#v18, #v20 #22

#v23 lm-ma-es

