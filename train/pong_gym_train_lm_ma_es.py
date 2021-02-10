from games.pong_gym import PongGame
from lib.lm_ma_es import LM_MA_ES
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import save_obj, load_obj, calc_weight_count
from multiprocessing import Process, Manager
import numpy as np
import sys
import os

_processes = 8
topology = [ (3, sigmoid), (1, sigmoid)]

class pongThread(Process):
    def __init__(self, i, output, pop, input_size, top):
        super().__init__()
        self.id = i
        self.population = pop
        self.input_size = input_size
        self.topology = top
        self.output = output

    def run(self):
        game = PongGame()
        print('running {0}'.format(self.id))
        self.output[self.id] = -np.array([game.play(
            FixedTopologyNeuralNetwork(
                self.input_size, self.topology, ind)
                ) for ind in self.population])

lm_ma_es = LM_MA_ES(np.zeros(calc_weight_count(6, topology)), 1.0)
# cma_es = load_obj(200, 'models/cmaes_pong_v18')

generation = 0

while True:
    generation += 1
    print('Generation:', generation)
    population = lm_ma_es.sample()

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
    lm_ma_es.update(population, f_eval)
 

    if generation % 100 == 0:
        save_obj(lm_ma_es, generation, 'models/lm_ma_es_pong')
        #games[0].play(FixedTopologyNeuralNetwork(6, topology, lm_ma_es.sample()[0]), render=True, move_fun=PongGame.make_move_det)

#popsize 32
#v11 -> pong_v2 deterministic  -> topology = [(3, sigmoid), (1, sigmoid)]
#v12 -> pong_v2, random -> topology = [(3, sigmoid), (1, sigmoid)]

#popsize 16 (deafullt for cmaes)
#v13 -> same as v11
#v14 -> same as v12

#popsize 32 topology = [(6, sigmoid), (3, sigmoid), (1, sigmoid)] #v15 #v16

#popsize 32 normal pong [(3, sigmoid), (1, sigmoid)]]
#v17, v19
#v18
