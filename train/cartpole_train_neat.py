from games.balance_game import BalanceGame
from lib.neat import Neat
from lib.custom_top_nn import CustomTopologyNeuralNetwork as NN
from lib.cma_es import CMA_ES_Active
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import save_obj, load_obj, calc_weight_count
from multiprocessing import Process, Manager
import numpy as np
import sys
import os

_processes = 4

class BalanceProcess(Process):
    def __init__(self, id, output, population, input_size):
        super().__init__()
        self.id = id
        self.population = population
        self.input_size = input_size
        self.output = output
        self.game = BalanceGame()

    def run(self):
        # print('running {0}'.format(self.id))
        game = BalanceGame()
        self.output[self.id] = np.array([self.game.play(ind, move_fun=BalanceGame.make_move_det) for ind in self.population])


neat = Neat(4, 1)
generation = 0

while True:  
    generation += 1
    print('Generation:', generation)
    population = neat.getFenotypes()

    chunks = np.array_split(population, _processes)
    results = Manager().list([None] * _processes)
    threads = [BalanceProcess(i, results, chunks[i], 2) for i in range(_processes)]

    for th in threads:
        th.start()
    
    for th in threads:
        th.join()

    f_eval = np.concatenate(tuple(results))
    print(f_eval)
    print(f_eval.mean(), f_eval.max(), f_eval.min())
    neat.update(f_eval, verbose=True) #neat maximize function evals

    if generation % 10 == 0:
        save_obj(neat, generation, 'models/cartpole/neat_cartpole')
        print(neat.bestgens[-1][0].nodesGens, neat.bestgens[-1][0].linksGens)
        # game.play(neat.bestgens[-1][2], render=True, games_amount=1)
