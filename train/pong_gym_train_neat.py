from games.pong_gym_v2 import PongGame
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

_processes = 8

class PongProcess(Process):
    def __init__(self, id, output, population, input_size):
        super().__init__()
        self.id = id
        self.population = population
        self.input_size = input_size
        self.output = output

    def run(self):
        # print('running {0}'.format(self.id))
        self.output[self.id] = np.array([games[self.id].play(ind, move_fun=PongGame.make_move_det) for ind in self.population])


neat = Neat(6, 1)
# neat = load_obj(175, 'models/neat_pong_v5')

# for spe in neat.species:
#     print("spe")
#     for gen in spe.population:
#         print(len(gen.nodesGens), len(gen.linksGens), end='\t')

games = [PongGame() for i in range(_processes)]

generation = 0

while True:  
    generation += 1
    print('Generation:', generation)
    population = neat.getFenotypes()

    chunks = np.array_split(population, _processes)
    results = Manager().list([None] * _processes)
    threads = [PongProcess(i, results, chunks[i], 2) for i in range(_processes)]

    for th in threads:
        th.start()
    
    for th in threads:
        th.join()

    f_eval = np.concatenate(tuple(results))
    print(f_eval)
    print(f_eval.mean(), f_eval.max(), f_eval.min())
    neat.update(f_eval, verbose=True) #neat maximize function evals

    if generation % 5 == 0:
        save_obj(neat, generation, 'models/neat_pong_v6')
        games[0].play(neat.bestgens[-1][2], render=True)
        print(neat.bestgens[-1][0].nodesGens, neat.bestgens[-1][0].linksGens)



#v3 -> det move, old neat
#v4 -> random move, old neat
#v5 -> detr move, new neat, normlaized weights diff, treshold = 0.75 TODO train more
#v6 -> detr move, new neat, weights diff, treshold = 3 TODO train more
