from games.pong_gym_v2 import PongGame
from lib.neat import Neat
from lib.custom_top_nn import CustomTopologyNeuralNetwork as NN
from lib.activator_funcs import sigmoid
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
        self.game = PongGame()

    def run(self):
        self.output[self.id] = np.array([self.game.play(ind, move_fun=PongGame.make_move_det) for ind in self.population])


neat = Neat(6, 1)
# neat = load_obj(100, 'models/neat_pong_v1')

# for spe in neat.species:
#     print("spe")
#     for gen in spe.population:
#         print(len(gen.nodesGens), len(gen.linksGens), end='\t')

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

    if generation % 10 == 0:
        save_obj(neat, generation, 'models/neat_pong_v4')
        print(neat.bestgens[-1][0].nodesGens, neat.bestgens[-1][0].linksGens)

