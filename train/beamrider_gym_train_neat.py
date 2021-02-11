from games.beamrider_gym import BeamRiderGame
from lib.neat import Neat
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import save_obj, load_obj, calc_weight_count
from multiprocessing import Process, Manager
import numpy as np
import sys
import os

_processes = 8

class beamriderThread(Process):
    def __init__(self, id, output, population):
        super().__init__()
        self.id = id
        self.population = population
        self.output = output

    def run(self):
        game = BeamRiderGame()
        self.output[self.id] = np.array([game.play(ind) for ind in self.population])

neat = Neat(128, 4)
neat = load_obj(550, 'models/beamrider/neat_beamrider')
    
generation = 550

while True:
    generation += 1
    print('Generation:', generation)
    population = neat.fenotypes

    chunks = np.array_split(population, _processes)
    results = Manager().list([None] * _processes)
    threads = [beamriderThread(i, results, chunks[i]) for i in range(_processes)]

    for th in threads:
        th.start()
    
    for th in threads:
        th.join()

    f_eval = np.concatenate(tuple(results))
    print(f_eval)
    print(f_eval.mean(), f_eval.max())
    neat.update(f_eval, verbose=True)
 

    if generation % 10 == 0:
        save_obj(neat, generation, 'models/beamrider/neat_beamrider')
        # games[0].play(neat.bestgens[-1][2], render=True)

