from games.pong import Game
from games.pong_gym import PongGame
from lib.cma_es import CMA_ES_Active
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
import numpy as np
import pickle

def save_obj(obj, gen):
    file = open('models/cma_es{0}.obj'.format(gen), 'wb') 
    pickle.dump(obj, file)
    file.close()

def load_obj(gen):
    file = open('models/cma_es{0}.obj'.format(gen), 'rb') 
    obj = pickle.load(file)
    file.close()
    return obj

# 4 input neurons, 1 hidden layer with 4 neurons, 1 output neuron -- weights + biases = 25
topology = [(4, sigmoid), (1, sigmoid)]
# cma_es = CMA_ES_Active(np.zeros(25), 1.0, popsize=30)
cma_es = load_obj(260)
game = Game()

generation = 260

while not cma_es.terminate():   
    generation += 1
    print('Generation:', generation)
    population = cma_es.sample()
    indiv = population[np.random.choice(population.shape[0])]
    f_eval = -np.array([game.play_two(FixedTopologyNeuralNetwork(4, topology, ind), FixedTopologyNeuralNetwork(4, topology, indiv)) for ind in population])
    print(f_eval)
    cma_es.update(population, f_eval)
 

    if generation % 10 == 0:
        save_obj(cma_es, generation)
        game.play_sample_game_two(FixedTopologyNeuralNetwork(4, topology, cma_es.sample()[0]), FixedTopologyNeuralNetwork(4, topology, cma_es.sample()[0]))

