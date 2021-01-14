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

# 4 input neurons, 1 hidden layer with 4 neurons, 1 output neuron -- weights + biases = 25
topology = [(4, sigmoid), (1, sigmoid)]
cma_es = CMA_ES_Active(np.zeros(25), 1.0)
game = PongGame()

generation = 1

while not cma_es.terminate():
    print('Generation:', generation)
    population = cma_es.sample()
    f_eval = -np.array([game.play(FixedTopologyNeuralNetwork(4, topology, ind)) for ind in population])
    print(f_eval)
    cma_es.update(population, f_eval)
    generation += 1

    if generation % 10 == 0:
        save_obj(cma_es, generation)

