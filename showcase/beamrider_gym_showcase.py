from games.beamrider_gym import BeamRiderGame
from lib.cma_es import CMA_ES_Active
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import save_obj, load_obj, calc_weight_count
from gym import wrappers
from multiprocessing import Process, Manager
import numpy as np
import sys
import os

topology = [(64, sigmoid), (4, sigmoid)]
cma_es = load_obj(10, 'models/beamrider/cmaes_v0')
    
game = BeamRiderGame()
individual = cma_es.sample()[0]
game.play(FixedTopologyNeuralNetwork(6, topology, individual), True, wait=0.1)
