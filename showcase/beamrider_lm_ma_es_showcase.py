from games.beamrider_gym import BeamRiderGame
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import save_obj, load_obj, calc_weight_count
from gym import wrappers
from multiprocessing import Process, Manager
import numpy as np
import sys
import os

topology = [(64, sigmoid), (64, sigmoid), (4, sigmoid)]
<<<<<<< HEAD:showcase/beamrider_lm_ma_es_showcase.py
lm_ma_es = load_obj(3000, 'models/beamrider/lm_ma_es_beamrider')
=======
lm_ma_es = load_obj(3400, 'models/beamrider/lm_ma_es_beamrider')
>>>>>>> bd71222e80b7079a4e0679439d556094ef793264:showcase/beamrider_gym_showcase.py
    
game = BeamRiderGame()
individual = lm_ma_es.sample()[0]
game.play(FixedTopologyNeuralNetwork(128, topology, individual), True, wait=0.01)
