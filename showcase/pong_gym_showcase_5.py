from games.pong_gym_v2 import PongGame
from lib.neat import Neat
from lib.utilities import load_obj

#TODO add neat showcases
neat = load_obj(110, 'models/neat_pong_v3')
game = PongGame()
individual = neat.bestgens[-1][2]
game.play(individual, True, wait=None, move_fun=PongGame.make_move_det)
