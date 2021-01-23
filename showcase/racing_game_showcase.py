from games.racing_game import RacingGame
from lib.neat import Neat
from lib.utilities import load_obj

neat = load_obj(50, 'models/neat_race_v1')
game = RacingGame()
individual = neat.bestgens[-1][2]
game.play(individual, True)
