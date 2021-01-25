
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lib.utilities import load_obj


model = load_obj(700, 'models/maes_racing_v1') 
scores = model.bestgens

# model = load_obj(200, 'models/neat_race_v3')
# scores = [] 
# for gen, scr, fen in model.bestgens:
    # scores.append(scr) 

plt.plot(scores, label='best')
plt.plot(model.genmeans, label='mean')

aver = 20
plt.plot(range(aver-1, len(scores)), np.convolve(scores, np.ones(aver) / aver, mode='valid'), label=f'moving best avg {aver}')
plt.plot(range(aver-1, len(model.genmeans)), np.convolve(model.genmeans, np.ones(aver) / aver, mode='valid'), label=f'moving mean avg {aver}')
plt.legend()
plt.show()

