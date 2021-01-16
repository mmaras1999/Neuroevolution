
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lib.utilities import load_obj


cma_es = load_obj(560, 'models/cmaes_pong_v10')

scores = []
for gen, scr in cma_es.bestgens:
    scores.append(scr)

plt.plot(scores, label='best')
aver = 30
plt.plot(range(aver-1, len(scores)), np.convolve(scores, np.ones(aver) / aver, mode='valid'), label=f'moving avg {aver}')
plt.legend()
plt.show()

