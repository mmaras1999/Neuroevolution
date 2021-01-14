import pickle
import os

def save_obj(obj, gen, path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    
    path = os.path.join(path, '{0}.obj'.format(gen))
    file = open(path, 'wb') 
    pickle.dump(obj, file)
    file.close()

def load_obj(path):
    pass

def calc_weight_count(input, topology):
    res = (input + 1) * topology[0][0]

    for layer in range(len(topology) - 1):
        res += (topology[layer][0] + 1) * (topology[layer + 1][0])

    return res
