# parameters
# Created by Antoine Didisheim, at 06.08.19
# job: store default parameters used throughout the projects in single .py

import itertools
from enum import Enum
import numpy as np
import pandas as pd
import socket


##################
# Constant
##################

class Constant:
    MAX_OPT = 1000
    GRID_SIZE = 1000


##################
# params Enum
##################

class Optimizer(Enum):
    SGD = 1
    SGS_DECAY = 2
    ADAM = 3
    RMS_PROP = 4
    ADAMAX = 5
    NADAM = 5
    ADAGRAD = 5


class Loss(Enum):
    MSE = 1
    MAE = 2

class CSSAMPLE(Enum):
    FULL = 1
    VILK = 2
    KELLY = 3

class ModelType(Enum):
    DNN = 1
    LSTM = 2








##################
# params classes
##################

class ParamsModels:
    def __init__(self):
        # self.kernel = RBF(length_scale=100)
        self.save_dir = './model_save/'
        self.res_dir = './res/'
        self.model_type = ModelType.DNN
        self.E = 3
        #
        self.layers = [64, 32, 16]
        # self.layers
        self.batch_size = 512
        self.activation = 'relu'
        self.opti = Optimizer.ADAM
        self.loss = Loss.MSE
        self.learning_rate = 0.001
        self.dropout = 0.01
        self.output_range = 0.1
        self.output_pos_only =True

        self.tex_dir = 'tex_res'
        self.tex_name = 'theta_prime'

        self.batch_normalization = True
        self.regulator = True





class DataParams:
    def __init__(self):
        if socket.gethostname() == 'work':
            self.dir = '/media/antoinedidisheim/ssd_ntfs/theta_prime/data/'
        elif socket.gethostname() == 'walrus':
            self.dir = '/hdd/antoine_hdd/theta_prime/data/'

        else:
            self.dir = 'data/'
        self.val_split = 0.01

        self.max_ret_in_training = None
        self.var_subset = None
        self.cs_sample = CSSAMPLE.FULL
        self.H = 20
        self.include_mom = True
        self.inter_quartile_version = True




# store all parameters into a single object
class Params:
    def __init__(self):
        self.name_detail = 'default'
        self.name = ''
        self.seed = 12345
        self.model = ParamsModels()
        self.data = DataParams()
        self.process = None
        self.update_model_name()

    def update_model_name(self):
        n = self.name_detail

        ## architecture name
        L = 'L'
        for l in self.model.layers:
            L += str(l) + '_'
        n += L
        n += 'Lr' + str(self.model.learning_rate)
        n += 'Dropout' + str(self.model.dropout)
        n += 'BS' + str(self.model.batch_size)
        n += 'Act' + str(self.model.activation)
        n += 'OutRange' + str(self.model.output_range)
        if self.model.output_pos_only:
            n+='p'
        n += 'Loss' + str(self.model.loss.name)

        if self.data.H!=20:
            n+= f'H{self.data.H}'

        ## data name
        if self.data.var_subset is not None:
            n+= 'Vsubset'+str(len(self.data.var_subset))
        if self.data.max_ret_in_training is not None:
            n+= 'Mret'+str(self.data.max_ret_in_training)


        if self.model.batch_normalization:
            n+='BatchN'
        if self.model.regulator:
            n+='l1'

        if self.data.include_mom:
            n+="WithMom"

        n+='Cssample'+str(self.data.cs_sample.name)

        n = n.replace('.','')
        self.name = n

    def print_values(self):
        """
        Print all parameters used in the model
        """
        for key, v in self.__dict__.items():
            try:
                print('########', key, '########')
                for key2, vv in v.__dict__.items():
                    print(key2, ':', vv)
            except:
                print(v)

    def update_param_grid(self, grid_list, id_comb):
        ind = []
        for l in grid_list:
            t = np.arange(0, len(l[2]))
            ind.append(t.tolist())
        combs = list(itertools.product(*ind))
        print('comb', str(id_comb + 1), '/', str(len(combs)))
        c = combs[id_comb]

        for i, l in enumerate(grid_list):
            self.__dict__[l[0]].__dict__[l[1]] = l[2][c[i]]

    def save(self, save_dir, file_name='/parameters.p'):
        # simple save function that allows loading of deprecated parameters object
        df = pd.DataFrame(columns=['key', 'value'])

        for key, v in self.__dict__.items():
            try:
                for key2, vv in v.__dict__.items():
                    temp = pd.DataFrame(data=[str(key) + '_' + str(key2), vv], index=['key', 'value']).T
                    df = df.append(temp)

            except:
                temp = pd.DataFrame(data=[key, v], index=['key', 'value']).T
                df = df.append(temp)
        df.to_pickle(save_dir + file_name, protocol=4)

    def load(self, load_dir, file_name='/parameters.p'):
        # simple load function that allows loading of deprecated parameters object
        df = pd.read_pickle(load_dir + file_name)
        # First check if this is an old pickle version, if so transform it into a df
        if type(df) != pd.DataFrame:
            loaded_par = df
            df = pd.DataFrame(columns=['key', 'value'])
            for key, v in loaded_par.__dict__.items():
                try:
                    for key2, vv in v.__dict__.items():
                        temp = pd.DataFrame(data=[str(key) + '_' + str(key2), vv], index=['key', 'value']).T
                        df = df.append(temp)

                except:
                    temp = pd.DataFrame(data=[key, v], index=['key', 'value']).T
                    df = df.append(temp)

        no_old_version_bug = True

        for key, v in self.__dict__.items():
            try:
                for key2, vv in v.__dict__.items():
                    t = df.loc[df['key'] == str(key) + '_' + str(key2), 'value']
                    if t.shape[0] == 1:
                        tt = t.values[0]
                        self.__dict__[key].__dict__[key2] = tt
                    else:
                        if no_old_version_bug:
                            no_old_version_bug = False
                            print('#### Loaded parameters object is depreceated, default version will be used')
                        print('Parameter', str(key) + '.' + str(key2), 'not found, using default: ',
                              self.__dict__[key].__dict__[key2])

            except:
                t = df.loc[df['key'] == str(key), 'value']
                if t.shape[0] == 1:
                    tt = t.values[0]
                    self.__dict__[key] = tt
                else:
                    if no_old_version_bug:
                        no_old_version_bug = False
                        print('#### Loaded parameters object is depreceated, default version will be used')
                    print('Parameter', str(key), 'not found, using default: ', self.__dict__[key])
