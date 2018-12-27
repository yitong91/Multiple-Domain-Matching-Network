import numpy as np
from collections import OrderedDict
import os 
import scipy.io
import utils

def initialize_signal():
    signal = {}
    signal['images'] = []
    signal['labels'] = []
    signal['domains'] = []
    return signal  

def load_SEED_dataset(dir_path,test_idx, uniform_id = False):
    names = np.load('./SEAD_names.npy').item() 
    files = [f for f in os.listdir(dir_path) if f.endswith('.mat')]
    pool = []
    if uniform_id:
        D = 2
    else:
        D = len(names.keys())
    source_train = {}
    source_valid = {}
    target = {}
    for i in range(len(names.keys())):
        if i == test_idx:
            continue
        pool.append(i)
    for i in range(len(files)):    
        mat = scipy.io.loadmat(dir_path + files[i])
        name = files[i].split('_')
        idx = names[name[0]]       
        signal = initialize_signal()
        signal['images'].append(np.transpose(mat['positive'].astype('float32'),axes = [0,2,1]))
        signal['images'].append(np.transpose(mat['neutral'].astype('float32'),axes = [0,2,1]))
        signal['images'].append(np.transpose(mat['negative'].astype('float32'),axes = [0,2,1]))
        signal['images'].append(np.transpose(mat['test_positive'].astype('float32'),axes = [0,2,1]))
        signal['images'].append(np.transpose(mat['test_neutral'].astype('float32'),axes = [0,2,1]))
        signal['images'].append(np.transpose(mat['test_negative'].astype('float32'),axes = [0,2,1]))
        signal['labels'].append(2. * np.ones((mat['positive'].shape[0],),dtype = np.float32))
        signal['labels'].append(1. * np.ones((mat['neutral'].shape[0],),dtype = np.float32))
        signal['labels'].append(0. * np.ones((mat['negative'].shape[0],),dtype = np.float32))
        signal['labels'].append(2. * np.ones((mat['test_positive'].shape[0],),dtype = np.float32))
        signal['labels'].append(1. * np.ones((mat['test_neutral'].shape[0],),dtype = np.float32))
        signal['labels'].append(0. * np.ones((mat['test_negative'].shape[0],),dtype = np.float32))
        N = mat['positive'].shape[0] + mat['neutral'].shape[0] + mat['negative'].shape[0] + mat['test_positive'].shape[0] + mat['test_neutral'].shape[0] + mat['test_negative'].shape[0]
        if uniform_id:
            if idx == test_idx:
                signal['domains']= np.ones((N,),dtype = np.float32)
            else:
                signal['domains']= np.zeros((N,),dtype = np.float32)
        else:
            signal['domains']= idx * np.ones((N,),dtype = np.float32)
        if idx == test_idx:
            target = signal
        elif idx in pool:
            if not source_train.has_key(name[0]):
                source_train[name[0]] = initialize_signal()     
            for j in range(len(signal['images'])):
                source_train[name[0]]['images'].append(signal['images'][j])
                source_train[name[0]]['labels'].append(signal['labels'][j])
            source_train[name[0]]['domains'].append(signal['domains'])   
        else:
            print('Error in %s.',name[0])
    for name in source_train.keys():
        source_train[name]['images'] = np.concatenate(source_train[name]['images'],axis = 0)
        source_train[name]['labels'] = utils.to_one_hot(np.concatenate(source_train[name]['labels'],axis = 0), N=3)
        source_train[name]['domains'] = utils.to_one_hot(np.concatenate(source_train[name]['domains'],axis = 0), N=D)
        N = source_train[name]['domains'].shape[0]
        N1 = np.int32(np.ceil(0.7 * N))
        shuffle = np.random.permutation(N)
        source_valid[name] = initialize_signal()
        source_valid[name]['images'] = source_train[name]['images'][shuffle[N1:]]
        source_valid[name]['labels'] = source_train[name]['labels'][shuffle[N1:]]
        source_valid[name]['domains'] = source_train[name]['domains'][shuffle[N1:]]
        source_train[name]['images'] = source_train[name]['images'][shuffle[:N1]]
        source_train[name]['labels'] = source_train[name]['labels'][shuffle[:N1]]
        source_train[name]['domains'] = source_train[name]['domains'][shuffle[:N1]]   

    target['images'] = np.concatenate(target['images'],axis = 0)
    target['labels'] = utils.to_one_hot(np.concatenate(target['labels'],axis = 0), N=3)
    target['domains'] = utils.to_one_hot(target['domains'], N=D)   
    return source_train, source_valid, target