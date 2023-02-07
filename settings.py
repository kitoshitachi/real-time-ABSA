import numpy as np
import tensorflow as tf
import warnings, random
import pickle

warnings.filterwarnings("ignore")
SEED = 111

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

DOMAIN = 'Restaurant'
CATERGORIES = ['DRINKS#PRICES',
 'FOOD#PRICES',
 'FOOD#STYLE&OPTIONS',
 'RESTAURANT#PRICES',
 'LOCATION#GENERAL',
 'SERVICE#GENERAL',
 'DRINKS#QUALITY',
 'AMBIENCE#GENERAL',
 'FOOD#QUALITY',
 'RESTAURANT#MISCELLANEOUS',
 'RESTAURANT#GENERAL',
 'DRINKS#STYLE&OPTIONS']

LABELS = [None,'positive','negative','neutral']

with open('input_tokenizer.pkl', 'rb') as fp:
    INPUT_TOKERNIZER = pickle.load(fp)


EMBEDDING_DIM = 100
MAX_LEN = 96
INPUT_VOCAB_SIZE = 4728
with open('embedding_matrix.pkl', 'rb') as fp:
    EMBEDDING_MATRIX = pickle.load(fp)
CHECKPOINT_PATH ='D:\\19521204\python\\realtime_ABSA\checkpoint_dir\checkpoint'