from time import time
import re
import numpy as np
import pandas as pd
from ftfy import fix_text
from pyvi.ViTokenizer import ViTokenizer
from keras.utils import pad_sequences
from my_model import BiLSTM_CNN
from settings import CATERGORIES, CHECKPOINT_PATH, INPUT_TOKERNIZER, LABELS, MAX_LEN
import tensorflow as tf
import tensorflow_addons as tfa

def clean_doc(text, word_segment=False, lower_case=True):
    text = re.sub('\\s+',' ',text) # remove multiple white spaces

    text = re.sub(r"[0-9]+", " number ", text)
    text= fix_text(text).replace("\n", " ")
    text = re.sub('[^\w ]','', text) # remove all special chars except white space
    text = re.sub('\\s+',' ',text) # remove multiple white spaces
    text = text.strip()

    if lower_case == True:
        text = text.lower()

    text = ViTokenizer.tokenize(text)

    if word_segment == False:
        return text.replace('_',' ')
    else:
        return text

def to_category_vector(label):
    vector = np.zeros(4).astype(np.float64)

    if 'positive' in label:
        vector[1] = 1.0
    elif 'neutral' in label:
        vector[2] = 1.0
    elif 'negative' in label:
        vector[3] = 1.0
    else:
        vector[0] = 1.0
    return vector

def find_tag(label:str, category):
    try:
        return re.findall('('+category+',\s(positive|negative|neutral))',label)[0][0]
    except IndexError:
        return ''

def create_output(label:str, categories):
    return [to_category_vector(find_tag(label, category)) for category in categories]

def read_file(filename):
    with open(filename,'r',encoding='utf8') as f:
        lines = [line.rstrip() for line in f]
        lines = [lines[x:x+4] for x in range(0, len(lines), 4)]
        df = pd.DataFrame(lines, columns=['id','text','label','null']).drop(columns=['null'])
        df['text'] = df['text'].apply(fix_text)
        
        return df

def dataloader(file_names):
    #read data
    data = pd.concat([read_file(file_name) for file_name in file_names])

    # label to vector
    data[CATERGORIES] = pd.DataFrame(
        data['label'].apply(lambda x: create_output(x,CATERGORIES)).tolist(),
        index= data.index
        )

    #clean document
    data['text'] = data['text'].apply(clean_doc)

    #text to vector
    data['vector_text'] = INPUT_TOKERNIZER.texts_to_sequences(data['text'])

    #padding
    data['vector_text'] = pad_sequences(data['vector_text'], maxlen=MAX_LEN,padding="post").tolist()

    # #ready data
    # inputs = tf.convert_to_tensor(data['vector_text'].to_list())
    # outputs = [np.array(data[category].to_list()) for category in CATERGORIES]

    return data

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
def get_callbacks():
    # checkpoint
    checkpoint = ModelCheckpoint(
        filepath=CHECKPOINT_PATH, monitor='loss', verbose=1, 
        save_best_only=True, save_weights_only=True,
        mode='min')

    # Update info
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    early_stop = EarlyStopping(monitor='loss', patience=2, mode='min')

    # all the goodies
    return [checkpoint, tensorboard, early_stop]