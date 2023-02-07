import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from settings import CATERGORIES, CHECKPOINT_PATH, MAX_LEN
from utils import dataloader, get_callbacks
from my_model import BiLSTM_CNN
import sys

train_df = dataloader(['Sentence-level-Restaurant/Train.txt','Sentence-level-Restaurant/Dev.txt'])

#ready data
inputs = tf.convert_to_tensor(train_df['vector_text'].to_list())
outputs = [np.array(train_df[category].to_list()) for category in CATERGORIES]

#init model
model = BiLSTM_CNN()

def train(batch_size:int = 32, epochs:int = 5,pretrain:bool = False):
  model.build((None, MAX_LEN))
  optimizer = tfa.optimizers.RectifiedAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-5)
  list_loss = ['categorical_crossentropy' for _ in range(len(CATERGORIES))]
  model.compile(loss = list_loss, optimizer=optimizer, metrics=['accuracy'])
  if pretrain:
    model.load_weights(CHECKPOINT_PATH)

  #training
  callbacks = get_callbacks()
  model.fit(inputs, outputs,
        validation_split=0.1, shuffle = True , verbose= 2,
        batch_size=batch_size, epochs=epochs, callbacks=[callbacks])

  #save
  # model.save_weights(CHECKPOINT_PATH)

if __name__ == "__main__":
  try:
    sys.argv[1]
    pre_train = True
  except IndexError:
    pre_train = False

  train(pretrain= pre_train)
  print("finish train!")