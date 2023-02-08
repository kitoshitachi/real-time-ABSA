from settings import CATERGORIES, CHECKPOINT_PATH, MAX_LEN
from my_model import BiLSTM_CNN
from utils import dataloader
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard
import tensorflow_addons as tfa

test_df = dataloader(['Sentence-level-Restaurant/Test.txt'])

#ready data
inputs = tf.convert_to_tensor(test_df['vector_text'].to_list())
outputs = [np.array(test_df[category].to_list()) for category in CATERGORIES]

tensorBoardCallback = TensorBoard(log_dir='/logs')
model = BiLSTM_CNN()
model.build((None, MAX_LEN))
optimizer = tfa.optimizers.RectifiedAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-5)
list_loss = ['categorical_crossentropy' for _ in range(len(CATERGORIES))]
model.compile(loss = list_loss, optimizer=optimizer, metrics=['accuracy'])
model.load_weights(CHECKPOINT_PATH).expect_partial()

#predict
predict = model.predict(inputs, callbacks = [tensorBoardCallback])
predict = pd.DataFrame(zip(*predict), columns = CATERGORIES)

for category in CATERGORIES:
  predict[category] = predict[category].apply(lambda x: np.argmax(x))
  test_df[category] = test_df[category].apply(lambda x: np.argmax(x))

result = {}
count_true = 0
count_all = 0
for category in CATERGORIES:
  for x1,x2 in zip(test_df[CATERGORIES][category].ne(0), predict[category].ne(0)):
    if x1 == True:
      count_all += 1
      if x1 == x2:
        count_true += 1
  
  result[category] = {'true_category': count_true, 'all': count_all}

print(" ============ độ chính xác dự đoán khía cạnh trong câu============ ")
for category in CATERGORIES:
  print(f"{category}: {round(result[category]['true_category']*100/result[category]['all'],2)}%")

count_all = 0
count_true = 0
for category in CATERGORIES:
  for x1,x2 in zip(test_df[CATERGORIES][category], predict[category]):
    if x1 != 0:
      count_all += 1
      if x1 == x2:
        count_true += 1
    else:
      if x2 != 0:
        count_all += 1
  
  result[category] = {'true_label': count_true, 'all_label': count_all}

print(" ============ độ chính xác dự đoán cảm xúc trên mỗi khía cạnh============ ")
for category in CATERGORIES:
  print(f"{category}: {round(result[category]['true_label']*100/result[category]['all_label'],2)}%")