import tensorflow as tf
from keras.layers import Embedding, SpatialDropout1D
from keras.layers import Bidirectional, LSTM, Conv1D
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.layers import Concatenate, Dropout, Dense
from settings import CATERGORIES, EMBEDDING_DIM, EMBEDDING_MATRIX, INPUT_VOCAB_SIZE, MAX_LEN, SEED
import tensorflow as tf
from keras.layers import Embedding, SpatialDropout1D
from keras.layers import Bidirectional, LSTM, Conv1D
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.layers import Concatenate, Dropout, Dense

class BiLSTM_CNN(tf.keras.Model):
    def __init__(self):
        super(BiLSTM_CNN, self).__init__()

        he_normal_initializer = tf.keras.initializers.he_normal(seed=SEED)

        self.embedding = Embedding(
            INPUT_VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN, weights=[EMBEDDING_MATRIX],  
            trainable=True, name = 'embedding'
        )
        
        self.BiLSTM = Bidirectional(
            LSTM(64, return_sequences=True), 
            name='BiLSTM', trainable=True
        )

        self.conv_1 = Conv1D(128, 3,
                  kernel_initializer=he_normal_initializer, name = "conv1", 
                  padding="same", activation="relu")

        self.conv_2 = Conv1D(128, 4,
                  kernel_initializer=he_normal_initializer, name = "conv2", 
                  padding="same", activation="relu")
        
        self.conv_3 = Conv1D(128, 5,
                  kernel_initializer=he_normal_initializer, name = "conv3", 
                  padding="same", activation="relu")                                
        
        self.global_maxpool1D = GlobalMaxPooling1D()
        self.global_avgpool1D = GlobalAveragePooling1D()

        self.concate = Concatenate(axis=1)

        self.dropout = Dropout(0.2)

        self.dense1 = Dense(300, name = 'fc_1', trainable=True)
        self.dense2 = Dense(100, name = 'fc_2', trainable=True)
        self.classifiers = [Dense(4, activation='softmax', name = f"output_{i}") for i in range(len(CATERGORIES))]

    def call(self, inputs):
        inputs = SpatialDropout1D(0.5)(self.embedding(inputs))

        inputs = self.BiLSTM(inputs)

        inputs = [self.conv_1(inputs), self.conv_2(inputs), self.conv_3(inputs)]

        inputs = [(self.global_maxpool1D(input), self.global_avgpool1D(input)) for input in inputs]

        max_pool, avg_pool = zip(*inputs) 

        max_pool = self.concate(max_pool)
        avg_pool = self.concate(avg_pool)
        combine = self.concate([max_pool, avg_pool])

        # 3 fully-connected layer with dropout regularization
        fc = self.dropout(self.dense1(combine))
        fc = self.dropout(self.dense2(fc))
        return [classifier(fc) for classifier in self.classifiers]
