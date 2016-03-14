import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping

max_features = 100000
maxlen = 100
batch_size = 32

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model = model_from_json(open('my_model_architecture.json').read())
model.load_weights('my_model_weights.h5')

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              class_mode="binary")



model.fit(
    X_train, y_train,
    batch_size=batch_size,
    nb_epoch=1,
    show_accuracy=True
)




result = model.predict_proba(X)

json_string = model.to_json()
open('my_model_architecture.json', 'w').write(json_string)
model.save_weights('my_model_weights.h5')

