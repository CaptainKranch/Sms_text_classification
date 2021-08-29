import os
import re
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

nltk.download('stopwords') # download stopwords
nltk.download('wordnet')   # download vocab for lemmatizer

from tensorflow import keras
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

dataset_train = pd.read_csv(train_file_path, sep = "\t", header = None, names = ["y", "x"])
# print(dataset_train.head())

dataset_test = pd.read_csv(test_file_path, sep = "\t", header = None, names = ["y", "x"])
# print(dataset_test.head())

y_train =  dataset_train["y"].astype("category").cat.codes
y_test = dataset_test["y"].astype("category").cat.codes
# print(y_train[:5])
# print(y_test[:5])


bar = dataset_train['y'].value_counts()

plt.bar(bar.index, bar)
plt.xlabel('Label')
plt.title('Number of ham and spam messages')


stopwords_eng = set(stopwords.words('english'))
print(len(stopwords_eng))

lemmatizer = WordNetLemmatizer()

def clean_txt(txt):
    txt = re.sub(r'([^\s\w])+', ' ', txt)
    txt = " ".join([lemmatizer.lemmatize(word) for word in txt.split()
                    if not word in stopwords_eng])
    txt = txt.lower()
    return txt

X_train = dataset_train['x'].apply(lambda x: clean_txt(x))
print(X_train[:5])

max_words = 1000
max_len = 500

t = Tokenizer(num_words=max_words)
t.fit_on_texts(X_train)

sequences = t.texts_to_sequences(X_train)
print(sequences[:5])

sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
print(sequences_matrix[:5])



i = tf.keras.layers.Input(shape=[max_len])
x = tf.keras.layers.Embedding(max_words, 50, input_length=max_len)(i)
x = tf.keras.layers.LSTM(64)(x)

x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1, activation='relu')(x)

model = tf.keras.models.Model(inputs=i, outputs=x)
model.compile(
    loss='binary_crossentropy',
    optimizer='RMSprop',
    metrics=['accuracy']
)
model.summary()

r = model.fit(sequences_matrix, y_train,
              batch_size=128, epochs=10,
              validation_split=0.2,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor='val_loss', min_delta=0.0001)])


plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()


plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()

def preprocessing(X):
  x = X.apply(lambda x: clean_txt(x))
  x = t.texts_to_sequences(x)
  return sequence.pad_sequences(x, maxlen=max_len)

var = model.evaluate(preprocessing(dataset_test['x']), y_test)

print('Loss: {:.3f}, Accuracy: {:.3f}'.format(var[0], var[1]))

def predict_message(pred_text):
  p = model.predict(preprocessing(pd.Series([pred_text])))[0]

  return (p[0], ("ham" if p<0.5 else "spam"))

pred_text = "how are you doing today?"

prediction = predict_message(pred_text)
print(prediction)

def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
