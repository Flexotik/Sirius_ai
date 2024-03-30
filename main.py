import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Загрузка данных из файла train.csv
train_data = pd.read_csv('train.csv', encoding='utf-8')

# Предобработка данных
X_train = train_data['text'].apply(lambda x: x[1:-1])  # Удаление кавычек
y_train = train_data['sentiment']

# Создание токенизатора и преобразование текста в последовательности чисел
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)

# Дополнение последовательностей до одинаковой длины
maxlen = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post')

# Создание и обучение нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 128),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # три класса: 0, 1, 2
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Загрузка данных из файла test.csv
test_data = pd.read_csv('test.csv', encoding='utf-8')

# Предобработка данных для тестового набора
X_test = test_data['text'].apply(lambda x: x[1:-1])  # Удаление кавычек
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post')

# Предсказание на тестовых данных
predictions = model.predict(X_test_pad)

# Преобразование вероятностей в предсказанные классы
predicted_classes = np.argmax(predictions, axis=1)

# Запись предсказаний в файл output.csv с кодировкой UTF-8
output_df = pd.DataFrame({'ID': test_data['ID'], 'N_sentiment': predicted_classes})
output_df.to_csv('output.csv', index=False, encoding='utf-8')
