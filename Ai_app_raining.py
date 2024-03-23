import tensorflow as tf
from transformers import BertTokenizer, T5ForConditionalGeneration
import tkinter as tk
from tkinter import ttk
import re 
import json

# Энкодер
encoder = BertTokenizer.from_pretrained('bert-base-uncased')

# Механизм внимания
attention = tf.keras.layers.Attention()

# Декодер
decoder = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Загрузка данных из файла JSON
def load_annotated_data(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        return data["messages"], data["summary"]
# Модель резюмирования
class SummarizationModel(tf.keras.Model):

    def __init__(self, encoder, attention, decoder):
        super().__init__()
        self.encoder = encoder
        self.attention = attention
        self.decoder = decoder

    # Предобработка сообщения
    def preprocess_data(messages):

        # Удаление лишних пробелов и переносов строк
        messages = [re.sub(r"\s+", " ", message) for message in messages]

        # Удаление ссылок
        messages = [re.sub(r"https?://\S+", "", message) for message in messages]

        # Удаление смайликов
        messages = [re.sub(r"[:;8][\w-]+", "", message) for message in messages]

        # Удаление имен пользователей
        messages = [re.sub(r"@\w+", "", message) for message in messages]

        # Удаление знаков препинания
        messages = [re.sub(r"[^\w\s]", "", message) for message in messages]

        # Преобразование к нижнему регистру
        messages = [message.lower() for message in messages]

        return messages

    def call(self, inputs):
        encoded_inputs = self.encoder(inputs['input_text'], return_tensors='tf')
        attention_weights = self.attention(encoded_inputs['last_hidden_state'])
        decoded_outputs = self.decoder.generate(encoded_inputs['input_ids'], attention_weights=attention_weights)
        return decoded_outputs

# Загрузка анотированных данных для обучения нейросети
train_input_texts, train_input_texts = load_annotated_data(r"Training_data.json")

# Обучение модели
train_data = tf.data.Dataset.from_tensor_slices((train_input_texts, train_input_texts))
model = SummarizationModel(encoder, attention, decoder)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(train_data, epochs=10)


# Окно приложения
root = tk.Tk()
root.title("Резюмер сообщений ВКонтакте")

# Текстовое поле для ввода сообщений
text_field = tk.Text(root)
text_field.pack()

# Кнопка для запуска резюмирования
summarize_button = ttk.Button(root, text="Резюмировать")
summarize_button.pack()

# Обработчик нажатия кнопки
def summarize_messages():
    messages = text_field.get("1.0", "end-1c")
    encoded_messages = encoder(messages, return_tensors='tf')
    attention_weights = attention(encoded_messages['last_hidden_state'])
    decoded_outputs = decoder.generate(encoded_messages['input_ids'], attention_weights=attention_weights)
    summary = decoder.decode(decoded_outputs, skip_special_tokens=True)
    summary_field.insert("1.0", summary)

summarize_button.config(command=summarize_messages)

# Текстовое поле для отображения резюме
summary_field = tk.Text(root)
summary_field.pack()

# Запуск приложения
root.mainloop()