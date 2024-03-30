import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Загрузка данных
train_data = pd.read_csv('train.csv',on_bad_lines='skip')
test_data = pd.read_csv('test.csv')

# Заполнение пропущенных значений пустыми строками
train_data['text'].fillna('', inplace=True)
test_data['text'].fillna('', inplace=True)

# Преобразование столбца 'labels' к строковому типу
train_data['labels'] = train_data['labels'].astype(str)

# Преобразование целевых меток в бинарный формат
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_data['labels'].apply(lambda x: x.strip('{}').split(', ')))

# Создание и обучение модели
classifier = MultiOutputClassifier(DecisionTreeClassifier())
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', classifier)
])
pipeline.fit(train_data['text'], y_train)

# Предсказание на тестовых данных
predictions = pipeline.predict(test_data['text'])

# Обратное преобразование предсказанных меток в формат строк
predicted_labels = mlb.inverse_transform(predictions)

# Запись предсказаний в файл
output_df = pd.DataFrame({'ID': range(1, len(test_data) + 1), 'labels': [', '.join(labels) for labels in predicted_labels]})
output_df['labels'] = output_df['labels'].apply(lambda x: '{' + x + '}')
output_df.to_csv('output.csv', index=False)
