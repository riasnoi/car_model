import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# Загрузка датасета
data = pd.read_csv('data.csv')

# Разделение на признаки и целевую переменную
X = data.drop(columns=['fault_type', 'Unnamed: 0'])
y = data['fault_type']

# Кодирование целевой переменной
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Расчет весов классов
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights = dict(enumerate(class_weights))

# Преобразование меток в категориальный вид
y_categorical = to_categorical(y_encoded)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train)
print(scaler)
X_test = scaler.transform(X_test)

# Создание модел
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train1.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Компиляция модели
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Обучение модели
history = model.fit(
    X_train1, y_train,
    epochs=25,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weights
)

# Оценка модели по метрике MSE
mse = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Mean Squared Error (MSE) on test data: {mse}")

# Предсказание для первых десяти записей из X_test
predictions = model.predict(X_test[:100])
counter = 0
for i in range(100):
    pred = np.argmax(predictions[i])
    real = np.argmax(y_test[i])
    if pred == real:
      counter += 1
      print(f"{i}+{counter} Предсказанное значение: {pred}, реальное значение: {real}")

model.save('car_model.keras')
