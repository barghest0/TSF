# train_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2


# Загружаем данные
df = pd.read_csv('data/bybit.csv', index_col='timestamp', parse_dates=True)

# Создание целевой переменной (1: цена вверх, 0: цена вниз)
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df.dropna(inplace=True)

# Масштабируем данные
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(
    df[['open', 'high', 'low', 'close', 'volume']])

# Создаём временные ряды
sequence_length = 30


def create_sequences(data, target, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])  # Берём последние N свечей
        y.append(target[i + sequence_length])  # Берём метку следующей свечи
    return np.array(X), np.array(y)


X, y = create_sequences(scaled_data, df['target'].values, sequence_length)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False)

# === 2. Построение и обучение модели GRU ===

model = Sequential([
    Bidirectional(GRU(100, return_sequences=True),
                  input_shape=(sequence_length, X.shape[2])),
    Dropout(0.3),
    GRU(100, kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Ранний останов, чтобы избежать переобучения
early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)


def lr_scheduler(epoch):
    if epoch < 10:
        return 0.001  # Начальная скорость
    else:
        return 0.0001  # Уменьшаем скорость по мере обучения


lr_schedule = LearningRateScheduler(lr_scheduler)

# Обучение модели
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, lr_schedule],
    verbose=1
)

# === 3. Оценка модели ===
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Точность на тестовой выборке: {test_accuracy:.2%}")

# === 4. Предсказание ===


def predict_next_candle(data, model, scaler):
    """Предсказание направления следующей свечи"""
    last_sequence = data[-sequence_length:]  # Последние 30 свечей
    # Добавляем измерение для GRU
    last_sequence = np.expand_dims(last_sequence, axis=0)
    prediction = model.predict(last_sequence)[0][0]
    direction = "вверх" if prediction > 0.5 else "вниз"
    return direction, prediction


direction, confidence = predict_next_candle(scaled_data, model, scaler)
print(f"Предсказание для следующей свечи: {
      direction} (уверенность: {confidence:.2%})")

# === 5. Сохранение модели ===
model.save('models/gru_bybit_model.keras')
