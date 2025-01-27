import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta, timezone
import time

# === 1. Сбор данных с Bybit ===
exchange = ccxt.bybit({
    'rateLimit': 1200,
    'enableRateLimit': True
})

symbol = 'BTC/USDT'
timeframe = '5m'
limit = 200  # Количество свечей

start_time = int((datetime.now(timezone.utc) -
                 timedelta(days=1)).timestamp() * 1000)


def fetch_historical_data(symbol, timeframe, since, limit):
    """Функция для загрузки исторических данных"""
    all_data = []
    while True:
        try:
            print(f"Загружаем данные с {since}")  # Лог
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
            all_data.extend(ohlcv)  # Добавляем новые данные в общий список
            since = ohlcv[-1][0] + 1  # Обновляем начало для следующей выборки
            # Пауза для соблюдения лимитов API
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Ошибка при запросе данных: {e}")
            break
    # Преобразуем в DataFrame
    df = pd.DataFrame(all_data, columns=[
                      'timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(
        df['timestamp'], unit='ms')  # Преобразуем время
    df.set_index('timestamp', inplace=True)
    return df


# Загружаем данные
df = fetch_historical_data(symbol, timeframe, start_time, limit)

df.to_csv('data/bybit.csv')


# === 2. Подготовка данных для LSTM ===

# Добавление целевой переменной (1: цена вверх, 0: цена вниз)
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df.dropna(inplace=True)  # Убираем строки с NaN

# Масштабируем данные
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(
    df[['open', 'high', 'low', 'close', 'volume']])

# Создаем временные ряды


def create_sequences(data, target, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)


sequence_length = 30
X, y = create_sequences(scaled_data, df['target'].values, sequence_length)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False)

# === 3. Построение и обучение модели LSTM ===

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Для классификации используем сигмоиду
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Ранний останов, чтобы избежать переобучения
early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

# Обучение модели
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# === 4. Оценка модели ===
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Точность на тестовой выборке: {test_accuracy:.2%}")

# === 5. Предсказание ===


def predict_next_candle(data, model, scaler):
    """Предсказание направления следующей свечи"""
    last_sequence = data[-sequence_length:]  # Последние 30 свечей
    # Добавляем измерение для LSTM
    last_sequence = np.expand_dims(last_sequence, axis=0)
    prediction = model.predict(last_sequence)[0][0]
    direction = "вверх" if prediction > 0.5 else "вниз"
    return direction, prediction


direction, confidence = predict_next_candle(scaled_data, model, scaler)
print(f"Предсказание для следующей свечи: {
      direction} (уверенность: {confidence:.2%})")

# === 6. Сохранение модели ===
model.save('models/lstm_bybit_model.h5')
