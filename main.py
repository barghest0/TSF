import ccxt
import pandas as pd
import time

# Инициализация клиента Bybit
exchange = ccxt.bybit({
    'rateLimit': 1200,
    'enableRateLimit': True
})

# Настройка рынка и таймфрейма
symbol = 'BTC/USDT'
timeframe = '1h'  # Можно использовать '1m', '5m', '15m', '1h', '1d' и т.д.
limit = 200  # Количество свечей за один запрос

# Загрузка исторических данных


def fetch_historical_data(symbol, timeframe, limit):
    since = exchange.parse8601('2022-01-01T00:00:00Z')  # Начало данных
    all_data = []

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Обновляем начало для следующей выборки
            time.sleep(exchange.rateLimit / 1000)  # Пауза между запросами
        except Exception as e:
            print(f"Ошибка: {e}")
            break

    return pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])


# Загрузка данных
df = fetch_historical_data(symbol, timeframe, limit)

# Обработка данных
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Визуализация данных
print(df.head())
