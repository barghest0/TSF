import ccxt
import pandas as pd
import time
from datetime import datetime, timezone

# Инициализация клиента Bybit
exchange = ccxt.bybit({
    'rateLimit': 1200,
    'enableRateLimit': True
})

# Настройка рынка и таймфрейма
symbol = 'BTC/USDT'
timeframe = '1h'  # Можно использовать '1m', '5m', '15m', '1h', '1d' и т.д.
limit = 200  # Количество свечей за один запрос


def get_start_of_day():
    now = datetime.now(timezone.utc)  # Текущее время с временной зоной UTC
    # Полночь текущего дня (UTC)
    start_of_day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    return int(start_of_day.timestamp() * 1000)


def fetch_historical_data(symbol, timeframe, limit):
    since = get_start_of_day()  # Загружаем с 2023 года
    all_data = []

    while True:
        try:
            print(f"Загружаем данные с {since}")  # Лог
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            print(f"ohlcv {ohlcv}")  # Лог
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Обновляем начало для следующей выборки
            time.sleep(exchange.rateLimit / 1000)  # Пауза между запросами
        except Exception as e:
            print(f"Ошибка: {e}")
            break

    print(f"Загружено {len(all_data)} записей.")
    return pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])


# Загрузка данных
df = fetch_historical_data(symbol, timeframe, limit)
print(df)

# Преобразование timestamp в читаемый формат
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Сохраняем данные в CSV
df.to_csv('data/today/bybit.csv', index=True)
