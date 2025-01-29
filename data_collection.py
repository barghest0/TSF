
import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import time

exchange = ccxt.bybit({
    'rateLimit': 1200,
    'enableRateLimit': True
})

symbol = 'BTC/USDT'
timeframe = '5m'
limit = 200

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


df = fetch_historical_data(symbol, timeframe, start_time, limit)

df.to_csv('data/bybit.csv')
