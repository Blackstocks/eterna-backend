import redis
import json
import time
from collections import deque
from threading import Thread, Event
import math
import numpy as np 
from newForm import calculate_and_scale_payout


def calculate_volatility(price_list):
        if len(price_list) < 2:
            return 0.0
        changes = [price_list[i+1] - price_list[i] for i in range(len(price_list)-1)]
        mean_change = sum(changes) / len(changes)
        variance = sum((c - mean_change) ** 2 for c in changes) / len(changes)
        return math.sqrt(variance)



def rolling_price_window(redis_url, channel, window_seconds=20, mark_interval_ms=100):
    r = redis.Redis.from_url(redis_url)
    p = r.pubsub()
    p.subscribe(channel)

    prices = deque()  # store tuples of (timestamp, price)
    stop_event = Event()

    def listen_prices():
        for message in p.listen():
            if stop_event.is_set():
                break
            if message['type'] == 'message':
                data = json.loads(message['data'])
                ts = data.get('ts')
                price = data.get('price')
                if ts is not None and price is not None:
                    prices.append((ts, price))
                    cutoff = ts - window_seconds
                    while prices and prices[0][0] < cutoff:
                        prices.popleft()

    

    def mark_ticks():
        while not stop_event.is_set():
            now = time.time()
            cutoff = now - window_seconds
            last_prices = [p[1] for p in prices if p[0] >= cutoff]
            last_40 = last_prices[-40:]

            volatility = calculate_volatility(last_40)
            if volatility == 0 : 
                volatility = 0.05
            offset = 25
            payout = calculate_and_scale_payout(
            S0=122200, Klower=122200-2.5, Kupper=122200+2.5,
            ts=0.0001, te=0.0002, r=0.01, sigma=volatility, P=1,
            offset=offset, scaling_method='sigmoid', scaling_kwargs={'max_offset':100, 'k':0.1}
            )
            print(f"Offset {offset}: Scaled payout = {payout:.4f}")

            # print(f"Marking {len(last_40)} prices at {now}: {last_40}")
            print(f"Calculated gradient-based volatility: {volatility}")

            time.sleep(mark_interval_ms / 1000)

    listener_thread = Thread(target=listen_prices, daemon=True)
    marker_thread = Thread(target=mark_ticks, daemon=True)

    listener_thread.start()
    marker_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        p.close()
        print("Stopped rolling price window.")


if __name__ == "__main__":
    # Example config - adjust your Redis URL and channel here if needed
    redis_url = "redis://127.0.0.1:6379"
    channel = "prices.btcusd"
    rolling_price_window(redis_url, channel, window_seconds=20, mark_interval_ms=100)
