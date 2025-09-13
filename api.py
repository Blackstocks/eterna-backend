from fastapi import FastAPI
from pydantic import BaseModel
import redis
import json

from formula import calculate_box_payout
from newForm import scale_payout

app = FastAPI()

# Redis connection (adjust URL if needed)
redis_url = "redis://127.0.0.1:6379"
r = redis.Redis.from_url(redis_url)

# Variables to hold real-time values (initialize with defaults)
real_time_price = 122200
real_time_volatility = 0.8

class PayoutRequest(BaseModel):
    offset: float
    ts: float

@app.post("/payout_multiplier")
def compute_payout_multiplier(request: PayoutRequest):
    global real_time_price, real_time_volatility

    # Get current price and volatility from Redis keys
    price_data = r.get("latest_price")
    vol_data = r.get("latest_volatility")
    if price_data:
        real_time_price = float(price_data)
    if vol_data:
        real_time_volatility = float(vol_data)

    S0 = real_time_price
    sigma = real_time_volatility
    Klower = S0 - 2.5
    Kupper = S0 + 2.5
    te = request.ts + 0.0001  # small time window
    r_param = 0.01
    P = 1
    chit = 0.1
    cmiss = 0.1
    F = 0.01
    M = 20.0
    hit = True

    # Calculate raw box payout
    raw_payout = calculate_box_payout(S0, Klower, Kupper, request.ts, te, r_param, sigma, P, chit, cmiss, F, M, hit)

    # Calculate payout multiplier by scaling based on offset
    multiplier = scale_payout(request.offset)

    return {"payout_multiplier": multiplier}
