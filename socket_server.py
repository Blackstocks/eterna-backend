from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import redis
import json

from formula import calculate_box_payout, seconds_to_years
from newForm import scale_payout, calculate_and_scale_payout
from fair import price_box_fair_ui

app = FastAPI()

redis_url = "redis://127.0.0.1:6379"
r = redis.Redis.from_url(redis_url)

# Default initial values
real_time_price = 122200
real_time_volatility = 0.8

@app.websocket("/ws/payout_multiplier")
async def websocket_payout_multiplier(websocket: WebSocket):
    await websocket.accept()
    global real_time_price, real_time_volatility
    try:
        while True:
            data = await websocket.receive_json()
            offset = data.get("offset")
            ts = data.get("ts")
            if offset is None or ts is None:
                await websocket.send_json({"error": "Missing 'offset' or 'ts' in request"})
                continue
            
            # Update real-time price and volatility from Redis if available
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
            te = ts + 0.0001
            r_param = 0.01
            P = 1
            chit = 0.1
            cmiss = 0.1
            F = 0.01
            M = 20.0
            hit = True

            raw_payout = calculate_box_payout(S0, Klower, Kupper, ts, te, r_param, sigma, P, chit, cmiss, F, M, hit)
            multiplier = scale_payout(offset)

            await websocket.send_json({"payout_multiplier": multiplier})
    except WebSocketDisconnect:
        print("Client disconnected from payout multiplier websocket")

@app.websocket("/ws/box_calculations")
async def websocket_box_calculations(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive calculation request
            data = await websocket.receive_json()
            
            # Extract parameters
            S0 = data.get("S0", real_time_price)
            offset = data.get("offset", 0.0)
            size = data.get("size", 5.0)
            ts_seconds = data.get("ts_seconds", 5.0)
            sigma = data.get("sigma", real_time_volatility)
            r = data.get("r", 0.01)
            house_factor = data.get("house_factor", 1.0)
            use_simplified = data.get("use_simplified", True)
            driftless = data.get("driftless", True)
            
            # Box bounds (if provided directly)
            K_lower = data.get("K_lower")
            K_upper = data.get("K_upper")
            
            # If bounds not provided, calculate from offset and size
            if K_lower is None or K_upper is None:
                box_center = S0 + offset
                half_size = size / 2.0
                K_lower = box_center - half_size
                K_upper = box_center + half_size
            
            # Time parameters
            ts = data.get("ts", ts_seconds / (365.25 * 24 * 3600))
            te = data.get("te", (ts_seconds + 5.0) / (365.25 * 24 * 3600))
            
            try:
                # Use fair pricing module (simplified approach)
                if use_simplified:
                    fair, ui, raw_after = price_box_fair_ui(
                        S0=S0,
                        offset=offset,
                        size=size,
                        ts_seconds=ts_seconds,
                        sigma=sigma,
                        r=r,
                        house_factor=house_factor,
                        cap_max=40.0,
                        floor_min=1.0,
                        sym_mode="simplified",
                        driftless=driftless,
                        use_true_uncapped=False  # Use capped version for UI
                    )
                    
                    # Also calculate uncapped version for reference
                    fair_uncapped, _, raw_uncapped = price_box_fair_ui(
                        S0=S0,
                        offset=offset,
                        size=size,
                        ts_seconds=ts_seconds,
                        sigma=sigma,
                        r=r,
                        house_factor=house_factor,
                        cap_max=1e10,
                        floor_min=1e-10,
                        sym_mode="simplified",
                        driftless=driftless,
                        use_true_uncapped=True
                    )
                    
                    # Calculate hit probability using the formula
                    hit_prob = calculate_box_payout(
                        S0=S0,
                        Klower=K_lower,
                        Kupper=K_upper,
                        ts=ts,
                        te=te,
                        r=r,
                        sigma=sigma,
                        P=1.0,
                        chit=0.0,
                        cmiss=0.0,
                        F=0.0,
                        M=1e12,
                        hit=True
                    )
                    
                    result = {
                        "calculations": {
                            "fair_multiplier": fair,
                            "ui_multiplier": ui,
                            "raw_multiplier": raw_after,
                            "fair_uncapped": fair_uncapped,
                            "raw_uncapped": raw_uncapped,
                            "hit_probability": hit_prob,
                            "offset": offset,
                            "box_size": size,
                            "ts_seconds": ts_seconds,
                            "box_bounds": {
                                "lower": K_lower,
                                "upper": K_upper,
                                "center": S0 + offset
                            }
                        }
                    }
                    
                else:
                    # Method 2: Use formula-based calculation
                    payout = calculate_and_scale_payout(
                        S0=S0,
                        Klower=K_lower,
                        Kupper=K_upper,
                        ts=ts,
                        te=te,
                        r=r,
                        sigma=sigma,
                        P=1.0,
                        offset=offset,
                        scaling_method='sigmoid',
                        scaling_kwargs={'max_offset': 100, 'k': 0.1, 'max_scale': 10},
                        final_min=1.0,
                        final_max=40.0,
                        use_normalization=True
                    )
                    
                    # Calculate hit probability
                    hit_prob = calculate_box_payout(
                        S0=S0,
                        Klower=K_lower,
                        Kupper=K_upper,
                        ts=ts,
                        te=te,
                        r=r,
                        sigma=sigma,
                        P=1.0,
                        chit=0.0,
                        cmiss=0.0,
                        F=0.0,
                        M=1e12,
                        hit=True
                    )
                    
                    result = {
                        "calculations": {
                            "fair_multiplier": payout,
                            "ui_multiplier": payout,
                            "raw_multiplier": payout,
                            "hit_probability": hit_prob,
                            "offset": offset,
                            "box_size": size,
                            "ts_seconds": ts_seconds,
                            "box_bounds": {
                                "lower": K_lower,
                                "upper": K_upper,
                                "center": S0 + offset
                            }
                        }
                    }
                
                await websocket.send_json(result)
                
            except Exception as calc_error:
                error_msg = f"Calculation error: {str(calc_error)}"
                print(f"BoxWebSocket: {error_msg}")
                await websocket.send_json({"error": error_msg})
                
    except WebSocketDisconnect:
        print("BoxWebSocket: Client disconnected")
