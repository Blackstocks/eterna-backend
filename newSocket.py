"""
Proportional Scaling WebSocket Server
======================================
WebSocket server that provides real-time payout calculations using
proportional scaling approach with full parameterization including ts, te, and box size.
"""

import asyncio
import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import warnings
warnings.filterwarnings('ignore')

# Import the formula and fair pricing functions
from formula import calculate_box_payout, seconds_to_years
from fair import price_box_fair_ui, _calculate_fair_multiplier_simplified

app = FastAPI()

# Add CORS middleware for browser connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProportionalScaler:
    """Handles proportional scaling of payoffs to target range."""
    
    def __init__(self, min_payout: float = 1.0, max_payout: float = 40.0):
        self.min_payout = min_payout
        self.max_payout = max_payout
        self.cache = {}  # Cache for computed matrices
        
    def proportional_scale(self, raw_values: np.ndarray) -> np.ndarray:
        """
        Proportionally scale values to fit within target range.
        Preserves relative differences between all values.
        """
        min_val = np.min(raw_values)
        max_val = np.max(raw_values)
        
        if max_val == min_val:
            # All values are the same
            return np.full_like(raw_values, (self.min_payout + self.max_payout) / 2)
        
        # Scale proportionally
        scaled = self.min_payout + (raw_values - min_val) * \
                 (self.max_payout - self.min_payout) / (max_val - min_val)
        
        return scaled
    
    def calculate_single_payout(
        self,
        S0: float,
        offset: float,
        size: float,
        ts: float,
        te: float,
        sigma: float,
        r: float = 0.01,
        house_factor: float = 1.0,
        sym_mode: str = "simplified",
        driftless: bool = True,
        use_proportional: bool = True
    ) -> Dict[str, float]:
        """
        Calculate payout for a single position with specific ts and te.
        
        Parameters:
        -----------
        S0 : float
            Current spot price
        offset : float
            Box center offset from spot
        size : float
            Box width
        ts : float
            Time to start in seconds
        te : float
            Time to end in seconds
        sigma : float
            Volatility
        r : float
            Risk-free rate
        house_factor : float
            House edge factor
        sym_mode : str
            Symmetrization mode
        driftless : bool
            Use driftless calculation
        use_proportional : bool
            If True, apply proportional scaling; if False, return raw values
        """
        # Calculate box boundaries
        box_center = S0 + offset
        Klower = box_center - size / 2.0
        Kupper = box_center + size / 2.0
        
        # Calculate using the formula directly
        if sym_mode == "formula":
            # Use the formula.py approach
            r_eff = (0.5 * sigma * sigma) if driftless else r
            
            raw_fair = calculate_box_payout(
                S0=S0,
                Klower=Klower,
                Kupper=Kupper,
                ts=seconds_to_years(ts),
                te=seconds_to_years(te),
                r=r_eff,
                sigma=sigma,
                P=1.0,
                chit=0.0,
                cmiss=0.0,
                F=0.0,
                M=1e12,
                hit=True
            )
        else:
            # Use simplified calculation (default)
            # For simplified mode, we use total time = te - ts
            total_time = te - ts
            raw_fair = _calculate_fair_multiplier_simplified(
                S0=S0,
                offset=offset,
                size=size,
                ts_seconds=total_time,  # Use duration
                sigma=sigma,
                min_prob=None,  # No bounds for raw calculation
                max_prob=None,
                use_true_uncapped=True
            )
        
        # Apply house factor
        raw_after_house = raw_fair * house_factor
        
        if use_proportional:
            # For proportional scaling, we need to calculate the range
            # Sample multiple positions to determine scaling
            sample_offsets = list(range(-100, 105, 10))
            sample_values = []
            
            for sample_offset in sample_offsets:
                sample_center = S0 + sample_offset
                sample_Klower = sample_center - size / 2.0
                sample_Kupper = sample_center + size / 2.0
                
                if sym_mode == "formula":
                    r_eff = (0.5 * sigma * sigma) if driftless else r
                    sample_fair = calculate_box_payout(
                        S0=S0,
                        Klower=sample_Klower,
                        Kupper=sample_Kupper,
                        ts=seconds_to_years(ts),
                        te=seconds_to_years(te),
                        r=r_eff,
                        sigma=sigma,
                        P=1.0,
                        chit=0.0,
                        cmiss=0.0,
                        F=0.0,
                        M=1e12,
                        hit=True
                    )
                else:
                    sample_fair = _calculate_fair_multiplier_simplified(
                        S0=S0,
                        offset=sample_offset,
                        size=size,
                        ts_seconds=te - ts,
                        sigma=sigma,
                        min_prob=None,
                        max_prob=None,
                        use_true_uncapped=True
                    )
                
                sample_values.append(sample_fair * house_factor)
            
            # Now scale all values including our target
            all_values = np.array(sample_values + [raw_after_house])
            scaled_values = self.proportional_scale(all_values)
            scaled_payout = scaled_values[-1]  # Get our target value
        else:
            scaled_payout = raw_after_house
        
        return {
            'raw_fair': float(raw_fair),
            'raw_after_house': float(raw_after_house),
            'scaled_payout': float(scaled_payout),
            'offset': offset,
            'size': size,
            'ts': ts,
            'te': te,
            'duration': te - ts
        }
    
    def compute_matrix_with_te(
        self,
        S0: float,
        sigma: float,
        size: float,
        offsets: List[float],
        ts_values: List[float],
        te_values: Optional[List[float]] = None,
        fixed_duration: Optional[float] = None,
        r: float = 0.01,
        house_factor: float = 1.0,
        sym_mode: str = "simplified",
        driftless: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute matrix with variable ts and te values.
        
        Either provide te_values list (same length as ts_values) or fixed_duration.
        If fixed_duration is provided, te = ts + fixed_duration for each ts.
        """
        if te_values is None and fixed_duration is None:
            raise ValueError("Either te_values or fixed_duration must be provided")
        
        if te_values is None:
            te_values = [ts + fixed_duration for ts in ts_values]
        
        if len(ts_values) != len(te_values):
            raise ValueError("ts_values and te_values must have the same length")
        
        # Initialize matrices
        raw_matrix = np.zeros((len(offsets), len(ts_values)))
        
        # Compute raw values
        for ts_idx, (ts, te) in enumerate(zip(ts_values, te_values)):
            for offset_idx, offset in enumerate(offsets):
                result = self.calculate_single_payout(
                    S0=S0,
                    offset=offset,
                    size=size,
                    ts=ts,
                    te=te,
                    sigma=sigma,
                    r=r,
                    house_factor=house_factor,
                    sym_mode=sym_mode,
                    driftless=driftless,
                    use_proportional=False  # Get raw values first
                )
                raw_matrix[offset_idx, ts_idx] = result['raw_after_house']
        
        # Apply proportional scaling to entire matrix
        scaled_matrix = self.proportional_scale(raw_matrix)
        
        return raw_matrix, scaled_matrix


# Global scaler instance
scaler = ProportionalScaler()


@app.websocket("/ws/proportional_payout")
async def websocket_proportional_payout(websocket: WebSocket):
    """WebSocket endpoint for proportional scaling payout calculations."""
    await websocket.accept()
    print("New WebSocket client connected for proportional payouts")
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            # Handle different message types
            msg_type = data.get('type', 'calculate')
            
            if msg_type == 'configure':
                # Update configuration
                try:
                    if 'min_payout' in data:
                        scaler.min_payout = float(data['min_payout'])
                    if 'max_payout' in data:
                        scaler.max_payout = float(data['max_payout'])
                    
                    await websocket.send_json({
                        'type': 'config_updated',
                        'min_payout': scaler.min_payout,
                        'max_payout': scaler.max_payout
                    })
                except Exception as e:
                    await websocket.send_json({'error': f'Configuration error: {str(e)}'})
                continue
            
            elif msg_type == 'calculate':
                # Calculate payout for specific position
                try:
                    # Extract and validate parameters
                    offset = float(data.get('offset'))
                    size = float(data.get('size', 5.0))
                    ts = float(data.get('ts'))  # Time to start in seconds
                    te = float(data.get('te'))  # Time to end in seconds
                    S0 = float(data.get('S0'))  # Current spot price
                    sigma = float(data.get('sigma'))  # Volatility
                    
                    # Optional parameters
                    r = float(data.get('r', 0.01))
                    house_factor = float(data.get('house_factor', 1.0))
                    sym_mode = data.get('sym_mode', 'simplified')
                    driftless = data.get('driftless', True)
                    use_proportional = data.get('use_proportional', True)
                    
                    # Validate inputs
                    if S0 <= 0:
                        await websocket.send_json({'error': 'Spot price (S0) must be positive'})
                        continue
                    if sigma <= 0:
                        await websocket.send_json({'error': 'Volatility (sigma) must be positive'})
                        continue
                    if size <= 0:
                        await websocket.send_json({'error': 'Box size must be positive'})
                        continue
                    if te <= ts:
                        await websocket.send_json({'error': 'End time (te) must be greater than start time (ts)'})
                        continue
                    
                    # Calculate payout
                    result = scaler.calculate_single_payout(
                        S0=S0,
                        offset=offset,
                        size=size,
                        ts=ts,
                        te=te,
                        sigma=sigma,
                        r=r,
                        house_factor=house_factor,
                        sym_mode=sym_mode,
                        driftless=driftless,
                        use_proportional=use_proportional
                    )
                    
                    # Add additional info
                    result.update({
                        'type': 'payout',
                        'S0': S0,
                        'sigma': sigma,
                        'timestamp': asyncio.get_event_loop().time()
                    })
                    
                    await websocket.send_json(result)
                    
                except (TypeError, ValueError) as e:
                    await websocket.send_json({
                        'error': f'Invalid parameters: {str(e)}'
                    })
                except Exception as e:
                    await websocket.send_json({
                        'error': f'Calculation error: {str(e)}'
                    })
            
            elif msg_type == 'get_matrix':
                # Return full matrix for visualization
                try:
                    S0 = float(data.get('S0'))
                    sigma = float(data.get('sigma'))
                    size = float(data.get('size', 5.0))
                    
                    offsets = data.get('offsets', list(range(-100, 105, 5)))
                    ts_values = data.get('ts_values', list(range(5, 100, 5)))
                    
                    # Handle te values - either list or fixed duration
                    te_values = data.get('te_values', None)
                    fixed_duration = data.get('fixed_duration', 5.0)  # Default 5 seconds duration
                    
                    raw_matrix, scaled_matrix = scaler.compute_matrix_with_te(
                        S0=S0,
                        sigma=sigma,
                        size=size,
                        offsets=offsets,
                        ts_values=ts_values,
                        te_values=te_values,
                        fixed_duration=fixed_duration if te_values is None else None,
                        r=float(data.get('r', 0.01)),
                        house_factor=float(data.get('house_factor', 1.0)),
                        sym_mode=data.get('sym_mode', 'simplified'),
                        driftless=data.get('driftless', True)
                    )
                    
                    await websocket.send_json({
                        'type': 'matrix',
                        'raw_matrix': raw_matrix.tolist(),
                        'scaled_matrix': scaled_matrix.tolist(),
                        'offsets': offsets,
                        'ts_values': ts_values,
                        'te_values': te_values if te_values else [ts + fixed_duration for ts in ts_values],
                        'min_payout': scaler.min_payout,
                        'max_payout': scaler.max_payout
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        'error': f'Matrix generation error: {str(e)}'
                    })
            
            elif msg_type == 'batch_calculate':
                # Calculate multiple payouts at once
                try:
                    positions = data.get('positions', [])
                    S0 = float(data.get('S0'))
                    sigma = float(data.get('sigma'))
                    size = float(data.get('size', 5.0))
                    
                    # Optional parameters
                    r = float(data.get('r', 0.01))
                    house_factor = float(data.get('house_factor', 1.0))
                    sym_mode = data.get('sym_mode', 'simplified')
                    driftless = data.get('driftless', True)
                    use_proportional = data.get('use_proportional', True)
                    
                    results = []
                    for pos in positions:
                        result = scaler.calculate_single_payout(
                            S0=S0,
                            offset=float(pos['offset']),
                            size=size,
                            ts=float(pos['ts']),
                            te=float(pos['te']),
                            sigma=sigma,
                            r=r,
                            house_factor=house_factor,
                            sym_mode=sym_mode,
                            driftless=driftless,
                            use_proportional=use_proportional
                        )
                        results.append(result)
                    
                    await websocket.send_json({
                        'type': 'batch_results',
                        'results': results,
                        'S0': S0,
                        'sigma': sigma,
                        'size': size
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        'error': f'Batch calculation error: {str(e)}'
                    })
            
            elif msg_type == 'clear_cache':
                # Clear cached matrices
                scaler.cache.clear()
                await websocket.send_json({
                    'type': 'cache_cleared',
                    'message': 'Matrix cache cleared successfully'
                })
            
            elif msg_type == 'ping':
                # Handle ping messages for keep-alive
                await websocket.send_json({
                    'type': 'pong',
                    'timestamp': asyncio.get_event_loop().time()
                })
            
            else:
                await websocket.send_json({
                    'error': f'Unknown message type: {msg_type}'
                })
    
    except WebSocketDisconnect:
        print("Client disconnected from proportional payout websocket")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "service": "Proportional Scaling WebSocket Server",
        "endpoints": {
            "/ws/proportional_payout": "WebSocket for real-time payout calculations",
            "/health": "Health check endpoint"
        },
        "message_types": {
            "calculate": "Calculate payout for specific position",
            "batch_calculate": "Calculate multiple positions at once",
            "configure": "Update min/max payout range",
            "get_matrix": "Get full payout matrix",
            "clear_cache": "Clear matrix cache"
        },
        "required_params_for_calculate": {
            "offset": "Position offset from spot price",
            "size": "Box width",
            "ts": "Time to start in seconds",
            "te": "Time to end in seconds",
            "S0": "Current spot price",
            "sigma": "Volatility"
        },
        "optional_params": {
            "r": "Risk-free rate (default: 0.01)",
            "house_factor": "House edge factor (default: 1.0)",
            "sym_mode": "Calculation mode: 'simplified' or 'formula' (default: 'simplified')",
            "driftless": "Use driftless calculation (default: True)",
            "use_proportional": "Apply proportional scaling (default: True)",
            "min_payout": "Minimum payout for scaling (default: 1)",
            "max_payout": "Maximum payout for scaling (default: 40)"
        },
        "example_message": {
            "type": "calculate",
            "offset": 50,
            "size": 5,
            "ts": 10,
            "te": 15,
            "S0": 122213,
            "sigma": 0.8
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "cache_size": len(scaler.cache)}


if __name__ == "__main__":
    import uvicorn
    print("Starting Proportional Scaling WebSocket Server...")
    print("Connect to ws://localhost:8000/ws/proportional_payout")
    print("\nExample message format:")
    print(json.dumps({
        "type": "calculate",
        "offset": 50,
        "size": 5,
        "ts": 10,
        "te": 15,
        "S0": 122213,
        "sigma": 0.8
    }, indent=2))
    uvicorn.run(app, host="0.0.0.0", port=8000)