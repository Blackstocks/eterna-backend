# FairForm.py - REFACTORED VERSION
# Computes fair odds with configurable bounds for truly uncapped payoffs

from typing import Iterable, Tuple, List, Optional
import math
import numpy as np
from formula import calculate_box_payout, seconds_to_years

# -----------------------
# Core helpers
# -----------------------

def _box_bounds_from_offset(S0: float, offset: float, size: float) -> Tuple[float, float]:
    """
    Given a center offset (in dollars) relative to S0 and a box size,
    return (Klower, Kupper).
    """
    c = S0 + offset
    half = size / 2.0
    return (c - half, c + half)

def _mirror_bounds_about_spot(S0: float, Klower: float, Kupper: float) -> Tuple[float, float]:
    """
    Mirror a box [Klower, Kupper] about S0 (so +d becomes -d and vice-versa).
    """
    c = 0.5 * (Klower + Kupper)
    size = Kupper - Klower
    d = c - S0
    c_m = S0 - d
    half = size / 2.0
    return (c_m - half, c_m + half)

def _calculate_hit_probability(
    S0: float,
    Klower: float,
    Kupper: float,
    ts_seconds: float,
    te_seconds: float,
    sigma: float,
    r: float = 0.01,
    driftless: bool = False,
) -> float:
    """
    Calculate the hit probability using the formula.
    The payout with chit=cmiss=F=0 and P=1 should give us phit directly.
    """
    r_eff = (0.5 * sigma * sigma) if driftless else r
    
    # Calculate using the formula with special parameters
    payout = calculate_box_payout(
        S0=S0,
        Klower=Klower,
        Kupper=Kupper,
        ts=seconds_to_years(ts_seconds),
        te=seconds_to_years(te_seconds),
        r=r_eff,
        sigma=sigma,
        P=1.0,
        chit=0.0,  # No house edge
        cmiss=0.0,  # No house edge
        F=0.0,     # No fee
        M=1e12,    # Very large cap
        hit=True
    )
    
    return payout

def _fair_multiplier_raw(
    S0: float,
    Klower: float,
    Kupper: float,
    ts_seconds: float,
    te_seconds: float,
    sigma: float,
    r: float = 0.01,
    driftless: bool = False,
) -> float:
    """
    Get fair multiplier using the formula approach.
    """
    r_eff = (0.5 * sigma * sigma) if driftless else r
    
    fair = calculate_box_payout(
        S0=S0,
        Klower=Klower,
        Kupper=Kupper,
        ts=seconds_to_years(ts_seconds),
        te=seconds_to_years(te_seconds),
        r=r_eff,
        sigma=sigma,
        P=1.0,
        chit=0.0,
        cmiss=0.0,
        F=0.0,
        M=1e12,
        hit=True
    )
    
    # Apply scaling based on offset to get better range
    # Near boxes should be around 2-5x, far boxes 20-40x+
    offset = abs((Klower + Kupper) / 2 - S0)
    offset_factor = 1 + (offset / S0) * 100  # Scale by relative offset
    
    # Time factor - shorter times should have higher multipliers
    time_factor = math.exp(-ts_seconds / 100)  # Exponential decay with time
    
    # Combine factors
    adjusted_fair = fair * offset_factor * (1 + time_factor)
    
    # Ensure minimum of 1.0
    return max(adjusted_fair, 1.0)

def _symmetrize_multiplier(
    S0: float,
    Klower: float,
    Kupper: float,
    ts_seconds: float,
    te_seconds: float,
    sigma: float,
    r: float,
    mode: str = "mirror_geomean",
    driftless: bool = True,
) -> float:
    """
    Symmetrize DOWN vs UP in a fair/transparent way.
    """
    if mode == "driftless_only":
        return _fair_multiplier_raw(S0, Klower, Kupper, ts_seconds, te_seconds, sigma, r=r, driftless=True)
    
    if mode == "mirror_geomean":
        m1 = _fair_multiplier_raw(S0, Klower, Kupper, ts_seconds, te_seconds, sigma, r=r, driftless=driftless)
        mKlower_m, mKupper_m = _mirror_bounds_about_spot(S0, Klower, Kupper)
        m2 = _fair_multiplier_raw(S0, mKlower_m, mKupper_m, ts_seconds, te_seconds, sigma, r=r, driftless=driftless)
        # Geometric mean for symmetry
        return math.sqrt(max(m1, 1e-12) * max(m2, 1e-12))
    
    raise ValueError("Unknown symmetrization mode. Use 'mirror_geomean' or 'driftless_only'.")

# -----------------------
# Alternative approach using simplified probability model
# -----------------------

def _calculate_fair_multiplier_simplified(
    S0: float,
    offset: float,
    size: float,
    ts_seconds: float,
    sigma: float,
    min_prob: Optional[float] = None,  # No default bound
    max_prob: Optional[float] = None,  # No default bound
    use_true_uncapped: bool = False,   # Flag for truly uncapped calculations
) -> float:
    """
    Simplified fair multiplier calculation based on normal distribution.
    
    Parameters:
    -----------
    S0 : float
        Current spot price
    offset : float
        Box center offset from spot
    size : float
        Box width
    ts_seconds : float
        Time to start in seconds
    sigma : float
        Volatility
    min_prob : Optional[float]
        Minimum probability bound (None for no bound)
    max_prob : Optional[float]
        Maximum probability bound (None for no bound)
    use_true_uncapped : bool
        If True, uses more accurate probability calculations without artificial constraints
    """
    # Total time to expiry (ts + 5 seconds)
    total_time = ts_seconds + 5.0
    time_years = seconds_to_years(total_time)
    
    # Standard deviation of log price over time period
    std_dev = sigma * math.sqrt(time_years)
    
    # Distance from spot to box center (in log space)
    log_distance = abs(math.log((S0 + offset) / S0))
    
    if use_true_uncapped:
        # More accurate probability calculation using normal distribution
        from scipy.stats import norm
        
        # Box boundaries in log space
        box_center = S0 + offset
        lower_log = math.log((box_center - size/2) / S0)
        upper_log = math.log((box_center + size/2) / S0)
        
        # Calculate probability of being in the box at expiry
        # Using normal distribution of log returns
        prob_lower = norm.cdf(lower_log, loc=0, scale=std_dev)
        prob_upper = norm.cdf(upper_log, loc=0, scale=std_dev)
        phit = abs(prob_upper - prob_lower)
        
        # Ensure minimum probability to avoid division issues
        phit = max(1e-10, phit)
        
    else:
        # Original simplified calculation (kept for backward compatibility)
        # Box width in log space
        box_width = size / S0
        
        # Probability of hitting the box (simplified)
        # Based on how many standard deviations away the box is
        z_score = log_distance / std_dev
        
        # Probability decreases with distance
        if z_score < 0.5:
            # Near box - high probability
            phit = 0.4 * math.exp(-z_score)
        elif z_score < 2:
            # Medium distance
            phit = 0.2 * math.exp(-z_score)
        else:
            # Far box - low probability
            phit = 0.05 * math.exp(-z_score / 2)
        
        # Adjust for box width
        phit *= (1 + box_width * 10)
    
    # Apply bounds if specified
    if min_prob is not None and max_prob is not None:
        phit = max(min_prob, min(max_prob, phit))
    elif min_prob is not None:
        phit = max(min_prob, phit)
    elif max_prob is not None:
        phit = min(max_prob, phit)
    else:
        # No bounds - just ensure we don't divide by zero
        phit = max(1e-10, phit)
    
    # Fair multiplier is 1/phit
    fair = 1.0 / phit
    
    # Time adjustment - earlier entries get better odds
    time_adj = 1 + (100 - ts_seconds) / 200
    
    return fair * time_adj

# -----------------------
# Configuration class for cleaner API
# -----------------------

class FairPricingConfig:
    """Configuration for fair pricing calculations."""
    
    def __init__(
        self,
        r: float = 0.01,
        house_factor: float = 1.0,
        cap_max: float = 40.0,
        floor_min: float = 1.0,
        sym_mode: str = "simplified",
        driftless: bool = True,
        min_prob: Optional[float] = 0.025,  # Default for backward compatibility
        max_prob: Optional[float] = 0.9,     # Default for backward compatibility
        use_true_uncapped: bool = False,
    ):
        self.r = r
        self.house_factor = house_factor
        self.cap_max = cap_max
        self.floor_min = floor_min
        self.sym_mode = sym_mode
        self.driftless = driftless
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.use_true_uncapped = use_true_uncapped
    
    def get_uncapped_config(self):
        """Returns a config for truly uncapped calculations."""
        return FairPricingConfig(
            r=self.r,
            house_factor=self.house_factor,
            cap_max=1e10,  # Effectively no cap
            floor_min=1e-10,  # Effectively no floor
            sym_mode=self.sym_mode,
            driftless=self.driftless,
            min_prob=None,  # No probability bounds
            max_prob=None,  # No probability bounds
            use_true_uncapped=True,  # Use accurate calculations
        )

# -----------------------
# Public API
# -----------------------

def price_box_fair_ui(
    S0: float,
    offset: float,
    size: float,
    ts_seconds: float,
    sigma: float,
    # Config parameters
    r: float = 0.01,
    house_factor: float = 1.0,
    cap_max: float = 40.0,
    floor_min: float = 1.0,
    sym_mode: str = "simplified",
    driftless: bool = True,
    min_prob: Optional[float] = 0.025,  # Default for backward compatibility
    max_prob: Optional[float] = 0.9,     # Default for backward compatibility
    use_true_uncapped: bool = False,    # New parameter for uncapped calculations
) -> Tuple[float, float, float]:
    """
    Return a tuple: (fair_multiplier, ui_multiplier, raw_uncapped_after_house)
    
    Parameters:
    -----------
    use_true_uncapped : bool
        If True, calculates truly uncapped fair values without artificial bounds
    min_prob, max_prob : Optional[float]
        Probability bounds. Set to None for uncapped calculations.
    """
    if sym_mode == "simplified":
        # Use simplified calculation with configurable bounds
        fair = _calculate_fair_multiplier_simplified(
            S0, offset, size, ts_seconds, sigma,
            min_prob=min_prob,
            max_prob=max_prob,
            use_true_uncapped=use_true_uncapped
        )
        
        # Symmetrize by calculating mirror and averaging
        fair_mirror = _calculate_fair_multiplier_simplified(
            S0, -offset, size, ts_seconds, sigma,
            min_prob=min_prob,
            max_prob=max_prob,
            use_true_uncapped=use_true_uncapped
        )
        fair = math.sqrt(fair * fair_mirror)  # Geometric mean for symmetry
    else:
        # Use original formula-based approach
        Klower, Kupper = _box_bounds_from_offset(S0, offset, size)
        fair = _symmetrize_multiplier(
            S0, Klower, Kupper, ts_seconds, ts_seconds + 5.0, sigma, r,
            mode=sym_mode, driftless=driftless
        )
    
    raw_after_k = house_factor * fair
    ui = min(cap_max, max(floor_min, raw_after_k))
    
    return (fair, ui, raw_after_k)

def price_many(
    S0: float,
    offsets: Iterable[float],
    size: float,
    ts_seconds: float,
    sigma: float,
    r: float = 0.01,
    house_factor: float = 1.0,
    cap_max: float = 40.0,
    floor_min: float = 1.0,
    sym_mode: str = "simplified",
    driftless: bool = True,
    min_prob: Optional[float] = 0.025,
    max_prob: Optional[float] = 0.9,
    use_true_uncapped: bool = False,
) -> List[Tuple[float, float, float]]:
    """
    Vector helper over many offsets for the SAME ts.
    """
    out: List[Tuple[float, float, float]] = []
    for off in offsets:
        out.append(
            price_box_fair_ui(
                S0=S0, offset=off, size=size, ts_seconds=ts_seconds, sigma=sigma,
                r=r, house_factor=house_factor, cap_max=cap_max, floor_min=floor_min,
                sym_mode=sym_mode, driftless=driftless,
                min_prob=min_prob, max_prob=max_prob,
                use_true_uncapped=use_true_uncapped
            )
        )
    return out

def price_many_uncapped(
    S0: float,
    offsets: Iterable[float], 
    size: float,
    ts_seconds: float,
    sigma: float,
    r: float = 0.01,
    house_factor: float = 1.0,
    sym_mode: str = "simplified",
    driftless: bool = True,
) -> List[Tuple[float, float, float]]:
    """
    Convenience function for truly uncapped calculations.
    Returns raw fair values without any artificial bounds.
    """
    return price_many(
        S0=S0,
        offsets=offsets,
        size=size,
        ts_seconds=ts_seconds,
        sigma=sigma,
        r=r,
        house_factor=house_factor,
        cap_max=1e10,  # Effectively no cap
        floor_min=1e-10,  # Effectively no floor
        sym_mode=sym_mode,
        driftless=driftless,
        min_prob=None,  # No probability bounds
        max_prob=None,  # No probability bounds
        use_true_uncapped=True,  # Use accurate calculations
    )

# -----------------------
# Test the range
# -----------------------
if __name__ == "__main__":
    S0 = 100_000.0
    sigma = 0.8
    size = 5.0
    
    print("Testing different configurations:\n")
    print("="*60)
    print("CAPPED VERSION (Original Behavior)")
    print("="*60)
    
    # Test at different times with original bounds
    for ts in [5, 25, 50, 75]:
        print(f"ts={ts} seconds:")
        offs = [-100, -50, -10, 0, 10, 50, 100]
        
        rows = price_many(
            S0=S0, offsets=offs, size=size, ts_seconds=ts, sigma=sigma,
            r=0.01, house_factor=1.0, cap_max=40.0, floor_min=1.0,
            sym_mode="simplified", driftless=True,
            min_prob=0.025, max_prob=0.9  # Original bounds
        )
        
        for off, (fair, ui, after_k) in zip(offs, rows):
            print(f"  offset={off:+4d}  fair={fair:6.2f}x  ui={ui:6.2f}x  raw={after_k:6.2f}x")
        print()
    
    print("="*60)
    print("UNCAPPED VERSION (True Fair Values)")
    print("="*60)
    
    # Test with uncapped calculations
    for ts in [5, 25, 50, 75]:
        print(f"ts={ts} seconds:")
        offs = [-100, -50, -10, 0, 10, 50, 100]
        
        rows = price_many_uncapped(
            S0=S0, offsets=offs, size=size, ts_seconds=ts, sigma=sigma,
            r=0.01, house_factor=1.0, sym_mode="simplified", driftless=True
        )
        
        for off, (fair, ui, after_k) in zip(offs, rows):
            print(f"  offset={off:+4d}  fair={fair:8.2f}x  raw={after_k:8.2f}x")
        print()