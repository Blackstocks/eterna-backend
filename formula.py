import math
from scipy.stats import norm
from scipy.integrate import quad

 
def seconds_to_years(seconds: float) -> float:
    return seconds / (365.25 * 24 * 3600)

def calculate_box_payout(
    S0: float,
    Klower: float,
    Kupper: float,
    ts: float,
    te: float,
    r: float,
    sigma: float,
    P: float,
    chit: float = 0.1,
    cmiss: float = 0.1,
    F: float = 0.01,
    M: float = 20.0,
    hit: bool = True
) -> float:
    """
    Calculate payout of a box option based on start and end times, volatility, and other financial parameters.
    """

    Δts = ts
    Δt = te - ts

    if Δts == 0:
        if Klower <= S0 <= Kupper:
            return float(P) if hit else 0.0
        else:
            return 0.0 if hit else float(P)

    mu = r - 0.5 * sigma ** 2

    def phi(x):
        return norm.cdf(x)

    # Probability of spot inside the box at start time ts
    pfront = (phi((math.log(Kupper / S0) - (r - 0.5 * sigma ** 2) * Δts) / (sigma * math.sqrt(Δts))) -
              phi((math.log(Klower / S0) - (r - 0.5 * sigma ** 2) * Δts) / (sigma * math.sqrt(Δts))))

    sigma_sqrt = sigma * math.sqrt(Δt)
    lam = math.sqrt(((mu / sigma ** 2) ** 2) + 2 * r / sigma ** 2)

    epsilon = 1e-12  # Small value to avoid log(0) and extreme exponentiation

    def integrand_upper(Ss):
        Ss_clamped = max(Ss, epsilon)
        val_exp = -((math.log(Ss_clamped / S0) - (r - 0.5 * sigma ** 2) * Δts) ** 2) / (2 * sigma ** 2 * Δts)
        val = math.exp(val_exp) / (Ss_clamped * sigma * math.sqrt(2 * math.pi * Δts))

        base = Kupper / Ss_clamped
        # Clip base to avoid overflow in powers
        base = min(max(base, epsilon), 1e10)

        term1 = pow(base, mu / sigma ** 2 + lam)
        term2 = pow(base, mu / sigma ** 2 - lam)

        return val * (term1 * phi((math.log(base) / sigma_sqrt) + lam * sigma_sqrt) +
                      term2 * phi((math.log(base) / sigma_sqrt) - 2 * lam * sigma_sqrt))

    def integrand_lower(Ss):
        Ss_clamped = max(Ss, epsilon)
        val_exp = -((math.log(Ss_clamped / S0) - (r - 0.5 * sigma ** 2) * Δts) ** 2) / (2 * sigma ** 2 * Δts)
        val = math.exp(val_exp) / (Ss_clamped * sigma * math.sqrt(2 * math.pi * Δts))

        base = Klower / Ss_clamped
        base = min(max(base, epsilon), 1e10)

        term1 = pow(base, mu / sigma ** 2 + lam)
        term2 = pow(base, mu / sigma ** 2 - lam)

        return val * (term1 * phi(-((math.log(base) / sigma_sqrt) + lam * sigma_sqrt)) +
                      term2 * phi(2 * lam * sigma_sqrt - (math.log(base) / sigma_sqrt)))

    # Use higher limit and increased max subdivisions for integration
    upper_limit = S0 * 10
    ptop, _ = quad(integrand_upper, Kupper, upper_limit, limit=200)

    lower_limit = S0 / 100
    pbottom, _ = quad(integrand_lower, lower_limit, Klower, limit=200)

    phit = pfront + ptop + pbottom
    pmiss = 1 - phit

    chit_coef = chit / (0.5 - chit)
    cmiss_coef = cmiss / (0.5 - cmiss)

    fhit = (1 + chit_coef * phit) / ((1 + chit_coef) * phit) if phit != 0 else 0
    fmiss = (1 + cmiss_coef * pmiss) / ((1 + cmiss_coef) * pmiss) if pmiss != 0 else 0

    if hit:
        f = min(fhit - F, M)
    else:
        f = min(fmiss - F, M)

    payout = f * P
    return float(payout)

# Example usage for standalone testing if needed
if __name__ == "__main__":
    S0 = 112200
    Klower = 112240
    Kupper = 112250
    ts = seconds_to_years(5)
    te = seconds_to_years(6)
    r = 0.01
    sigma = 0.8
    P = 1
    chit = 0.1
    cmiss = 0.1
    F = 0.01
    M = 20000.0
 
    payout_hit = calculate_box_payout(S0, Klower, Kupper, ts, te, r, sigma, P, chit, cmiss, F, M, hit=True)
    payout_miss = calculate_box_payout(S0, Klower, Kupper, ts, te, r, sigma, P, chit, cmiss, F, M, hit=False)

    print(f"Payout if hit: {payout_hit:.6f}")
    print(f"Payout if miss: {payout_miss:.6f}")
