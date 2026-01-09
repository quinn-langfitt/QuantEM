"""Pauli Check Extrapolation (PCE) for error mitigation.

Extrapolates expectation values from circuits with varying numbers of
Pauli checks to the maximum check limit.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Tuple

__all__ = [
    "analyze_pce_results",
    "extrapolate_expectations",
]


def _fit_linear(
    num_checks: np.ndarray,
    exp_values: np.ndarray,
    n_max: int
) -> Tuple[float, Dict[str, float]]:
    """Fit linear model to PCE data.

    Mathematical Model:
        E(n) = α + β·n

    where:
        - n: number of Pauli checks
        - E(n): expectation value with n checks
        - α, β: fitted parameters

    Args:
        num_checks: Array of check counts
        exp_values: Corresponding expectation values
        n_max: Target extrapolation point

    Returns:
        Tuple of (extrapolated_value, params_dict)
    """
    coeffs = np.polyfit(num_checks, exp_values, deg=1)
    beta, alpha = coeffs[0], coeffs[1]
    extrapolated = alpha + beta * n_max
    params = {"alpha": float(alpha), "beta": float(beta)}
    return extrapolated, params


def _fit_exponential(
    num_checks: np.ndarray,
    exp_values: np.ndarray,
    n_max: int
) -> Tuple[float, Dict[str, float]]:
    """Fit exponential model to PCE data.

    Mathematical Model:
        E(n) = A·B^n + C

    where:
        - n: number of Pauli checks
        - E(n): expectation value with n checks
        - C: asymptotic limit as n → ∞ (for |B| < 1)
        - A, B: fitted parameters

    Args:
        num_checks: Array of check counts
        exp_values: Corresponding expectation values
        n_max: Target extrapolation point

    Returns:
        Tuple of (extrapolated_value, params_dict)
    """
    def exp_func(n, a, b, c):
        return a * (b ** n) + c

    c_init = exp_values[0]
    b_init = 0.9
    a_init = (exp_values[-1] - exp_values[0]) / (b_init ** num_checks[-1] - 1)

    popt, _ = curve_fit(
        exp_func,
        num_checks,
        exp_values,
        p0=[a_init, b_init, c_init],
        bounds=([-np.inf, 0.6, -np.inf], [np.inf, 1.2, np.inf]),
        maxfev=10000
    )

    a, b, c = popt
    extrapolated = a * (b ** n_max) + c
    params = {"a": float(a), "b": float(b), "c": float(c)}
    return extrapolated, params


def _fit_polynomial(
    num_checks: np.ndarray,
    exp_values: np.ndarray,
    n_max: int
) -> Tuple[float, Dict[str, float]]:
    """Fit quadratic polynomial model to PCE data.

    Mathematical Model:
        E(n) = a·n² + b·n + c

    where:
        - n: number of Pauli checks
        - E(n): expectation value with n checks
        - a, b, c: fitted parameters

    Args:
        num_checks: Array of check counts
        exp_values: Corresponding expectation values
        n_max: Target extrapolation point

    Returns:
        Tuple of (extrapolated_value, params_dict)
    """
    coeffs = np.polyfit(num_checks, exp_values, deg=2)
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    extrapolated = a * (n_max ** 2) + b * n_max + c
    params = {"a": float(a), "b": float(b), "c": float(c)}
    return extrapolated, params


_MODELS = {
    "linear": _fit_linear,
    "exponential": _fit_exponential,
    "polynomial": _fit_polynomial,
}


def extrapolate_expectations(
    expectations: Dict[int, float],
    n_max: int,
    model: str = "exponential"
) -> Dict:
    """Extrapolate expectation value to maximum check limit.

    Args:
        expectations: Dictionary mapping num_checks -> expectation_value
                     e.g., {0: 0.5, 1: 0.7, 2: 0.85, 3: 0.92}
        n_max: Maximum number of checks to extrapolate to
        model: Extrapolation model - "linear", "exponential", or "polynomial"

    Returns:
        Dictionary with:
            - extrapolated_value: Predicted expectation at n_max
            - model: Model used
            - params: Fitted model parameters

    Example:
        >>> expectations = {0: 0.5, 1: 0.7, 2: 0.85, 3: 0.92}
        >>> result = extrapolate(expectations, n_max=4, model="exponential")
        >>> print(result['extrapolated_value'])
        0.97
    """
    if len(expectations) < 2:
        raise ValueError("Need at least 2 data points for extrapolation")

    if model in ["exponential", "polynomial"] and len(expectations) < 3:
        raise ValueError(f"{model.capitalize()} model requires at least 3 data points")

    num_checks = np.array(sorted(expectations.keys()))
    exp_values = np.array([expectations[n] for n in num_checks])

    if model not in _MODELS:
        available = ", ".join(_MODELS.keys())
        raise ValueError(f"Unknown model '{model}'. Available: {available}")

    extrapolated, params = _MODELS[model](num_checks, exp_values, n_max)

    return {
        "extrapolated_value": float(extrapolated),
        "model": model,
        "params": params,
    }


def analyze_pce_results(
    results: Dict[int, float],
    n_max: int = 0,
    model: str = "exponential"
) -> Dict:
    """Analyze execution results from a PCE suite.

    This function takes the expectation values obtained from running the circuits
    generated by `compile_pce` and extrapolates them to estimate the noiseless limit.

    For the exponential model E(n) = A + B·C^n, the asymptotic value is A (if |C| < 1).
    Set n_max to a large value to approximate the asymptotic limit, or use a specific
    target check count for intermediate extrapolation.

    Args:
        results: Dictionary mapping num_checks -> expectation_value.
                 Example: {0: 0.8, 2: 0.9, 4: 0.95}
        n_max: Target extrapolation point (use large value for asymptotic limit).
               Default is 0 (zero-noise extrapolation).
        model: Fitting model ("exponential", "linear", "polynomial")

    Returns:
        Dictionary with extrapolated_value, model, and fitted params.
    """
    if not results:
        raise ValueError("No results provided for analysis")

    # Sort checks to ensure order
    sorted_checks = sorted(results.keys())
    
    # Simple validation
    if len(sorted_checks) < 2:
        # Not enough points, return best effort (e.g. the one with most checks)
        best_check = sorted_checks[-1]
        return {
            "extrapolated_value": results[best_check],
            "model": "none (too few points)",
            "params": {},
            "warning": "Not enough data points for extrapolation"
        }

    return extrapolate_expectations(results, n_max, model)
