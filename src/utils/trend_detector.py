# src/utils/trend_detector.py
import numpy as np

def compute_trend_slope(errors, window=20):
    """
    Fit a straight line through the last `window` errors.
    Returns the slope. Positive slope = error is rising.
    """
    if len(errors) < window:
        return 0.0
    recent = errors[-window:]
    x = np.arange(len(recent))
    slope, _ = np.polyfit(x, recent, 1)
    return slope

def early_warning_check(errors, mu, sigma,
                        warn_multiplier=2.0,
                        slope_threshold=0.0005,
                        trend_window=20):
    """
    Two-level alert system.

    Returns a dict with:
      - status:  'NORMAL', 'WARNING', or 'ALARM'
      - reason:  human-readable explanation
      - slope:   current rising slope
    """
    if len(errors) == 0:
        return {"status": "NORMAL", "reason": "No data", "slope": 0.0}

    latest_error = errors[-1]
    alarm_thresh  = mu + 3 * sigma
    warn_thresh   = mu + warn_multiplier * sigma
    slope = compute_trend_slope(errors, window=trend_window)

    if latest_error > alarm_thresh:
        return {
            "status": "ALARM",
            "reason": f"Error {latest_error:.5f} crossed alarm threshold {alarm_thresh:.5f}",
            "slope": slope
        }
    elif latest_error > warn_thresh or slope > slope_threshold:
        return {
            "status": "WARNING",
            "reason": (f"Error is rising (slope={slope:.6f}) "
                       f"or above warning level ({warn_thresh:.5f})"),
            "slope": slope
        }
    else:
        return {"status": "NORMAL", "reason": "Operating normally", "slope": slope}