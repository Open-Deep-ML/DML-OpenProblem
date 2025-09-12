Given a list of 2D noisy position measurements of an object moving in a plane, and a target future timestep `t_future`, estimate the position of the object at that time using a Kalman Filter.

Assume:
- The object moves with constant velocity.
- Equal time intervals between each measurement.

Function signature:
```python
def predict_future_position_using_kalman_filter(measurements: List[Tuple[float, float]], t_future: float) -> Tuple[float, float]:
