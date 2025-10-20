# 2D Kalman Filter for Future Position Prediction

This module demonstrates the use of a 2D Kalman Filter to estimate and predict the position of an object in motion, based on noisy measurements. It assumes a constant velocity model and fixed time intervals between observations.

---

## Problem Statement

Given:
- A list of noisy 2D position measurements over time.
- A future time `t_future` for which we want to predict the object's position.

Objective:
- Implement a Kalman Filter to estimate the current state (position and velocity) of the object.
- Predict the object's position at a future time step `t_future`.

Assumptions:
- Constant velocity motion.
- Time interval between measurements is constant (`Δt = 1`).
---

## State Representation

The state vector is defined as:

$$
\mathbf{x} = 
\begin{bmatrix}
x \\\\ v_x \\\\ y \\\\ v_y
\end{bmatrix}
$$

Where:
- $x, y$: position in 2D space  
- $v_x, v_y$: velocity in the x and y directions

---

## Kalman Filter Equations

### 1. Prediction Step

$$
\mathbf{x}_{\text{pred}} = \mathbf{F} \cdot \mathbf{x}
$$

$$
\mathbf{P}_{\text{pred}} = \mathbf{F} \cdot \mathbf{P} \cdot \mathbf{F}^T + \mathbf{Q}
$$

Where:
- $\mathbf{F}$: State transition matrix  
- $\mathbf{P}$: State covariance matrix  
- $\mathbf{Q}$: Process noise covariance matrix  

---

### 2. Update Step

Given a new measurement $\mathbf{z}$:

$$
\mathbf{y} = \mathbf{z} - \mathbf{H} \cdot \mathbf{x}_{\text{pred}}
$$

$$
\mathbf{S} = \mathbf{H} \cdot \mathbf{P}_{\text{pred}} \cdot \mathbf{H}^T + \mathbf{R}
$$

$$
\mathbf{K} = \mathbf{P}_{\text{pred}} \cdot \mathbf{H}^T \cdot \mathbf{S}^{-1}
$$

$$
\mathbf{x} = \mathbf{x}_{\text{pred}} + \mathbf{K} \cdot \mathbf{y}
$$

$$
\mathbf{P} = (\mathbf{I} - \mathbf{K} \cdot \mathbf{H}) \cdot \mathbf{P}_{\text{pred}}
$$

Where:
- $\mathbf{H}$: Measurement matrix  
- $\mathbf{R}$: Measurement noise covariance  
- $\mathbf{K}$: Kalman gain  
- $\mathbf{I}$: Identity matrix  

---

##  Matrices

### State Transition Matrix $\mathbf{F}$ (Assuming $\Delta t = 1$):

$$
\mathbf{F} =
\begin{bmatrix}
1 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 1 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

### Measurement Matrix $\mathbf{H}$:

$$
\mathbf{H} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}
$$

### Process Noise Covariance $\mathbf{Q}$:

$$
\mathbf{Q} = q \cdot \mathbf{I}_4
$$

Where $q$ is a small positive constant (e.g., $1e{-4}$).

### Measurement Noise Covariance $\mathbf{R}$:

$$
\mathbf{R} = r \cdot \mathbf{I}_2
$$

Where $r$ reflects measurement uncertainty.

---

## 🛠️ Initialization

- **Initial State ($\mathbf{x}$)**: Estimated using the first two position measurements.
- **Initial Covariance ($\mathbf{P}$)**: Initialized with large values to represent high uncertainty.
- **Initial Velocity**: Approximated from the difference between the first two position measurements.

---

##  Function Signature

```python
def predict_future_position_using_kalman_filter(
    measurements: List[Tuple[float, float]],
    t_future: float
) -> Tuple[float, float]:
