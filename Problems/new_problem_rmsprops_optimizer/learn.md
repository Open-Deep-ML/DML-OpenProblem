# Learn Section

## RMSprop Optimizer

RMSprop (Root Mean Square Propagation) is an optimization algorithm that adjusts the learning rate dynamically using the moving average of squared gradients.

### Update Rules

1. Compute the moving average of squared gradients:  
   $$ v*t = \beta v*{t-1} + (1 - \beta) \cdot g_t^2 $$

2. Update the parameter using the adjusted learning rate:  
   $$ \theta*t = \theta*{t-1} - \frac{\alpha}{\sqrt{v_t} + \epsilon} \cdot g_t $$

### Hyperparameters

- **α (learning rate)**: Step size for parameter updates.
- **β (decay rate)**: Controls how much past gradients influence the moving average.
- **ϵ (small constant)**: Prevents division by zero for numerical stability.

### Intuition

RMSprop prevents oscillations in gradient updates by scaling gradients based on past values, making it useful for optimizing deep neural networks.
