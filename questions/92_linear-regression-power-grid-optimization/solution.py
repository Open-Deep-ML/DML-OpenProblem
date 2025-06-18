import math

PI = 3.14159

def power_grid_forecast(consumption_data):
    # consumption_data: list of 10 daily consumption values
    # days: 1 through 10
    days = list(range(1, 11))
    n = len(days)

    # 1) Remove daily fluctuation f_i = 10 * sin(2π * i / 10)
    detrended = []
    for i, cons in zip(days, consumption_data):
        fluctuation_i = 10 * math.sin((2 * PI * i) / 10)
        detrended_value = cons - fluctuation_i
        detrended.append(detrended_value)

    # 2) Perform linear regression on the detrended data
    sum_x = sum(days)
    sum_y = sum(detrended)
    sum_xy = sum(x * y for x, y in zip(days, detrended))
    sum_x2 = sum(x**2 for x in days)

    # slope (m) and intercept (b) for y = m*x + b
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    b = (sum_y - m * sum_x) / n

    # 3) Predict day 15's base usage
    day_15_base = m * 15 + b

    # 4) Add back daily fluctuation for day 15
    day_15_fluctuation = 10 * math.sin((2 * PI * 15) / 10)
    day_15_prediction = day_15_base + day_15_fluctuation

    # 5) Round and add 5% safety margin
    day_15_rounded = round(day_15_prediction)
    final_15 = math.ceil(day_15_rounded * 1.05)

    return final_15
