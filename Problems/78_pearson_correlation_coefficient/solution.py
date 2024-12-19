import numpy as np
from typing import List

def pearson_correlation(x : List[float] , y : List[float]) -> float :

    x = np.array(x,dtype = float)
    y = np.array(y,dtype = float)

    if len(x) != len(y):
        raise Exception("Input arrays must have the same length.")

    # Compute the means of x and y   
    x_mean : float = np.mean(x)
    y_mean : float = np.mean(y)

    # Compute covariance --> Cov(X,Y)
    cov_x_y : float = np.sum((x - x_mean) * (y - y_mean)) / len(x)


    # Manually compute standard deviations
    std_x : float =  np.sqrt(np.sum((x - x_mean) **2) / len(x))
    std_y  : float =  np.sqrt(np.sum((y - y_mean) **2) / len(y))

    if std_x == 0 or std_y == 0 :
        raise Exception("Standard deviation of input arrays must not be zero.")


    pearson_corr : float = cov_x_y / (std_x * std_y)
    return pearson_corr



if __name__ == '__main__':
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    pearson_r = pearson_correlation(x, y)
    print("Pearson Correlation Coefficient:", pearson_r)