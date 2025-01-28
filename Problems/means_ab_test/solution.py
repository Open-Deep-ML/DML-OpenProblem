import scipy.stats as sts
import numpy as np


def ttest_ind(a:np.ndarray, b:np.ndarray, equal_var:bool=False):
    """
    Function for performing an AB means comparison test. NOTE: You are
    only allowed to call .cdf(), .ppf() or .pmf() methods from scipy! 

    Parameters
    ----------
    a : np.ndarray
        Treatment group
    b : np.ndarray
        Control group
    equal_var : bool
        We either apply equal variance test or Welch's T test
        
    Returns
    -------
    t_stat : float
        Student's t statistic
    p_val : float
        P value
    dof : float
        Degrees of freedom
    """
    n, m = a.shape[0], b.shape[0]
    se1, se2 = a.std() / n ** .5, b.std() / m ** .5
    t_stat = (a.mean() - b.mean()) / (se1**2 + se2**2)**.5
    if equal_var:
        dof = n + m - 2
    else:
        dof = (se1**2 + se2**2)**2 / (se1**4 / (n-1) + se2**4 / (m-1))
    p_val = 2*sts.t.cdf(t_stat,df=dof)
    return round(t_stat, 4), round(p_val, 4), round(dof, 4)


def test_ttest_ind():
    # Test 1
    a, b = np.array([ 0.244, -0.292,  0.833,  2.09 ,  
                     0.605,  2.807,  0.96 ,  1.145, 0.789,  1.563]), \
           np.array([ 4.637,  0.622,  8.212,  1.796,  1.053, -2.633,  
                     4.456, -1.515,  1.921,  5.349])
    e_tstat, e_pval, e_dof = -1.2828, 0.2276, 10.3175
    tstat, pval, dof = ttest_ind(a, b, False)
    assert e_tstat == tstat and e_pval == pval and e_dof == dof, 'Test case 1 failed'

    # Test 2
    a, b = np.array([-3.452,  3.607,  2.018,  2.29 , -2.559,  4.946, -2.071, -2.182,
            -3.241,  0.393,  4.221, -2.234,  2.481,  8.247,  4.546]), \
           np.array([-1.912,  9.69 ,  3.588, -0.242,  5.202,  3.174, -3.98 ,  3.67 ,
            -4.944,  3.446,  6.102,  0.326, -2.531,  2.794, -5.508])
    e_tstat, e_pval, e_dof = -0.0875, 0.9309, 28
    tstat, pval, dof = ttest_ind(a, b, True)
    assert e_tstat == tstat and e_pval == pval and e_dof == dof, 'Test case 2 failed'

    print('All tests passed')


if __name__ == '__main__':
    test_ttest_ind()