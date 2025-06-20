import numpy as np


def roc_auc(y_true: list[float], probas: list[float]) -> float:
    """
    Parameters
    ----------
    y_true : list[float]
        True labels
    probas : list[float]
        Output probabilities of our binary classifier
        
    Returns
    -------
    auc : float
        ROC AUC rounded to 5 floating points
    """
    thresh = sorted(probas + [0], reverse=True)
    y_true, probas = np.array(y_true), np.array(probas)

    fpr, tpr = [0], [0]
    auc = 0
    
    for t in thresh:
        y_pred = np.where(probas < t, 0, 1)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = (y_true == 1).sum() - tp

        fp = (y_pred == 1).sum() - tp
        tn = (y_true == 0).sum() - fp

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))
    
        auc += (fpr[-1] - fpr[-2]) * (tpr[-1] + tpr[-2])

    return round(1/2 * auc, 5)


def test_roc_auc():
    # Test 1
    y = [0, 0, 1, 1]
    y_proba = [0.1, 0.4, 0.35, 0.8]
    assert roc_auc(y, y_proba) == .75, 'Test case 1 failed'

    # Test 2
    y = [1, 1, 1, 0, 1, 0, 0, 0, 1, 1]
    y_proba = [
        0.9945685360621648,
        0.9937332904188113,
        0.9958526266087151,
        4.391062222999706e-09,
        0.9959272720187046,
        0.10851446498385146,
        0.001096202856869512,
        4.995474609174945e-06,
        0.9921605697799972,
        0.9826790537446354
    ]
    assert roc_auc(y, y_proba) == 1.0, 'Test case 2 failed'

    # Test 3
    y = [0, 0, 0, 0, 0, 1, 1, 1, 0, 1]
    y_proba = [
        0.8318040739657637,
        0.421445304232661,
        0.003309769194418868,
        0.015529393142531172,
        0.0001635684705459328,
        0.6988867797464966,
        0.9534132112895218,
        0.8471417487716292,
        0.0005832121647006822,
        0.9990059733653113
    ]
    assert roc_auc(y, y_proba) == 0.95833, 'Test case 3 failed'

    # Test 4
    y = [0, 0, 1, 1, 1, 0, 1]
    y_proba = [
        8.99e-1,9.95e-1,5e-3,
        2.3e-4,1e-4,9e-1,2.1e-4
    ]
    assert roc_auc(y, y_proba) == 0.0, 'Test case 4 failed'

    print('All tests passed')


if __name__ == '__main__':
    test_roc_auc()