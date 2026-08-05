import numpy as np
import pandas as pd
from src.distribution_fitting import DistributionFitter

def test_fitter_identifies_normal_distribution():

    np.random.seed(44)
    fake_data = np.random.normal(loc=50, scale=5, size=2000)
    fake_series = pd.Series(fake_data)

    fitter = DistributionFitter(
        data=fake_series,
        candidate_dists=['uniform', 'norm', 'weibull_min']
    )

    fitter.fit()
    best_config = fitter.get_best_config()

    assert best_config['distrbution'] == 'norm'