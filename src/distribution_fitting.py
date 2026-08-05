import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any 

class DistributionFitter:
    """
    Fits emprical data to candiate SciPy continuous probability distrbutions, 
    ranks them by the Kolmogorov-Smirnov (KS) test, and generates visual plots.
    """
    def __init__(self, data: pd.Series, data_name: str = "Emperical Data", candidate_dists: List[str] = None):
        self.data = data.dropna().values.astype(np.float64)
        self.data_name = data_name

        # If empty is provided in YAML, fall back to 100+ distributions search.
        self.candidate_dists = candidate_dists if candidate_dists else _distn_names

        self.results_df = None
        self.best_dist_name = None
        self.best_params = None

    def fit(self) -> pd.DataFrame:
        """ Tests the dataset against every continuous OpenTURNS distribution factory """
        results = []

        print(f"Evaluating {len(self.candidate_dists)} distributions for '{self.data_name}'...")

        for dist_name in self.candidate_dists:
            try:
                dist_getattr = getattr(st, dist_name) # Get the SciPy distribution
                params = dist_getattr.fit(self.data) # Fit to data and get its parameters

                # KS test for goodness-of-fit. Lower KS means better fit.
                ks_stat, ks_pvalue = st.kstest(self.data, dist_name, args=params)

                results.append({
                    'distribution': dist_name,
                    'ks_statistic': ks_stat,
                    'p_value': ks_pvalue,
                    'parameters': params 
                })
            except Exception:
                continue # Skip distributions that fail to converse

        # Send error message if there's no fit to data
        if not results:
            raise ValueError(f"SciPy failed to fit any distribution for '{self.data_name}'. Check input data.")

        # Convert to DataFrame and sort by best fit (lowest KS statistic)
        self.results_df = pd.DataFrame(results).sort_values(by='ks_statistic', ascending=True).reset_index(drop=True)

        # Get the top-ranking results
        top_row = self.results_df.iloc[0]
        self.best_dist_name = top_row['distribution']
        self.best_params = top_row['parameters']

        return self.results_df
    
    def get_best_config(self) -> Dict[str, Any]:
        """ Returns a summary of the best fit. """
        if self.best_dist_name is None:
            self.fit()

        return {
            'distribution': self.best_dist_name,
            'parameters': self.best_params, 
            'ks_statistic': float(self.results_df.iloc[0]['ks_statistic'])
        }

    def plot_pdf_overlay(self, top_n: int = 3, save_path: str = None):
        """ Plot the empirical histogram overlaid with the top N fitted OpenTURNS PDFs """
        if self.results_df is None:
            self.fit()

        plt.figure(figsize=(10, 6))
        sns.histplot(self.data, stat="density", bins=50, color="lighgrey", label=f"Emperical ({self.data_name})")

        x_space = np.linspace(min(self.data), max(self.data), 1000)
        colors = ['red', 'blue', 'green', 'purple', 'organge']

        for i in range(min(top_n, len(self.results_df))):
            row = self.results_df.iloc[i]
            dist_model = getattr(st, row['distribution'])

            pdf = dist_model.pdf(x_space, *row['parameter'])

            plt.plot(x_space, pdf, label=f"#{i+1}: {row['distribution']} (KS: {row['ks_statistic']:.3f})",
                     color=colors[i % len(colors)], lw=2)

        plt.title(f"Distribution Fit Overlay - {self.data_name}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_qq_diagnostics(self, save_path: str = None):

        if self.best_dist_model is None:
            self.fit()

        fig, ax = plt.subplot(figsize=(6, 6))
        st.probplot(self.data, dist=self.best_dist_name, sparams=self.best_params, plot=ax)
        ax.set_title(f"Q-Q Plot: Best Fit ({self.best_dist_name}) vs. {self.data_name}")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
