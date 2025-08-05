import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from src.confidence_interaval import ConfidenceInterval
from src.data_analysis import DataAnalysis

class Plotter:

    def __init__(self, data):
        self.data = data

    def __get_bins__(self, dx):
        mean = np.mean(self.data)
        std = np.std(self.data)
        a = mean - 3 * std
        b = mean + 3 * std
        n = round((b - a) / dx)
        xs = np.linspace(a, b, n)
        return xs
    
    def plot_hist(self):
        fig, ax = plt.subplots(1, 1)
        ax.hist(x=self.data, bins=20, histtype='step', linewidth=3.5, density=False)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.axvline(np.mean(self.data), color='black', linestyle='dashed', linewidth=4)
        ax.set_xlabel("X", size=20)
        ax.set_ylabel("P(X = x)", size=20)
        ax.set_title("Histogram", size=20)
        plt.show()

    def plot_cdf(self, dx=1e-5):
        data_analysis = DataAnalysis(self.data)
        xs = self.__get_bins__(dx)
        fig, ax = plt.subplots(1, 1)
        ax.plot(xs, data_analysis.cdf_range(xs), label='Fe')
        ax.plot(xs, st.norm.cdf(xs, loc=np.mean(self.data), scale=np.std(self.data), label='F'))
        ax.axvline(np.mean(self.data), color='k', linestyle='dashed', linewidth=2)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel("X", size=20)
        ax.set_ylabel("P(X <= x)", size=20)
        plt.title("Cumulative Distribution Function (CDF)", size=20)
        ax.legend()
        plt.show()

    def plot_pdf(self, dx):
        data_analysis = DataAnalysis(self.data)
        mean = np.mean(self.data)
        std = np.std(self.data)
        a = mean - 3 * std
        b = mean + 3 * std
        xs, fe = data_analysis.pdf_range(a, b, dx)
        fig, ax = plt.subplots(1, 1)
        ax.plot(xs, fe, label='Fe')
        ax.plot(xs, st.norm.pdf(xs, loc=mean, scale=std), label='F')
        ax.axvline(np.mean(self.data), color='k', linestyle='dashed', linewidth=2)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel("X", size=20)
        ax.set_ylabel("P(X = x)", size=20)
        plt.title("Probability Density Function (PDF)", size=20)
        ax.legend()
        plt.show()

    def plot_confidence_interval(self, alpha):
        confidence_interval = ConfidenceInterval(self.data, alpha,)
        mean = confidence_interval.mean
        lower_bound = confidence_interval.lower_bound()
        upper_bound = confidence_interval.upper_bound()

        fig, ax = plt.subplots(1, 1)
        ax.plot(1, mean, marker='o', markersize=10)
        ax.vlines(x=1, ymin=lower_bound, ymax=upper_bound, linewidth=2)
        ax.hlines(y=lower_bound, xmin=1-0.1, xmax=1+0.1, linewidth=2)
        ax.hlines(y=upper_bound, xmin=1-0.1, xmax=1+0.1, linewidth=2)
        ax.set_xticks([1])
        ax.set_xticklabels([""])
        ax.tick_params(axis='both', which='major', labelsize=15)

        ax.set_xlim(0,2)
        ax.set_ylabel("X", size=20)
        plt.title(f"Confidence Interval (alpha={alpha})", size=20)
        plt.show()