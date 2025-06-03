import numpy as np
import scipy.stats as st

class DataAnalysis:
    
    def __init__(self, data):
        self.data = data

    def cdf_range(self, xs):
        cdf_values = []

        for x in xs:
            cdf = np.mean(self.data <= x)
            cdf_values.append(cdf)

        return cdf_values
    
    def pdf_range(self, a, b, dx):
        xs = np.arange(a, b, dx)
        pdf_values = st.norm.pdf(xs, loc=np.mean(self.data), scale=np.std(self.data))
        return xs, pdf_values
    
    def std(self):
        return np.std(self.data)
    
    def cdf_value(self, x):
        return st.norm.cdf(x, loc=np.mean(self.data), scale=np.std(self.data))
    
    def pdf_value(self, x):
        return st.norm.pdf(x, loc=np.mean(self.data), scale=np.std(self.data))