"""Source code for blairstats.prob"""

from ._utils import *

class Normal:
    def __init__(self, mean:ArrayFloat, variance:ArrayFloat) -> None:
        self.mean = mean
        self.variance = variance

    def pdf(self, x):

    def cdf(self, x):

    def qf(self, x):

    def cf(self, x):

    def mgf(self, x):

    def moment(self, n:int):
