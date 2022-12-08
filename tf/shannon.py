import pandas
import scipy
import scipy.stats
import numpy

df = pandas.read_csv("data\\001.txt", "\n")

[hist, bin_edges] = numpy.histogram(df, bins=100)

print(hist)

probs = [x / sum(hist) for x in hist]

shannon = scipy.stats.entropy(probs)
print(str(shannon))