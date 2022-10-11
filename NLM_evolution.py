import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from pandas import read_csv

# reading CSV file
data = read_csv("neural_language_models.csv")

# converting column data to list
x = data['year'].tolist()
y = data['training size (GB)'].tolist()

fig, ax1 = plt.subplots(1, 1, figsize=(8,6))

# create trendline on logscale
coefs = np.polyfit(x, np.log(y), 1)
pred_f = coefs[1] + np.multiply(x, coefs[0])

# plot datapoints
ax1.scatter(x, np.exp(np.log(y))) # logs here too!
# plot trendline
ax1.plot(x, np.exp(pred_f), 'k--') # pred_f is already in logs
ax1.set_xlabel('neural language model milestones (years)')
ax1.set_ylabel('dataset size (GiB) (logscale)')
ax1.set_yscale('log')


# create 95% confidence interval
X = sm.add_constant(x)
ols_model = sm.OLS(np.log(y), X)
est = ols_model.fit()
out = est.conf_int(alpha=0.05, cols=None)
pred = est.get_prediction(X).summary_frame()

# plot confidence interval
ax1.plot(x,np.exp(pred['mean_ci_lower']),linestyle='--',color='blue')
ax1.plot(x,np.exp(pred['mean_ci_upper']),linestyle='--',color='blue')
ax1.grid(axis='y')

# trendline equation
textstr = "y=%.6fx+%.6f"%(coefs[0],coefs[1])

# save
fname = TBD
plt.savefig(fname, dpi='figure')
