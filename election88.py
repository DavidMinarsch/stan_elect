"""
import os
os.chdir('/Users/davidminarsch/Desktop/PythonMLM/Election_Example')
exec(open("election88.py").read())
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_context('notebook')
import pystan

# Read the data & define variables
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/election88
# Data are at /Users/davidminarsch/Desktop/PythonMLM/ARM_Data/election88 and /Users/davidminarsch/Desktop/PythonMLM/Election_Example
# Set up the data for the election88 example

# Load in data for region indicators
# state_abbr: abbreviations of state names
# regions:  1=northeast, 2=south, 3=north central, 4=west, 5=d.c.
# not_dc: indicator variable which is 1 for non_dc states

state_info = pd.read_csv('/Users/davidminarsch/Desktop/PythonMLM/Election_Example/state.csv')
state_info = state_info.rename(columns={'Unnamed: 0': 'state'})
#state_info = state_info.drop(state_info.columns[[0]], axis=1)

# Load in data from the CBS polls in 1988
# org: organisation which collected the poll
# year: year id
# survey: survey id
# bush: indicator (=1) for support of bush
# state: state id
# edu: categorical variable indicating level of education
# age: categorical variable indicating age
# female: indicator (=1) for female
# black: indicator (=1) for black
# weight: sample weight
polls = pd.read_csv('/Users/davidminarsch/Desktop/PythonMLM/Election_Example/polls.csv')
polls = polls.drop(polls.columns[[0]], axis=1)

# also include a measure of previous vote as a state-level predictor
presvote = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/ARM_Data/election88/presvote.csv")
presvote = presvote.drop(presvote.columns[[0]], axis=1)
#g76_84pr: state average in previous election
#stnum2: state id
candidate_effects = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/ARM_Data/election88/candidate_effects.csv")
candidate_effects = candidate_effects.drop(candidate_effects.columns[[0]], axis=1)
v_prev = presvote.g76_84pr
presvote = presvote.rename(columns={'g76_84pr': 'v_prev', 'stnum2': 'state'})
not_dc = [x for x in range(0,51)] 
del not_dc[8] #we remove 8 since this is equivalent to position 9 (state id 9)
v_prev[not_dc] += (candidate_effects.X76 + candidate_effects.X80 + candidate_effects.X84) / 3.0
presvote.v_prev = v_prev

#merge all three dataframes into one:
polls = pd.merge(polls, state_info, on='state', how='left')
polls = pd.merge(polls, presvote, on='state', how='left')

#select subset of polls:
polls_subset = polls.loc[polls['survey'] == '9158']
#drop nan in polls_subset.bush
polls_subset_no_nan = polls_subset[polls_subset.bush.notnull()]

# define other data summaries
y = polls_subset.bush                  	# 1 if support bush, 0 if support dukakis
y_w = polls_subset.drop(polls_subset.columns[[0, 1, 2, 4, 5, 6, 7, 8]], axis=1)  #weight and bush only
y_w_s = polls_subset.drop(polls_subset.columns[[0, 1, 2, 5, 6, 7, 8]], axis=1)  #weight and bush and state only
n = len(y)             					# of survey respondents
n_age = max(polls_subset.age)          	# of age categories
n_edu = max(polls_subset.edu)          	# of education categories
n_state = max(polls_subset.state)      	# of states
n_region = max(state_info.region)    	# of regions

# compute unweighted and weighted averages for the U.S.
y_no_nan = [x for x in y if str(x).lower() != 'nan']    # remove the undecideds
y_w_no_nan = y_w[y_w.bush.notnull()]					# remove the undecideds
y_w_s_no_nan = y_w_s[y_w_s.bush.notnull()]				# remove the undecideds
mean_nat_raw = round(np.mean(y_no_nan),3) 				# national mean of raw data
mean_nat_wgt_raw = round(y_w_no_nan.bush.dot(y_w_no_nan.weight) / sum(y_w_no_nan.weight),3) 			# national weighted mean of raw data

# compute weighted averages for the states
mean_state_wgt_raw = dict(zip(state_info.state, [0 for i in range(n_state)]))
for key in mean_state_wgt_raw:
	y_w_select = y_w_s_no_nan[y_w_s_no_nan.state == key]
	s = sum(y_w_select.weight)
	if s == 0:
		mean_state_wgt_raw[key] = 'nan'
	else:
		mean_state_wgt_raw[key] = round(y_w_select.bush.dot(y_w_select.weight) / sum(y_w_select.weight),3)

# load in 1988 election data as a validation check
election88 = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/ARM_Data/election88/election88.csv")
election88 = election88.drop(election88.columns[[0]], axis=1)
# stnum: state id
# st: state abbreviation
# electionresult: is the outcome of the election
# samplesize: 
# raking:
# merge_:

# load in 1988 census data
census88 = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/ARM_Data/election88/census88.csv")
census88 = census88.drop(census88.columns[[0]], axis=1)
census88 = pd.merge(census88, state_info, on='state', how='left')
census88 = pd.merge(census88, presvote, on='state', how='left')
# edu: categorical variable indicating level of education
# age: categorical variable indicating age
# female: indicator (=1) for female
# black: indicator (=1) for black
# N: size of population in this cell


## Multilevel logistic regression
multilevel_logistic = """
data {
  int<lower=0> N;
  int<lower=0> n_state;
  vector<lower=0,upper=1>[N] black;
  vector<lower=0,upper=1>[N] female;
  int<lower=1,upper=n_state> state[N];
  int<lower=0,upper=1> y[N];
} 
parameters {
  vector[n_state] a;
  vector[2] b;
  real<lower=0,upper=100> sigma_a;
  real mu_a;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = b[1] * black[i] + b[2] * female[i] + a[state[i]];
} 
model {
  mu_a ~ normal(0, 1);
  a ~ normal (mu_a, sigma_a);
  b ~ normal (0, 100);
  y ~ bernoulli_logit(y_hat);
}
"""
multilevel_logistic_data_dict = {'N': len(polls_subset_no_nan.bush), 'n_state': n_state, 'black': polls_subset_no_nan.black, 'female': polls_subset_no_nan.female,
	'state': polls_subset_no_nan.state , 'y': polls_subset_no_nan.bush.astype(int)}
multilevel_logistic_fit = pystan.stan(model_code=multilevel_logistic, data=multilevel_logistic_data_dict, iter=1000, chains=2)
a_sample = pd.DataFrame(multilevel_logistic_fit['a'])
plt.figure(figsize=(16, 6))
sns.boxplot(data=a_sample, whis=np.inf, color="c")
plt.savefig('StateIntercepts.png')
plt.show()
print(round(a_sample.mean(),1))
print(round(a_sample.std(),1))
b_sample = pd.DataFrame(multilevel_logistic_fit['b'])
plt.figure(figsize=(16, 6))
sns.boxplot(data=b_sample, whis=np.inf, color="c")
plt.savefig('IndiciatorSlopes.png')
plt.show()
print(round(b_sample.mean(),1))
print(round(b_sample.std(),1))

## A fuller model
polls_subset_no_nan = polls_subset_no_nan.dropna()
# set up the predictors
#polls_subset_no_nan['age_edu'] = n_edu * (polls_subset_no_nan.age - 1) + polls_subset_no_nan.edu
polls_subset_no_nan = polls_subset_no_nan.assign(age_edu=n_edu * (polls_subset_no_nan.age - 1) + polls_subset_no_nan.edu)
n_age_edu = max(polls_subset_no_nan.age_edu)          	# of age x edu categories

full_model = """
data {
  int<lower=0> N; 
  int<lower=0> n_age;
  int<lower=0> n_age_edu;  
  int<lower=0> n_edu; 
  int<lower=0> n_region; 
  int<lower=0> n_state; 
  int<lower=0,upper=n_age> age[N];
  int<lower=0,upper=n_age_edu> age_edu[N];
  vector<lower=0,upper=1>[N] black;
  int<lower=0,upper=n_edu> edu[N];
  vector<lower=0,upper=1>[N] female;
  int<lower=0,upper=n_region> region[N];
  int<lower=0,upper=n_state> state[N];
  vector[N] v_prev;
  int<lower=0,upper=1> y[N];
} 
parameters {
  vector[n_age] a;
  vector[n_edu] b;
  vector[n_age_edu] c;
  vector[n_state] d;
  vector[n_region] e;
  vector[5] beta;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_b;
  real<lower=0,upper=100> sigma_c;
  real<lower=0,upper=100> sigma_d;
  real<lower=0,upper=100> sigma_e;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = beta[1] + beta[2] * black[i] + beta[3] * female[i]
      + beta[4] * v_prev[i] + beta[5] * female[i] * black[i]
      + a[age[i]] + b[edu[i]] + c[age_edu[i]] + d[state[i]] + e[region[i]];
} 
model {
  a ~ normal (0, sigma_a);
  b ~ normal (0, sigma_b);
  c ~ normal (0, sigma_c);
  d ~ normal (0, sigma_d);
  e ~ normal (0, sigma_e);
  beta ~ normal(0, 100);
  y ~ bernoulli_logit(y_hat);
}
"""
full_model_data_dict = {'N': len(polls_subset_no_nan.bush), 'n_age': n_age, 'n_age_edu': n_age_edu,
	'n_edu': n_edu, 'n_region': n_region, 'n_state': n_state, 'age': polls_subset_no_nan.age,
	'age_edu': polls_subset_no_nan.age_edu, 'black': polls_subset_no_nan.black,
	'edu': polls_subset_no_nan.edu, 'female': polls_subset_no_nan.female,
	'region': polls_subset_no_nan.region, 'state': polls_subset_no_nan.state ,
	'v_prev': polls_subset_no_nan.v_prev, 'y': polls_subset_no_nan.bush.astype(int)}
n_iter = 5000
full_model_fit = pystan.stan(model_code=full_model, data=full_model_data_dict, iter=n_iter, chains=2)

#Output:
param_fm  = full_model_fit.extract(permuted=True)
beta_fm = pd.DataFrame(param_fm['beta'])
beta_fm.rename(columns=lambda x: 'beta_' + str(x+1), inplace=True)
beta_fm_m = beta_fm.mean()
a_fm = pd.DataFrame(param_fm['a'])
a_fm.rename(columns=lambda x: 'a_' + str(x), inplace=True)
a_fm_m = a_fm.mean()
b_fm = pd.DataFrame(param_fm['b'])
b_fm.rename(columns=lambda x: 'b_' + str(x), inplace=True)
b_fm_m = b_fm.mean()
c_fm = pd.DataFrame(param_fm['c'])
c_fm.rename(columns=lambda x: 'c_' + str(x), inplace=True)
c_fm_m = c_fm.mean()
d_fm = pd.DataFrame(param_fm['d'])
d_fm.rename(columns=lambda x: 'd_' + str(x), inplace=True)
d_fm_m = d_fm.mean()
e_fm = pd.DataFrame(param_fm['e'])
e_fm.rename(columns=lambda x: 'e_' + str(x), inplace=True)
e_fm_m = e_fm.mean()
param_fm_dict = pd.concat([a_fm, b_fm, c_fm, d_fm, e_fm, beta_fm], axis=1)
sample_1 = pd.concat([a_fm, b_fm, c_fm, d_fm, e_fm, beta_fm[beta_fm.columns[[0,1,2,4]]]], axis=1)
print(round(a_fm_m,1))
print(round(b_fm_m,1))
print(round(c_fm_m,1))
print(round(d_fm_m,1))
print(round(e_fm_m,1))
print(round(beta_fm_m,1))
plt.figure(figsize=(16, 6))
sns.boxplot(data=sample_1, whis=np.inf, color="c")
plt.savefig('Demographics_Full.png')
plt.show()
# The box shows the quartiles of the dataset while the whiskers extend to show the rest of the
# distribution, except for points that are determined to be “outliers” using a method that is a
# function of the inter-quartile range.

# create linear predictors
n = len(polls_subset_no_nan.bush)
lin_pred = np.full((1,n), np.nan)
polls_subset_no_nan = polls_subset_no_nan.reset_index(drop=True)
for i in range(0,n):
	lin_pred[0][i] = np.mean(beta_fm.ix[:,0] + beta_fm.ix[:,1]*polls_subset_no_nan.black[i] + 
		beta_fm.ix[:,2]*polls_subset_no_nan.female[i] + beta_fm.ix[:,4]*polls_subset_no_nan.female[i]*polls_subset_no_nan.black[i]
		+ a_fm.ix[:,polls_subset_no_nan.age[i]-1] + b_fm.ix[:,polls_subset_no_nan.edu[i]-1] +
		c_fm.ix[:,polls_subset_no_nan.age_edu[i]-1])

"""Plot sample states (Not correct!)"""
sample_states = np.array([2,3,4,8,6,7,5,9]) #state_info.state_abbr
iterator = np.array(range(0,8))
fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex =True)
axes = axes.ravel()
for i, c in zip(sample_states,iterator):
    xvals = np.linspace(np.amin(lin_pred), np.amax(lin_pred))
    # Median estimate
    axes[c].plot(xvals, sp.special.expit((d_fm[[i-1]]).median()[0] + xvals), 'k:')
    axes[c].set_xlim(np.amin(lin_pred),np.amax(lin_pred))
    axes[c].set_ylim(0, 1)
    axes[c].set_title(state_info.state_abbr[i-1])
    axes[c].set_ylabel('Pr (support Bush)')
    axes[c].set_xlabel('linear predictor')
plt.savefig('Sample_States.png')
plt.show()

## Using the model inferences to estimate avg opinion for each state
# construct the n.sims x 3264 matrix
import scipy as sp
census88 = census88.assign(age_edu=n_edu * (census88.age - 1) + census88.edu)
L = census88.shape[0]
y_pred = np.full((n_iter,L), np.nan)
for l in range(0, L):
    y_pred[:,l] = sp.special.expit(beta_fm.ix[:,0] + beta_fm.ix[:,1]*census88.black[l] + 
    beta_fm.ix[:,2]*census88.female[l] + beta_fm.ix[:,3]*census88.v_prev[l] +
    beta_fm.ix[:,4]*census88.female[l]*census88.black[l] +
    a_fm.ix[:,census88.age[l]-1] + b_fm.ix[:,census88.edu[l]-1] +
    c_fm.ix[:,census88.age_edu[l]-1] + d_fm.ix[:,census88.state[l]-1] +
    e_fm.ix[:,census88.region[l]-1])

# average over strata within each state
y_pred_state = np.full((n_iter,n_state), np.nan)
for j in range(1,n_state+1):
    sel = [s for s in range(L) if census88.state[s] ==  j]
    y_pred_state[:,j-1] = np.divide((np.dot(y_pred[:,sel],(census88[census88.state == j]).N)),sum((census88[census88.state == j]).N))

# average over strata within each state/get point prediction and 50% interval:
y_pred_state = pd.DataFrame(y_pred_state)
state_pred = pd.DataFrame(np.full((n_state,3), np.nan))
state_pred[0] = y_pred_state.quantile(q=0.25, axis=0, numeric_only=True, interpolation='linear')
state_pred[1] = y_pred_state.quantile(q=0.5, axis=0, numeric_only=True, interpolation='linear')
state_pred[2] = y_pred_state.quantile(q=0.75, axis=0, numeric_only=True, interpolation='linear')

plt.figure(figsize=(16, 6))
sns.boxplot(data=y_pred_state, whis=np.inf, color="c")
plt.savefig('Estimates_state.png')
plt.show()