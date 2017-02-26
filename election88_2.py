"""
import os
os.chdir('/Users/davidminarsch/Desktop/PythonMLM/Election_Example')
exec(open("election88_2.py").read())
"""
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_context('notebook')
import pystan
from collections import OrderedDict
import pickle
from pystan import StanModel

"""Multilevel Modeling with Poststratification (MRP)"""
# Use multilevel regression to model individual survey responses as a function of demographic and geographic
# predictors, partially pooling respondents across states/regions to an extent determined by the data.
# The final step is poststratification.

# Read the data & define variables
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/election88
# Data are at /Users/davidminarsch/Desktop/PythonMLM/ARM_Data/election88 and /Users/davidminarsch/Desktop/PythonMLM/Election_Example
# Set up the data for the election88 example

"""Step 1: gather national opinion polls (they need to include respondent information down to the level of disaggregation
the analysis is targetting) """
# Load in data from the CBS polls with the following covariates (individual level):
# - org: organisation which collected the poll
# - year: year id
# - survey: survey id
# - bush: indicator (=1) for support of bush
# - state: state id
# - edu: categorical variable indicating level of education
# - age: categorical variable indicating age
# - female: indicator (=1) for female
# - black: indicator (=1) for black
# - weight: sample weight
polls = pd.read_csv('/Users/davidminarsch/Desktop/PythonMLM/Election_Example/polls.csv')
polls = polls.drop(polls.columns[[0]], axis=1)

"""Step 2: create a separate dataset of state-level predictors """
# Load in data for region indicators (state level). The variables are:
# - state_abbr: abbreviations of state names
# - regions:  1=northeast, 2=south, 3=north central, 4=west, 5=d.c.
# - not_dc: indicator variable which is 1 for non_dc states
state_info = pd.read_csv('/Users/davidminarsch/Desktop/PythonMLM/Election_Example/state.csv')
state_info = state_info.rename(columns={'Unnamed: 0': 'state'})

# Include a measure of previous vote as a state-level predictor. The variables are:
# - g76_84pr: state average in previous election
# - stnum2: state id
presvote = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/ARM_Data/election88/presvote.csv")
presvote = presvote.drop(presvote.columns[[0]], axis=1)
presvote = presvote.rename(columns={'g76_84pr': 'v_prev', 'stnum2': 'state'})

# Include a measure of candidate effects as a state-level predictor and add empty row for DC.
candidate_effects = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/ARM_Data/election88/candidate_effects.csv")
candidate_effects = candidate_effects.drop(candidate_effects.columns[[0]], axis=1)
candidate_effects = candidate_effects.rename(columns={'state': 'state_abbr'})
candidate_effects.loc[:,'candidate_effects_weighted'] = (candidate_effects.loc[:,'X76'] + candidate_effects.loc[:,'X80'] + candidate_effects.loc[:,'X84']) / 3.0
candidate_effects_1 = candidate_effects.iloc[:9]
candidate_effects = pd.concat([candidate_effects_1,candidate_effects.ix[8:]]).reset_index(drop=True)
candidate_effects.iloc[8] = 0
candidate_effects = candidate_effects.set_value(8, 'state_abbr', 'DC')
presvote.loc[:,'v_prev'] += candidate_effects.loc[:,'candidate_effects_weighted']

# Merge all three dataframes into one:
polls = pd.merge(polls, state_info, on='state', how='left')
polls = pd.merge(polls, presvote, on='state', how='left')

# Select subset of polls:
polls_subset = polls.loc[polls['survey'] == '9158']

# Change female to sex and black to race:
polls_subset.loc[:,'sex'] = polls_subset.loc[:,'female'] + 1
polls_subset.loc[:,'race'] = polls_subset.loc[:,'black'] + 1

# Drop unnessary columns: 
polls_subset = polls_subset.drop(['org', 'year', 'survey', 'region', 'not_dc', 'state_abbr', 'weight', 'female', 'black'], axis=1)
polls_subset['main'] = np.where(polls_subset['bush'] == 1, 1, np.where(polls_subset['bush'] == 0, 1, 0))

# Drop nan in polls_subset.bush
polls_subset_no_nan = polls_subset[polls_subset.bush.notnull()]
polls_subset_no_nan = polls_subset_no_nan.drop(['main'], axis=1)

# define other data summaries
n = len(polls_subset.bush)              # of survey respondents
n_no_nan = len(polls_subset_no_nan.bush)             # of survey respondents
n_sex = max(polls_subset.sex)           # of sex categories
n_race = max(polls_subset.race)         # of race categories
n_age = max(polls_subset.age)           # of age categories
n_edu = max(polls_subset.edu)           # of education categories
n_state = max(polls_subset.state)       # of states

""" Extra Step: Validation Data"""
# load in 1988 election data as a validation check
election88 = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/ARM_Data/election88/election88.csv")
election88 = election88.drop(election88.columns[[0]], axis=1)
# stnum: state id
# st: state abbreviation
# electionresult: is the outcome of the election
# samplesize: 
# raking:
# merge_:

"""Step 3: Load 1988 census data to enable poststratification."""
census88 = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/ARM_Data/election88/census88.csv")
census88 = census88.drop(census88.columns[[0]], axis=1)
census88 = pd.merge(census88, state_info, on='state', how='left')
census88 = pd.merge(census88, presvote, on='state', how='left')
# edu: categorical variable indicating level of education
# age: categorical variable indicating age
# female: indicator (=1) for female
# black: indicator (=1) for black
# N: size of population in this cell
# Change female to sex and black to race:
census88.loc[:,'sex'] = census88.loc[:,'female'] + 1
census88.loc[:,'race'] = census88.loc[:,'black'] + 1
census88 = census88.drop(['female', 'black'], axis=1)

"""Step 4: Fit a regression model for an individual survey response given demographics, geography etc."""
################################
#### 1st model: Probability that a voter casts a vote on a main party candidate
################################
# Pr(Y_i \in {Obama, Romney}) = logit^{-1}(alpha[1] + alpha[2] * v_prev_j[i] + a^state_j[i] + a^edu_j[i] + a^sex_j[i] + a^age_j[i]
#    + a^race_j[i] + a^partyID_j[i] + a^ideology_j[i] + a^lastvote_j[i])
# a^{}_j[i] are the varying coefficients associated with each categorical variable; with independent prior distributions:
# a^{}_j[i] ~ N(0,sigma^2_var)
# the variance parameters are assigned a hyper prior distribution:
# sigma^2_var ~ invX^2(v,sigma^2_0)
# with a weak prior specification for v and sigma^2_0

# Model description:
model_1 = """
data {
  int<lower=0> N;
  int<lower=0> n_state;
  int<lower=0> n_edu;
  int<lower=0> n_sex;
  int<lower=0> n_age;
  int<lower=0> n_race;
  #int<lower=0> n_party_id;
  #int<lower=0> n_ideology;
  #int<lower=0> n_lastvote;
  vector[N] state_v_prev;
  int<lower=0,upper=n_state> state[N];
  int<lower=0,upper=n_edu> edu[N];
  int<lower=0,upper=n_sex> sex[N];
  int<lower=0,upper=n_age> age[N];
  int<lower=0,upper=n_race> race[N];
  #int<lower=0,upper=n_party_id> party_id[N];
  #int<lower=0,upper=n_ideology> ideology[N];
  #int<lower=0,upper=n_lastvote> lastvote[N];
  int<lower=0,upper=1> y[N];
} 
parameters {
  vector[2] alpha;
  vector[n_state] a;
  vector[n_edu] b;
  vector[n_sex] c;
  vector[n_age] d;
  vector[n_race] e;
  #vector[n_party_id] f;
  #vector[n_ideology] g;
  #vector[n_lastvote] h;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_b;
  real<lower=0,upper=100> sigma_c;
  real<lower=0,upper=100> sigma_d;
  real<lower=0,upper=100> sigma_e;
  #real<lower=0,upper=100> sigma_f;
  #real<lower=0,upper=100> sigma_g;
  #real<lower=0,upper=100> sigma_h;
  real<lower=0> mu;
  real<lower=0,upper=100> sigma_0;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = alpha[1] + alpha[2] * state_v_prev[i] + a[state[i]] + b[edu[i]] + c[sex[i]] + d[age[i]] +
        e[race[i]]; #+ f[party_id[i]] + g[ideology[i]] + h[lastvote[i]];
} 
model {
  a ~ normal (0, sigma_a);
  b ~ normal (0, sigma_b);
  c ~ normal (0, sigma_c);
  d ~ normal (0, sigma_d);
  e ~ normal (0, sigma_e);
  #f ~ normal (0, sigma_f);
  #g ~ normal (0, sigma_g);
  #h ~ normal (0, sigma_h);
  alpha ~ normal(0, 100);
  sigma_a ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_b ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_c ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_d ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_e ~ scaled_inv_chi_square(mu,sigma_0);
  #sigma_f ~ scaled_inv_chi_square(mu,sigma_0);
  #sigma_g ~ scaled_inv_chi_square(mu,sigma_0);
  #sigma_h ~ scaled_inv_chi_square(mu,sigma_0);
  mu ~ uniform(0, 100);
  sigma_0 ~ uniform(0, 100);
  y ~ bernoulli_logit(y_hat);
}
"""

# Model parameters and data:
model_1_data_dict = {'N': n, 'n_state': n_state, 'n_edu': n_edu, 'n_sex': n_sex, 'n_age': n_age, 'n_race': n_race,
  'state': polls_subset.state, 'edu': polls_subset.edu, 'sex': polls_subset.sex, 'age': polls_subset.age,
  'race': polls_subset.race, 'state_v_prev': polls_subset.v_prev, 'y': polls_subset.main}

# Fitting the model:
n_chains = 2
n_iter = 1000
#full_model_fit = pystan.stan(model_code=full_model, data=full_model_data_dict, iter=n_iter, chains=2)
#sm = StanModel(model_code=model_1)
#with open('model_1.pkl', 'wb') as f:
#    pickle.dump(sm, f)
sm = pickle.load(open('model_1.pkl', 'rb'))
model_1_fit = sm.sampling(data=model_1_data_dict, iter=n_iter, chains=n_chains)

# Plot coefficients with confidence intervals:
params_demo = model_1_fit.extract(['alpha', 'b', 'c', 'd', 'e'])
params_alpha_0 = pd.DataFrame({'Intercept' : params_demo['alpha'][:,0]})
params_b = pd.DataFrame(OrderedDict({'Edu ' + str(i+1) : params_demo['b'][:,i] for i in range(0,params_demo['b'].shape[1])}))
params_c = pd.DataFrame(OrderedDict({'Sex ' + str(i+1) : params_demo['c'][:,i] for i in range(0,params_demo['c'].shape[1])}))
params_d = pd.DataFrame(OrderedDict({'Age ' + str(i+1) : params_demo['d'][:,i] for i in range(0,params_demo['d'].shape[1])}))
params_e = pd.DataFrame(OrderedDict({'Race ' + str(i+1) : params_demo['e'][:,i] for i in range(0,params_demo['e'].shape[1])}))
params_demo = pd.concat([params_alpha_0, params_b, params_c, params_d, params_e], axis=1)
ticks_list = list(params_demo.columns.values)
plt.figure(figsize=(10,15))
plt.plot(params_demo.median(), range(params_demo.shape[1]), 'ko', ms = 10)
plt.hlines(range(params_demo.shape[1]), params_demo.quantile(0.025), params_demo.quantile(0.975), 'k')
plt.hlines(range(params_demo.shape[1]), params_demo.quantile(0.25), params_demo.quantile(0.75), 'k', linewidth = 3)
plt.axvline(0, linestyle = 'dashed', color = 'k')
plt.xlabel('Median Coefficient Estimate (50 and 95% CI)')
plt.yticks(range(params_demo.shape[1]), ticks_list)
plt.ylim([-1, params_demo.shape[1]])
plt.xlim([(min(params_demo.quantile(0.025))-0.5), (max(params_demo.quantile(0.975))+0.5)])
plt.title('Coefficients')
plt.tight_layout()
plt.savefig('DemoCoefficients_ConfidenceIntervals.png')
plt.show()

# Plot coefficients with confidence intervals:
params_state = model_1_fit.extract(['alpha', 'a'])
params_alpha_1 = pd.DataFrame({'Prev Vote' : params_state['alpha'][:,1]})
params_a = pd.DataFrame(OrderedDict({'State ' + str(i+1) : params_state['a'][:,i] for i in range(0,params_state['a'].shape[1])}))
params_state = pd.concat([params_alpha_1, params_a], axis=1)
ticks_list = list(params_state.columns.values)
plt.figure(figsize=(10,15))
plt.plot(params_state.median(), range(params_state.shape[1]), 'ko', ms = 10)
plt.hlines(range(params_state.shape[1]), params_state.quantile(0.025), params_state.quantile(0.975), 'k')
plt.hlines(range(params_state.shape[1]), params_state.quantile(0.25), params_state.quantile(0.75), 'k', linewidth = 3)
plt.axvline(0, linestyle = 'dashed', color = 'k')
plt.xlabel('Median Coefficient Estimate (50 and 95% CI)')
plt.yticks(range(params_state.shape[1]), ticks_list)
plt.ylim([-1, params_state.shape[1]])
plt.xlim([(min(params_state.quantile(0.025))-0.5), (max(params_state.quantile(0.975))+0.5)])
plt.title('State Intercepts')
plt.tight_layout()
plt.savefig('StateIntercepts_ConfidenceIntervals.png')
plt.show()

# Traceplot:
model_1_fit.plot()
plt.savefig('ParameterDistributions_model_1.png')
plt.show()

################################
#### 2nd model: Probability that a voter casts a vote for Bush
################################
# 2nd model:
# Pr(Y_i = Obama | Y_i \in {Obama, Romney}) = logit^{-1}(beta_0 + beta_1 + b^state_j[i] + b^edu_j[i]
#     + b^sex_j[i] + b^age_j[i] + b^race_j[i] + b^partyID_j[i] + b^ideology_j[i] + b^lastvote_j[i])
# b^{}_j[i] ~ N(0,eta^2_var)
# eta^2_var ~ invX^2(mu,eta^2_0)
# run daily with four-dat moving window(t, t-1, t-2, t-3)

# Model description:
model_2 = """
data {
  int<lower=0> N;
  int<lower=0> n_state;
  int<lower=0> n_edu;
  int<lower=0> n_sex;
  int<lower=0> n_age;
  int<lower=0> n_race;
  #int<lower=0> n_party_id;
  #int<lower=0> n_ideology;
  #int<lower=0> n_lastvote;
  vector[N] state_v_prev;
  int<lower=0,upper=n_state> state[N];
  int<lower=0,upper=n_edu> edu[N];
  int<lower=0,upper=n_sex> sex[N];
  int<lower=0,upper=n_age> age[N];
  int<lower=0,upper=n_race> race[N];
  #int<lower=0,upper=n_party_id> party_id[N];
  #int<lower=0,upper=n_ideology> ideology[N];
  #int<lower=0,upper=n_lastvote> lastvote[N];
  int<lower=0,upper=1> y[N];
} 
parameters {
  vector[2] alpha;
  vector[n_state] a;
  vector[n_edu] b;
  vector[n_sex] c;
  vector[n_age] d;
  vector[n_race] e;
  #vector[n_party_id] f;
  #vector[n_ideology] g;
  #vector[n_lastvote] h;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_b;
  real<lower=0,upper=100> sigma_c;
  real<lower=0,upper=100> sigma_d;
  real<lower=0,upper=100> sigma_e;
  #real<lower=0,upper=100> sigma_f;
  #real<lower=0,upper=100> sigma_g;
  #real<lower=0,upper=100> sigma_h;
  real<lower=0> mu;
  real<lower=0,upper=100> sigma_0;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = alpha[1] + alpha[2] * state_v_prev[i] + a[state[i]] + b[edu[i]] + c[sex[i]] + d[age[i]] + e[race[i]];
    #+ f[party_id[i]] + g[ideology[i]] + h[lastvote[i]];
} 
model {
  a ~ normal (0, sigma_a);
  b ~ normal (0, sigma_b);
  c ~ normal (0, sigma_c);
  d ~ normal (0, sigma_d);
  e ~ normal (0, sigma_e);
  #f ~ normal (0, sigma_f);
  #g ~ normal (0, sigma_g);
  #h ~ normal (0, sigma_h);
  alpha ~ normal(0, 100);
  sigma_a ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_b ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_c ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_d ~ scaled_inv_chi_square(mu,sigma_0);
  sigma_e ~ scaled_inv_chi_square(mu,sigma_0);
  #sigma_f ~ scaled_inv_chi_square(mu,sigma_0);
  #sigma_g ~ scaled_inv_chi_square(mu,sigma_0);
  #sigma_h ~ scaled_inv_chi_square(mu,sigma_0);
  mu ~ uniform(0, 100);
  sigma_0 ~ uniform(0, 100);
  y ~ bernoulli_logit(y_hat);
}
"""

# Model parameters and data:
model_2_data_dict = {'N': n_no_nan, 'n_state': n_state, 'n_edu': n_edu, 'n_sex': n_sex, 'n_age': n_age, 'n_race': n_race,
  'state': polls_subset_no_nan.state, 'edu': polls_subset_no_nan.edu, 'sex': polls_subset_no_nan.sex, 'age': polls_subset_no_nan.age,
  'race': polls_subset_no_nan.race, 'state_v_prev': polls_subset_no_nan.v_prev, 'y': polls_subset_no_nan.bush.astype(int)}

# Fitting the model:
n_chains = 2
n_iter = 1000
#full_model_fit = pystan.stan(model_code=full_model, data=full_model_data_dict, iter=n_iter, chains=2)
#sm = StanModel(model_code=model_2)
#with open('model_2.pkl', 'wb') as f:
#    pickle.dump(sm, f)
sm = pickle.load(open('model_2.pkl', 'rb'))
model_2_fit = sm.sampling(data=model_2_data_dict, iter=n_iter, chains=n_chains)

# Plot coefficients with confidence intervals:
params_demo = model_2_fit.extract(['alpha', 'b', 'c', 'd', 'e'])
params_alpha_0 = pd.DataFrame({'Intercept' : params_demo['alpha'][:,0]})
params_b = pd.DataFrame(OrderedDict({'Edu ' + str(i+1) : params_demo['b'][:,i] for i in range(0,params_demo['b'].shape[1])}))
params_c = pd.DataFrame(OrderedDict({'Sex ' + str(i+1) : params_demo['c'][:,i] for i in range(0,params_demo['c'].shape[1])}))
params_d = pd.DataFrame(OrderedDict({'Age ' + str(i+1) : params_demo['d'][:,i] for i in range(0,params_demo['d'].shape[1])}))
params_e = pd.DataFrame(OrderedDict({'Race ' + str(i+1) : params_demo['e'][:,i] for i in range(0,params_demo['e'].shape[1])}))
params_demo = pd.concat([params_alpha_0, params_b, params_c, params_d, params_e], axis=1)
ticks_list = list(params_demo.columns.values)
plt.figure(figsize=(10,15))
plt.plot(params_demo.median(), range(params_demo.shape[1]), 'ko', ms = 10)
plt.hlines(range(params_demo.shape[1]), params_demo.quantile(0.025), params_demo.quantile(0.975), 'k')
plt.hlines(range(params_demo.shape[1]), params_demo.quantile(0.25), params_demo.quantile(0.75), 'k', linewidth = 3)
plt.axvline(0, linestyle = 'dashed', color = 'k')
plt.xlabel('Median Coefficient Estimate (50 and 95% CI)')
plt.yticks(range(params_demo.shape[1]), ticks_list)
plt.ylim([-1, params_demo.shape[1]])
plt.xlim([(min(params_demo.quantile(0.025))-0.5), (max(params_demo.quantile(0.975))+0.5)])
plt.title('Coefficients')
plt.tight_layout()
plt.savefig('DemoCoefficients_ConfidenceIntervals_m2.png')
plt.show()

# Plot coefficients with confidence intervals:
params_state = model_2_fit.extract(['alpha', 'a'])
params_alpha_1 = pd.DataFrame({'Prev Vote' : params_state['alpha'][:,1]})
params_a = pd.DataFrame(OrderedDict({'State ' + str(i+1) : params_state['a'][:,i] for i in range(0,params_state['a'].shape[1])}))
params_state = pd.concat([params_alpha_1, params_a], axis=1)
ticks_list = list(params_state.columns.values)
plt.figure(figsize=(10,15))
plt.plot(params_state.median(), range(params_state.shape[1]), 'ko', ms = 10)
plt.hlines(range(params_state.shape[1]), params_state.quantile(0.025), params_state.quantile(0.975), 'k')
plt.hlines(range(params_state.shape[1]), params_state.quantile(0.25), params_state.quantile(0.75), 'k', linewidth = 3)
plt.axvline(0, linestyle = 'dashed', color = 'k')
plt.xlabel('Median Coefficient Estimate (50 and 95% CI)')
plt.yticks(range(params_state.shape[1]), ticks_list)
plt.ylim([-1, params_state.shape[1]])
plt.xlim([(min(params_state.quantile(0.025))-0.5), (max(params_state.quantile(0.975))+0.5)])
plt.title('State Intercepts')
plt.tight_layout()
plt.savefig('StateIntercepts_ConfidenceIntervals_m2.png')
plt.show()

# Traceplot:
model_2_fit.plot()
plt.savefig('ParameterDistributions_model_2.png')
plt.show()

"""# Plot individual parameter's different chains:
b = basic_model_fit.extract(permuted=True)['b']
b_split = np.array_split(b, n_chains) # assumes that the b array is just one chain tacked onto the end of another
for i in range(n_chains):
    plt.plot(b_split[i])
plt.savefig('Traceplot.png')
plt.show()"""

"""Poststratification"""
## Using the model inferences to estimate avg opinion for each state
# construct the n.sims x 3264 matrix
params_m1 = model_1_fit.extract(['alpha', 'a', 'b', 'c', 'd', 'e'])
alpha_m1 = pd.DataFrame(params_m1['alpha'])
a_m1 = pd.DataFrame(params_m1['a'])
b_m1 = pd.DataFrame(params_m1['b'])
c_m1 = pd.DataFrame(params_m1['c'])
d_m1 = pd.DataFrame(params_m1['d'])
e_m1 = pd.DataFrame(params_m1['e'])
params_m2 = model_2_fit.extract(['alpha', 'a', 'b', 'c', 'd', 'e'])
alpha_m2 = pd.DataFrame(params_m2['alpha'])
a_m2 = pd.DataFrame(params_m2['a'])
b_m2 = pd.DataFrame(params_m2['b'])
c_m2 = pd.DataFrame(params_m2['c'])
d_m2 = pd.DataFrame(params_m2['d'])
e_m2 = pd.DataFrame(params_m2['e'])
L = census88.shape[0]
y_pred = np.full((int((n_iter / 2) * n_chains),L), np.nan)
y_pred_cond = np.full((int((n_iter / 2) * n_chains),L), np.nan)
for l in range(0, L):
  y_pred[:,l] = sp.special.expit(alpha_m1.ix[:,0] + alpha_m1.ix[:,1] * census88.v_prev[l] + 
    a_m1.ix[:,census88.state[l]-1] + b_m1.ix[:,census88.edu[l]-1] + c_m1.ix[:,census88.sex[l]-1] + 
    d_m1.ix[:,census88.age[l]-1] + e_m1.ix[:,census88.race[l]-1])
  y_pred_cond[:,l] = sp.special.expit(alpha_m2.ix[:,0] + alpha_m2.ix[:,1] * census88.v_prev[l] + 
    a_m2.ix[:,census88.state[l]-1] + b_m2.ix[:,census88.edu[l]-1] + c_m2.ix[:,census88.sex[l]-1] + 
    d_m2.ix[:,census88.age[l]-1] + e_m2.ix[:,census88.race[l]-1])

# Convert to unconditional probabilities:
y_bush = y_pred_cond * y_pred
y_non_bush = (1 - y_pred_cond) * y_pred

# Normalized:
y_bush_norm = y_bush / (y_bush + y_non_bush)
y_non_bush_norm = y_non_bush / (y_bush + y_non_bush)

# average over strata within each state
y_pred_state = np.full((int((n_iter / 2) * n_chains),n_state), np.nan)
for j in range(1,n_state+1):
    sel = [s for s in range(L) if census88.state[s] ==  j]
    y_pred_state[:,j-1] = np.divide((np.dot(y_bush_norm[:,sel],(census88[census88.state == j]).N)),sum((census88[census88.state == j]).N))
y_pred_state = pd.DataFrame(y_pred_state)

y_pred_state_bush = np.full((int((n_iter / 2) * n_chains),n_state), np.nan)
for j in range(1,n_state+1):
    sel = [s for s in range(L) if census88.state[s] ==  j]
    y_pred_state_bush[:,j-1] = np.divide((np.dot(y_bush[:,sel],(census88[census88.state == j]).N)),sum((census88[census88.state == j]).N))
y_pred_state_bush = pd.DataFrame(y_pred_state_bush)

y_pred_state_non_bush = np.full((int((n_iter / 2) * n_chains),n_state), np.nan)
for j in range(1,n_state+1):
    sel = [s for s in range(L) if census88.state[s] ==  j]
    y_pred_state_non_bush[:,j-1] = np.divide((np.dot(y_non_bush[:,sel],(census88[census88.state == j]).N)),sum((census88[census88.state == j]).N))
y_pred_state_non_bush = pd.DataFrame(y_pred_state_non_bush)

"""#Old plotting method:
plt.figure(figsize=(16, 6))
sns.boxplot(data=y_pred_state, whis=np.inf, color="c")
plt.savefig('Estimates_state.png')
plt.show()"""

# New plotting method:
ticks_list = list(state_info.state_abbr.values)
plt.figure(figsize=(10,20))
plt.plot(y_pred_state.median(), range(y_pred_state.shape[1]), 'ko', ms = 10)
plt.plot(election88.electionresult, range(election88.shape[0]), 'r.', ms = 10)
plt.hlines(range(y_pred_state.shape[1]), y_pred_state.quantile(0.025), y_pred_state.quantile(0.975), 'k')
plt.hlines(range(y_pred_state.shape[1]), y_pred_state.quantile(0.25), y_pred_state.quantile(0.75), 'k', linewidth = 3)
plt.axvline(0.5, linestyle = 'dashed', color = 'k')
plt.xlabel('Median State Estimate (50 and 95% CI) and Actual Election Outcome (red)')
plt.yticks(range(y_pred_state.shape[1]), ticks_list)
plt.ylim([-1, y_pred_state.shape[1]])
plt.xlim([(min(y_pred_state.quantile(0.025))-0.5), (max(y_pred_state.quantile(0.975))+0.5)])
plt.title('State Estimates')
plt.tight_layout()
plt.savefig('State_Estimates_Normalized.png')
plt.show()

# New plotting method:
ticks_list = list(state_info.state_abbr.values)
plt.figure(figsize=(10,20))
plt.plot(y_pred_state_bush.median(), range(y_pred_state_bush.shape[1]), 'ro', ms = 10)
plt.plot(y_pred_state_non_bush.median(), range(y_pred_state_non_bush.shape[1]), 'bo', ms = 10)
plt.plot(election88.electionresult, range(election88.shape[0]), 'm.', ms = 10)
plt.hlines(range(y_pred_state_bush.shape[1]), y_pred_state_bush.quantile(0.025), y_pred_state_bush.quantile(0.975), 'r')
plt.hlines(range(y_pred_state_bush.shape[1]), y_pred_state_bush.quantile(0.25), y_pred_state_bush.quantile(0.75), 'r', linewidth = 3)
plt.hlines(range(y_pred_state_non_bush.shape[1]), y_pred_state_non_bush.quantile(0.025), y_pred_state_non_bush.quantile(0.975), 'b')
plt.hlines(range(y_pred_state_non_bush.shape[1]), y_pred_state_non_bush.quantile(0.25), y_pred_state_non_bush.quantile(0.75), 'b', linewidth = 3)
plt.axvline(0.5, linestyle = 'dashed', color = 'k')
plt.xlabel('Median State Estimate (50 and 95% CI) and Actual Election Outcome (red)')
plt.yticks(range(y_pred_state_bush.shape[1]), ticks_list)
plt.ylim([-1, y_pred_state_bush.shape[1]])
plt.xlim([(min(y_pred_state_bush.quantile(0.025))-0.5), (max(y_pred_state_bush.quantile(0.975))+0.5)])
plt.title('State Estimates')
plt.tight_layout()
plt.savefig('State_Estimates_Actual.png')
plt.show()

#"""Extension: A more intricate model"""
#extended_model_fit = pystan.stan(file='election88_expansion.stan', data=full_model_data_dict, iter=1000, chains=4)

