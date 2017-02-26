"""
import os
os.chdir('/Users/davidminarsch/Desktop/PythonMLM/Election_Example')
exec(open("election88.py").read())
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
candidate_effects['candidate_effects_weighted'] = (candidate_effects['X76'] + candidate_effects['X80'] + candidate_effects['X84']) / 3.0
candidate_effects_1 = candidate_effects.iloc[:9]
candidate_effects = pd.concat([candidate_effects_1,candidate_effects.ix[8:]]).reset_index(drop=True)
candidate_effects.iloc[8] = 0
candidate_effects = candidate_effects.set_value(8, 'state_abbr', 'DC')
presvote['v_prev'] += candidate_effects['candidate_effects_weighted']

#merge all three dataframes into one:
polls = pd.merge(polls, state_info, on='state', how='left')
polls = pd.merge(polls, presvote, on='state', how='left')

#select subset of polls:
polls_subset = polls.loc[polls['survey'] == '9158']
#drop nan in polls_subset.bush
polls_subset_no_nan = polls_subset[polls_subset.bush.notnull()]
# ideally we do a two-step procedure, where we first estimate support for a main candidate;

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
print("National mean of raw data (percent for Bush): %.2f" % (mean_nat_raw * 100))
# National mean of raw data (percent for Bush): 55.80
mean_nat_wgt_raw = round(y_w_no_nan.bush.dot(y_w_no_nan.weight) / sum(y_w_no_nan.weight),3) 			# national weighted mean of raw data
print("National weighted mean of raw data (percent for Bush): %.2f" % (mean_nat_wgt_raw * 100))
# National weighted mean of raw data (percent for Bush): 54.30

# compute weighted averages for the states
mean_state_wgt_raw = dict(zip(state_info.state, [0 for i in range(n_state)]))
for key in mean_state_wgt_raw:
	y_w_select = y_w_s_no_nan[y_w_s_no_nan.state == key]
	s = sum(y_w_select.weight)
	if s == 0:
		mean_state_wgt_raw[key] = 'nan'
	else:
		mean_state_wgt_raw[key] = round(y_w_select.bush.dot(y_w_select.weight) / sum(y_w_select.weight),3)

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

"""Step 3: Load census data to enable poststratification."""
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


# Drop incomplete observations:
polls_subset_no_nan = polls_subset_no_nan.dropna() 
# Set up the predictors:
polls_subset_no_nan['age_edu'] = n_edu * (polls_subset_no_nan['age'] - 1) + polls_subset_no_nan['edu']
n_age_edu = max(polls_subset_no_nan.age_edu)            # of age x edu categories

"""Step 4: Fit a regression model for an individual survey response given demographics, geography etc."""
################################
#### A very basic model
################################
# multilevel logistic regression with a simple example including two individual predictors - female and black - and the 51 states
# Pr(y_i=1) = logit^{−1}(α_j[i]+β^female·female_i+β^black·black_i) , fori=1,...,n
# α_j ∼ N(μ_α,σ^2_state) ,forj=1,...,51.
# lmer(y ~ black + female + (1|state), family=binomial(link="logit"))
basic_model = """
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

# Model parameters and data:
basic_model_data_dict = {'N': len(polls_subset_no_nan.bush), 'n_state': n_state, 'black': polls_subset_no_nan.black,
  'female': polls_subset_no_nan.female, 'state': polls_subset_no_nan.state , 'y': polls_subset_no_nan.bush.astype(int)}

# Fitting the model:
n_chains = 2
n_iter = 1000
basic_model_fit = pystan.stan(model_code=basic_model, data=basic_model_data_dict, iter=n_iter, chains=n_chains)

# Analysis of the results:
b_sample = pd.DataFrame(basic_model_fit['b'])
print("Coefficient mean and standard error, respectively, for black: (%.2f, %.2f)" % (round(b_sample[0].mean(),1), round(b_sample[0].std(),1)))
# Coefficient mean and standard error, respectively, for black: (-1.80, 0.20)
print("Coefficient mean and standard error, respectively, for female: (%.2f, %.2f)" % (round(b_sample[1].mean(),1), round(b_sample[1].std(),1)))
# Coefficient mean and standard error, respectively, for female: (-0.10, 0.10)
mu_a_sample = pd.DataFrame(basic_model_fit['mu_a'])
print("Average intercept of states mean and standard error, respectively: (%.2f, %.2f)" % (round(mu_a_sample.mean(),1), round(mu_a_sample.std(),1)))
# Average intercept of states mean and standard error, respectively: (0.40, 0.10)
sigma_a_sample = pd.DataFrame(basic_model_fit['sigma_a'])
print("Estimate of standard deviation of mu_a: %.2f" % round(sigma_a_sample.mean(),1))
# Estimate of standard deviation of mu_a: 0.40

# Plot indicator parameter slopes and state intercept mean with confidence intervals:
params = basic_model_fit.extract(['b', 'mu_a'])
params = pd.DataFrame({'Coefficient Black' : params['b'][:,0], 'Coefficient Female' : params['b'][:,1], 'Mean of State Coefficients' : params['mu_a']})
ticks_list = list(params.columns.values)
plt.plot(params.median(), range(3), 'ko', ms = 10)
plt.hlines(range(3), params.quantile(0.025), params.quantile(0.975), 'k')
plt.hlines(range(3), params.quantile(0.25), params.quantile(0.75), 'k', linewidth = 3)
plt.axvline(0, linestyle = 'dashed', color = 'k')
plt.xlabel('Median Coefficient Estimate (50 and 95% CI)')
plt.yticks(range(3), ['Coefficient Black', 'Coefficient Female', 'Mean of State Coefficients'])
plt.ylim([-1, 3])
plt.xlim([(min(params.quantile(0.025))-0.5), (max(params.quantile(0.975))+0.5)])
plt.title('Coefficients')
plt.tight_layout()
plt.savefig('ConfidenceIntervals_IndicatorSlopesAndStateMean.png')
plt.show()

# Plot indicator parameter slopes and state intercept mean with confidence intervals:
params = basic_model_fit.extract(['a'])
params_df = pd.DataFrame(OrderedDict({'State ' + str(i+1) : params['a'][:,i] for i in range(0,params['a'].shape[1])}))
ticks_list = ['State ' + str(i+1) for i in range(0,params['a'].shape[1])]
plt.figure(figsize=(5,10))
plt.plot(params_df.median(), range(params_df.shape[1]), 'ko', ms = 10)
plt.hlines(range(params_df.shape[1]), params_df.quantile(0.025), params_df.quantile(0.975), 'k')
plt.hlines(range(params_df.shape[1]), params_df.quantile(0.25), params_df.quantile(0.75), 'k', linewidth = 3)
plt.axvline(0, linestyle = 'dashed', color = 'k')
plt.xlabel('Median Coefficient Estimate (50 and 95% CI)')
plt.yticks(range(params_df.shape[1]), ticks_list)
plt.ylim([-1, params_df.shape[1]])
plt.xlim([(min(params_df.quantile(0.025))-0.5), (max(params_df.quantile(0.975))+0.5)])
plt.title('State Intercepts')
plt.tight_layout()
plt.savefig('ConfidenceIntervals_StateIntercepts.png')
plt.show()

# Traceplot:
basic_model_fit.plot()
plt.savefig('ParameterDistributions.png')
plt.show()

# Plot individual parameter's different chains:
b = basic_model_fit.extract(permuted=True)['b']
b_split = np.array_split(b, n_chains) # assumes that the b array is just one chain tacked onto the end of another
for i in range(n_chains):
    plt.plot(b_split[i])
plt.savefig('Traceplot.png')
plt.show()

################################
#### A fuller model: (election88_full)
################################
# Pr(y_i=1) = logit^{−1}( β^0 + β^female · female_i + β^black · black_i +
# + β^female.black · female_i · black_i + α^age_k[i] + α^edu_l[i] + α^age.edu_k[i]l[i] + α^state_j[i])
# α^state_j[i] ∼ N (α^region_m[j] + β^v.prev · v.prev_j, σ^2_state)
# α^age_k ∼ N(0, σ^2_age), for k = 1,...,4
# α^edu_l ∼ N(0, σ^2_edu), for l = 1,...,4
# α^age.edu_kl ∼ N(0, σ^2_age.edu ), for k = 1,...,4, l = 1,...,4
# α^region_m ∼ N(0, σ^2_region ), for m = 1,...,5
# As with the non-nested linear models in Section 13.5, this model can be expressed in equivalent ways by moving the
# constant term β^0 around. Here we have included β^0 in the data-level regression and included no intercepts in the
# group-level models for the different batches of α’s.
# For model with redundant parametrization see: election88_full_redundant_params.stan
# age.edu <- n.edu*(age-1) + edu
# region.full <- region[state]
# v.prev.full <- v.prev[state]
# lmer(formula = y ~ black + female + black:female + v.prev.full + (1 | age) + (1 | edu) + (1 | age.edu) + (1 | state)
# + (1 | region.full), family = binomial(link = "logit"))
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

# Model parameters and data:
full_model_data_dict = {'N': len(polls_subset_no_nan.bush), 'n_age': n_age, 'n_age_edu': n_age_edu,
	'n_edu': n_edu, 'n_region': n_region, 'n_state': n_state, 'age': polls_subset_no_nan.age,
	'age_edu': polls_subset_no_nan.age_edu, 'black': polls_subset_no_nan.black,
	'edu': polls_subset_no_nan.edu, 'female': polls_subset_no_nan.female,
	'region': polls_subset_no_nan.region, 'state': polls_subset_no_nan.state ,
	'v_prev': polls_subset_no_nan.v_prev, 'y': polls_subset_no_nan.bush.astype(int)}

# Fitting the model:
n_chains = 2 #4
n_iter = 1000
#full_model_fit = pystan.stan(model_code=full_model, data=full_model_data_dict, iter=n_iter, chains=2)
#sm = StanModel(model_code=full_model)
#with open('full_model.pkl', 'wb') as f:
#    pickle.dump(sm, f)
sm = pickle.load(open('full_model.pkl', 'rb'))
full_model_fit = sm.sampling(data=full_model_data_dict, iter=n_iter, chains=n_chains)

# Alternative formulation with redundant parametrization:
"""full_model_alt = 
data {
  int<lower=0> N;
  int<lower=0> n_race;
  int<lower=0> n_sex;
  int<lower=0> n_race_sex;
  int<lower=0> n_age;
  int<lower=0> n_edu;
  int<lower=0> n_age_edu;
  int<lower=0> n_state;
  int<lower=0> n_region;
  int<lower=0,upper=n_race> race[N];
  int<lower=0,upper=n_sex> sex[N];
  int<lower=0,upper=n_race_sex> race_sex[N];
  int<lower=0,upper=n_age> age[N];
  int<lower=0,upper=n_edu> edu[N];
  int<lower=0,upper=n_age_edu> age_edu[N];
  int<lower=0,upper=n_state> state[N];
  int<lower=0,upper=n_region> region[N];
  vector[N] v_prev;
  int<lower=0,upper=1> y[N];
} 
parameters {
  vector[n_race] a;
  vector[n_sex] b;
  vector[n_race_sex] c;
  vector[n_age] d;
  vector[n_edu] e;
  vector[n_age_edu] f;
  vector[n_state] g;
  vector[n_region] h;
  vector[2] beta;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_b;
  real<lower=0,upper=100> sigma_c;
  real<lower=0,upper=100> sigma_d;
  real<lower=0,upper=100> sigma_e;
  real<lower=0,upper=100> sigma_f;
  real<lower=0,upper=100> sigma_g;
  real<lower=0,upper=100> sigma_h;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = beta[1] + beta[2] * v_prev[i] + a[race[i]] + b[sex[i]]
      + c[race_sex[i]] + d[age[i]] + e[edu[i]] + f[age_edu[i]] + g[state[i]] + h[region[i]];
} 
model {
  a ~ normal (0, sigma_a);
  b ~ normal (0, sigma_b);
  c ~ normal (0, sigma_c);
  d ~ normal (0, sigma_d);
  e ~ normal (0, sigma_e);
  f ~ normal (0, sigma_f);
  g ~ normal (0, sigma_g);
  h ~ normal (0, sigma_h);
  beta ~ normal(0, 100);
  y ~ bernoulli_logit(y_hat);
}

# Convert black to race:
polls_subset_no_nan['race'] = polls_subset_no_nan['black'] + 1
n_race = max(polls_subset_no_nan.race)
# Convert female to sex:
polls_subset_no_nan['sex'] = polls_subset_no_nan['female'] + 1
n_sex = max(polls_subset_no_nan.sex)
# Create race_sex:
polls_subset_no_nan['race_sex'] = n_race * (polls_subset_no_nan.sex - 1) + polls_subset_no_nan.race
n_race_sex = max(polls_subset_no_nan.race_sex)

# Model parameters and data:
full_model_data_dict_alt = {'N': len(polls_subset_no_nan.bush), 'n_race': n_race, 'n_sex': n_sex, 'n_race_sex': n_race_sex,
  'n_age': n_age, 'n_edu': n_edu, 'n_age_edu': n_age_edu, 'n_state': n_state, 'n_region': n_region, 'race': polls_subset_no_nan.race,
  'sex': polls_subset_no_nan.sex, 'race_sex': polls_subset_no_nan.race_sex, 'age': polls_subset_no_nan.age, 'edu': polls_subset_no_nan.edu,
  'age_edu': polls_subset_no_nan.age_edu, 'state': polls_subset_no_nan.state, 'region': polls_subset_no_nan.region, 
  'v_prev': polls_subset_no_nan.v_prev, 'y': polls_subset_no_nan.bush.astype(int)}

# Fitting the model:
#sm = StanModel(model_code=full_model_alt)
#with open('full_model_alt.pkl', 'wb') as f:
#    pickle.dump(sm, f)
sm = pickle.load(open('full_model_alt.pkl', 'rb'))
full_model_fit_alt = sm.sampling(data=full_model_data_dict_alt, iter=n_iter, chains=n_chains)
"""

#Output:
param_fm  = full_model_fit.extract(permuted=True)
beta_fm = pd.DataFrame(param_fm['beta'])
beta_fm.rename(columns=lambda x: 'beta_' + str(x+1), inplace=True)
print("Coefficient mean and standard error, respectively, for black [rough percentage effect]: (%.2f, %.2f) [%.2f]" % (round(beta_fm['beta_2'].mean(),1), round(beta_fm['beta_2'].std(),1), (round(beta_fm['beta_2'].mean(),1) / 4 * 100)))
# Coefficient mean and standard error, respectively, for black [rough percentage effect]: (-1.70, 0.30) [-42.50]
# African-American men were ~42% less likely than other men to support Bush, after controlling for age, education, and state.
print("Coefficient mean and standard error, respectively, for female [rough percentage effect]: (%.2f, %.2f) [%.2f]" % (round(beta_fm['beta_3'].mean(),1), round(beta_fm['beta_3'].std(),1), (round(beta_fm['beta_3'].mean(),1) / 4 * 100)))
# Coefficient mean and standard error, respectively, for female [rough percentage effect]: (-0.10, 0.10) [-2.50]
# non-African-American women were very slightly less likely than non-African- American men to support Bush, after controlling for age, education, and state. However, the standard error on this coefficient is as large as the estimate itself, indicating that our sample size is too small for us to be certain of this pattern in the population.
print("Coefficient mean and standard error, respectively, for black x female [rough percentage effect]: (%.2f, %.2f) [%.2f]" % (round(beta_fm['beta_5'].mean(),1), round(beta_fm['beta_5'].std(),1), (round(beta_fm['beta_5'].mean(),1) / 4 * 100)))
# Coefficient mean and standard error, respectively, for black x female [rough percentage effect]: (-0.20, 0.40) [-5.00]
# The large standard error on the coefficient for black:female indicates that the sample size is too small to estimate this interaction precisely.
print("Coefficient mean and standard error, respectively, for v_prev [rough percentage effect]: (%.2f, %.2f) [%.2f]" % (round(beta_fm['beta_4'].mean(),1), round(beta_fm['beta_4'].std(),1), (round(beta_fm['beta_4'].mean(),1) / 4 )))
# Coefficient mean and standard error, respectively, for v_prev [rough percentage effect]: (6.80, 1.90) [1.70]
# A 1% difference in a state’s support for Republican candidates in previous elections mapped to a predicted 1.7% difference in support for Bush in 1988.
a_fm = pd.DataFrame(param_fm['a'])
a_fm.rename(columns=lambda x: 'a_' + str(x), inplace=True)
b_fm = pd.DataFrame(param_fm['b'])
b_fm.rename(columns=lambda x: 'b_' + str(x), inplace=True)
c_fm = pd.DataFrame(param_fm['c'])
c_fm.rename(columns=lambda x: 'c_' + str(x), inplace=True)
d_fm = pd.DataFrame(param_fm['d'])
d_fm.rename(columns=lambda x: 'd_' + str(x), inplace=True)
e_fm = pd.DataFrame(param_fm['e'])
e_fm.rename(columns=lambda x: 'e_' + str(x), inplace=True)
param_fm_df = pd.concat([a_fm, b_fm, c_fm, d_fm, e_fm, beta_fm], axis=1)
param_fm_demo_df = pd.concat([a_fm, b_fm, c_fm, beta_fm[beta_fm.columns[[0,1,2,4]]]], axis=1)
plt.figure(figsize=(20, 6))
sns.boxplot(data=param_fm_demo_df , whis=np.inf, color="c")
plt.savefig('Demographics_Full.png')
plt.show()
# The box shows the quartiles of the dataset while the whiskers extend to show the rest of the
# distribution, except for points that are determined to be “outliers” using a method that is a
# function of the inter-quartile range.

# Plot coefficients with confidence intervals:
ticks_list = list(param_fm_demo_df.columns.values)
plt.figure(figsize=(10,20))
plt.plot(param_fm_demo_df.median(), range(param_fm_demo_df.shape[1]), 'ko', ms = 10)
plt.hlines(range(param_fm_demo_df.shape[1]), param_fm_demo_df.quantile(0.025), param_fm_demo_df.quantile(0.975), 'k')
plt.hlines(range(param_fm_demo_df.shape[1]), param_fm_demo_df.quantile(0.25), param_fm_demo_df.quantile(0.75), 'k', linewidth = 3)
plt.axvline(0, linestyle = 'dashed', color = 'k')
plt.xlabel('Median Coefficient Estimate (50 and 95% CI)')
plt.yticks(range(param_fm_demo_df.shape[1]), ticks_list)
plt.ylim([-1, param_fm_demo_df.shape[1]])
plt.xlim([(min(param_fm_demo_df.quantile(0.025))-0.5), (max(param_fm_demo_df.quantile(0.975))+0.5)])
plt.title('Coefficients')
plt.tight_layout()
plt.savefig('ConfidenceIntervals_Coefficients_fm.png')
plt.show()
# Recall that a change of x on the logistic scale corresponds to a change of at most x/4 on the probability scale.

"""# create linear predictors of demographics
n = len(polls_subset_no_nan.bush)
lin_pred = np.full((1,n), np.nan)
polls_subset_no_nan = polls_subset_no_nan.reset_index(drop=True)
for i in range(0,n):
	lin_pred[0][i] = np.mean(beta_fm.ix[:,0] + beta_fm.ix[:,1] * polls_subset_no_nan.black[i] + 
		beta_fm.ix[:,2] * polls_subset_no_nan.female[i] + beta_fm.ix[:,4] * polls_subset_no_nan.female[i] * polls_subset_no_nan.black[i]
		+ a_fm.ix[:,polls_subset_no_nan.age[i]-1] + b_fm.ix[:,polls_subset_no_nan.edu[i]-1] + c_fm.ix[:,polls_subset_no_nan.age_edu[i]-1])

#Plot sample states
sample_states = np.array([2,3,4,8,6,7,5,9]) #state_info.state_abbr
iterator = np.array(range(0,8))
fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex =True)
axes = axes.ravel()
for i, c in zip(sample_states,iterator):
  state_population = polls_subset_no_nan[polls_subset_no_nan['state'] == i]
    xvals = np.linspace(np.amin(lin_pred), np.amax(lin_pred))
    # Median estimate
    axes[c].plot(xvals, sp.special.expit((d_fm[[i-1]]).median()[0] + lin_pred[0][i]), 'k:')
    axes[c].plot(xvals, sp.special.expit((d_fm[[i-1]]).median()[0] + lin_pred[0][i]), 'k:')
    axes[c].set_xlim(np.amin(lin_pred),np.amax(lin_pred))
    axes[c].set_ylim(0, 1)
    axes[c].set_title(state_info.state_abbr[i-1])
    axes[c].set_ylabel('Pr (support Bush)')
    axes[c].set_xlabel('linear predictor')
plt.savefig('Sample_States.png')
plt.show()"""

"""Poststratification"""
## Using the model inferences to estimate avg opinion for each state
# construct the n.sims x 3264 matrix
census88 = census88.assign(age_edu=n_edu * (census88.age - 1) + census88.edu)
L = census88.shape[0]
y_pred = np.full((int((n_iter / 2) * n_chains),L), np.nan)
for l in range(0, L):
    y_pred[:,l] = sp.special.expit(beta_fm.ix[:,0] + beta_fm.ix[:,1] * census88.black[l] + 
    beta_fm.ix[:,2] * census88.female[l] + beta_fm.ix[:,3] * census88.v_prev[l] +
    beta_fm.ix[:,4] * census88.female[l] * census88.black[l] +
    a_fm.ix[:,census88.age[l]-1] + b_fm.ix[:,census88.edu[l]-1] +
    c_fm.ix[:,census88.age_edu[l]-1] + d_fm.ix[:,census88.state[l]-1] +
    e_fm.ix[:,census88.region[l]-1])

# average over strata within each state
y_pred_state = np.full((int((n_iter / 2) * n_chains),n_state), np.nan)
for j in range(1,n_state+1):
    sel = [s for s in range(L) if census88.state[s] ==  j]
    y_pred_state[:,j-1] = np.divide((np.dot(y_pred[:,sel],(census88[census88.state == j]).N)),sum((census88[census88.state == j]).N))
y_pred_state = pd.DataFrame(y_pred_state)

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
plt.savefig('State_Estimates.png')
plt.show()

#"""Extension: A more intricate model"""
#extended_model_fit = pystan.stan(file='election88_expansion.stan', data=full_model_data_dict, iter=1000, chains=4)

