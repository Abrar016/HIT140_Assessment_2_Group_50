#!/usr/bin/env python
# coding: utf-8

# # Loading Data

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as st
import math
import statistics as stats
import matplotlib.pyplot as plt

a = np.loadtxt('po1_data.txt',  delimiter = ',')


print(a)

df = pd.DataFrame(a)


df.columns = ['subject identifier', ' jitter in %', 'absolute jitter in ms', 'jitter r.a.p.', 'jitter p.p.q.5','jitter d.d.p.' ,
              'shimmer in %','absolute shimmer dB', 'shimmer a.p.q.3', 'shimmer a.p.q.5','shimmer a.p.q.11','shimmer d.d.a', 
              'autocorrelation between NHR and HNR', 'NHR','HNR','median pitch','mean pitch','sd of pitch','min pitch',
              'max pitch','number of pulses','number of periods','mean period','sd of period','fraction of unvoiced frames',
              'num of voice breaks','degree of voice breaks','UPDRS','PD label']


print(df)

column_names = list(df.columns)




# # Data Wrangling

# In[2]:


df.isnull()

print(df.describe())

print(df.dtypes)

print(df.info())


# In[7]:


df1 = df[df["PD label"] == 1]
df2 = df[df["PD label"] == 0]


# # t-test

# In[8]:


salient_features = []
for i in range(1,28):
    print('Analysis of the measurement variable', column_names[i])
    print()
    sample1 = df1.iloc[:, i].to_numpy()
    sample2 = df2.iloc[:, i].to_numpy()
    
    

    # the basic statistics of sample 1:
    x_bar1 = st.tmean(sample1)
    s1 = st.tstd(sample1)
    n1 = len(sample1)
    print("\n Statistics of sample 1: %.3f (mean), %.3f (std. dev.), and %d (n)." % (x_bar1, s1, n1))
    plt.hist(sample1, color='Red', edgecolor='black')
    plt.title("Histogram View")
    plt.xlabel(column_names[i])
    plt.ylabel("Frequency")
    plt.show()
    
    
    # the basic statistics of sample 2:
    x_bar2 = st.tmean(sample2)
    s2 = st.tstd(sample2)
    n2 = len(sample2)
    print("\n Statistics of sample 2: %.3f (mean), %.3f (std. dev.), and %d (n)." % (x_bar2, s2, n2))
    plt.hist(sample2, color='Yellow', edgecolor='black')
    plt.title("Histogram View")
    plt.xlabel(column_names[i])
    plt.ylabel("Frequency")
    plt.show()

    # perform two-sample t-test
    # null hypothesis: mean of sample 1 = mean of sample 2
    # alternative hypothesis: mean of sample 1 is not equal to mean of sample 2
    # note the argument equal_var=False, which assumes that two populations do not have equal variance
    t_stats, p_val = st.ttest_ind_from_stats(x_bar1, s1, n1, x_bar2, s2, n2, equal_var=False, alternative='two-sided')
    print("\n Computing t* ...")
    print("\t t-statistic (t*): %.2f" % t_stats)

    print("\n Computing p-value ...")
    print("\t p-value: %.4f" % p_val)

    print("\n Conclusion:")
    if p_val < 0.05:
        print("\t We reject the null hypothesis for", column_names[i])
        salient_features.append(column_names[i])
    else:
        print("\t We accept the null hypothesis for", column_names[i])
        print()


# # Output

# In[5]:


print(salient_features)


# In[6]:


len(salient_features)


# In[ ]:




