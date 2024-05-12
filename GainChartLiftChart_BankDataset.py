#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd


# In[8]:


bank_df=pd.read_csv("bank.csv")


# In[9]:


bank_df.head(5)


# In[10]:


bank_df.info()


# In[12]:


x_features=list(bank_df.columns)
x_features.remove('y')
x_features


# In[13]:


encoded_bank_df = pd.get_dummies( bank_df[x_features], drop_first=True)


# In[14]:


Y=bank_df.y
X=encoded_bank_df


# In[16]:


import statsmodels.api as sm
logit_model=sm.Logit( Y, sm.add_constant(X)).fit()


# In[17]:


logit_model.summary2()


# In[19]:


def get_significant_vars(lm):
    #Store the p-values and corresponding column names in a dataframe
    var_p_vals_df= pd.DataFrame(lm.pvalues)
    var_p_vals_df ['vars'] = var_p_vals_df.index
    var_p_vals_df.columns= ['pvals', 'vars']
    

    #Filter the column names where p-value is less than 0.05
    return list(var_p_vals_df [var_p_vals_df.pvals <= 0.05] ['vars'])


# In[20]:


significant_vars= get_significant_vars(logit_model)
significant_vars


# In[23]:


x_features=[
 'duration',
 'campaign',
 'pdays',
 'emp_var_rate',
 'cons_price_idx',
 'cons_conf_idx',
 'euribor3m',
 'job_blue-collar',
 'job_retired',
 'education_university.degree',
 'default_unknown',
 'contact_telephone',
 'month_aug',
 'month_jun',
 'month_mar',
 'month_may',
 'month_nov',
 'month_sep',
 'day_of_week_wed',
 'poutcome_nonexistent',
 'poutcome_success']
X_2=sm.add_constant(X[x_features])
logit_model_2=sm.Logit(Y,X_2).fit()


# In[24]:


logit_model_2.summary2()


# In[26]:


y_pred_df= pd.DataFrame( { 'actual': Y, 'predicted_prob': logit_model_2.predict( sm.add_constant(X[x_features])) })


# In[27]:


sorted_predict_df = y_pred_df [['predicted_prob', 'actual']].sort_values('predicted_prob', ascending =False)


# In[28]:


num_per_decile =int(len(sorted_predict_df) / 10) 
print("Number of observations per decile:", num_per_decile)


# In[30]:


def get_deciles (df):
    #Set first decile
    df ['decile']=1
    idx=0

    #Iterate through all 10 deciles
    for each_d in range(0, 10):
        #Setting each 452 observations to one decile in sequence
        df.iloc[idx: idx+num_per_decile, df.columns.get_loc('decile')]= each_d
        idx += num_per_decile
    df ['decile']=df ['decile']+1
    return df


# In[31]:


deciles_predict_df =get_deciles (sorted_predict_df)


# In[32]:


deciles_predict_df[0:10]


# In[37]:



#calculating gain:

gain_lift_df=pd.DataFrame(
deciles_predict_df.groupby(
'decile')['actual'].sum()).reset_index()


# In[39]:


gain_lift_df.columns=['decile', 'gain']


# In[40]:


gain_lift_df['gain_percentage']=(100 * gain_lift_df.gain.cumsum()/gain_lift_df.gain.sum())


# In[41]:


gain_lift_df


# In[42]:



import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


plt.figure(figsize=(8,4))
plt.plot(gain_lift_df['decile'], gain_lift_df['gain_percentage'], '-')
plt.show()


# In[44]:


gain_lift_df['lift']=(gain_lift_df.gain_percentage / (gain_lift_df.decile * 10))
gain_lift_df


# In[45]:


plt.figure( figsize=(8,4))
plt.plot( gain_lift_df['decile'], gain_lift_df['lift'], '-')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




