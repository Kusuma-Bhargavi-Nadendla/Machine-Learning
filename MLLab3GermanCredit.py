#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
credit_df = pd.read_csv( "GermanCredit.csv")
credit_df.info()


# In[5]:


x_features=list(credit_df.columns)
x_features.remove( 'credit_risk')
x_features


# In[6]:


encoded_credit_df=pd.get_dummies(credit_df[x_features], drop_first=True)
list(encoded_credit_df.columns)


# In[7]:


encoded_credit_df


# In[8]:


encoded_credit_df[ ['status_... >= 200 DM / salary for at least 1 year', 'status_0 <= ... < 200 DM', 'status_no checking account']].head(5)


# In[10]:


import statsmodels.api as sm
Y = credit_df.credit_risk
X = sm.add_constant(encoded_credit_df)


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test=train_test_split(X, Y, test_size=0.3, random_state=42)


# In[12]:


import statsmodels.api as sm
logit=sm.Logit(y_train, X_train)
logit_model=logit.fit()


# In[13]:


logit_model.summary2()


# In[15]:


#Model Diagnostics


def get_sigificant_vars( lm ):
    #store the p-values and corresponding column in a dataframe
    var_p_vals_df = pd.DataFrame( lm.pvalues )
    var_p_vals_df['vars']=var_p_vals_df.index
    var_p_vals_df.columns=['pvals', 'vars']
    #Filter the column names where p-value is less than 0.05
    return list( var_p_vals_df[var_p_vals_df.pvals<=0.05]['vars'])


# In[16]:


significant_vars=get_sigificant_vars(logit_model)
significant_vars


# In[17]:


final_logit=sm.Logit(y_train, sm.add_constant(X_train[significant_vars])).fit()


# In[18]:


final_logit.summary2()


# In[24]:


pred=final_logit.predict(sm.add_constant(X_test[significant_vars]))


# In[25]:


y_pred_df=pd.DataFrame( {"actual":y_test, "predicted_prob":pred})


# In[26]:


y_pred_df.sample(10, random_state=42)


# In[ ]:





# In[27]:


y_pred_df['predicted']=y_pred_df.predicted_prob.map( lambda x: 1 if x > 0.5 else 0)
y_pred_df.sample(10, random_state=42)


# In[31]:


#creating a confusion matrix:

import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


from sklearn import metrics
def draw_cm( actual, predicted ):
    ##cret
    cm=metrics.confusion_matrix( actual, predicted, [1,0])
    sn.heatmap(cm,annot=True, fmt='.2f',
    xticklabels = ["Bad credit", "Good Credit"],
    yticklabels = ["Bad credit", "Good Credit"]  )
    plt.ylabel('True label')
    plt.xlabel('predicted label')
    plt.show()


# In[33]:


draw_cm( y_pred_df.actual, y_pred_df.predicted)


# In[34]:


#F-score

print( metrics.classification_report( y_pred_df.actual, y_pred_df.predicted))


# In[35]:



plt.figure(figsize=(8,6))
#plotting distribution of predicted probability values for bad credits
sn.distplot(y_pred_df[y_pred_df.actual==1]["predicted_prob"], kde=False, color='b', label='Bad Credit')
#Plotting distribution of predicted probability values for good credits
sn.distplot(y_pred_df[y_pred_df.actual==0]["predicted_prob"], kde=False, color='g', label='Good credits')
plt.legend()
plt.show()


# In[41]:


def draw_roc(actual, probs):
    #Obtain fpr, tpr, thresholds
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs, drop_intermediate = False)
    auc_score= metrics.roc_auc_score(actual, probs)
    plt.figure(figsize=(8, 6))
    #Plot the fpr and tpr values for different threshold values plt.plot(fpr, tpr, label-ROC curve (area %0.2f) % auc_score)
    plt.plot(fpr,tpr,label='ROC curve (area=%0.2f)' % auc_score)
    #draw a diagonal Line connecting the origin and top right most point
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0]) 
    plt.ylim([0.0, 1.05])
    #Setting x and y Labels
    plt.xlabel('False Positive Rate or [1 True Negative Rate]')
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()
    return fpr, tpr, thresholds


# In[42]:


fpr, tpr, thresholds= draw_roc(y_pred_df.actual,y_pred_df.predicted_prob)


# In[44]:


tpr_fpr= pd.DataFrame({ 'tpr': tpr, 'fpr': fpr, 'thresholds': thresholds } )
tpr_fpr['diff']=tpr_fpr.tpr - tpr_fpr.fpr 
tpr_fpr.sort_values('diff', ascending=False) [0:5]


# In[45]:


y_pred_df['predicted_new']= y_pred_df.predicted_prob.map( lambda x: 1 if x > 0.78 else 0)


# In[46]:


draw_cm(y_pred_df.actual, y_pred_df.predicted_new)


# In[47]:


print(metrics.classification_report(y_pred_df.actual, y_pred_df.predicted_new ))


# In[52]:


#cost-based approach:

def get_total_cost(actual, predicted, cost_FPs, cost_FNs):

  #Get the confusion matrix and calculate cost 
  cm=metrics.confusion_matrix(actual, predicted, [1,0])
  cm_mat= np.array(cm)
  return cm_mat[0,1] * cost_FNs + cm_mat [1,0] * cost_FPs
get_total_cost(y_pred_df.actual, y_pred_df.predicted_prob.map(lambda x: 1 if x> (each_prob/100) else 0), 1, 5)


# In[50]:


cost_df= pd.DataFrame(columns=['prob', 'cost'])


# In[51]:


idx =0

## Iterate cut-off probability values between 0.1 and 0.5 
for each_prob in range(10, 50):
    cost= get_total_cost(y_pred_df.actual, y_pred_df.predicted_prob.map(
    lambda x: 1 if x> (each_prob/100) else 0), 1, 5) 
    cost_df.loc[idx] = [(each_prob/100), cost]
    idx += 1


# In[53]:



cost_df.sort_values('cost', ascending= True) [0:5]


# In[56]:



y_pred_df[ 'predicted_using_cost'] = y_pred_df.predicted_prob.map(lambda x: 1 if x> 0.29 else 0)


# In[57]:



draw_cm(y_pred_df.actual, y_pred_df.predicted_using_cost)

