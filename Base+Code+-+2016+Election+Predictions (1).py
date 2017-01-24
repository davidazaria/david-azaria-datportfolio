
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns


# In[2]:

aca = pd.read_csv('Desktop/aca201520162017.csv')
votes = pd.read_csv('Desktop/votes.csv')


# In[50]:

aca.columns


# In[39]:

votes.columns


# In[4]:

votes.head(5)


# In[5]:

votes['Trump_Win'] = np.where(votes.Trump > votes.Clinton, 1, 0)


# In[6]:

votes.head(5)


# In[9]:

votes['Women'] = np.where(votes.SEX255214 > 50.0, 1, 0)


# In[67]:

votes['White_County'] = np.where(votes.White > 50.0, 1, 0)


# In[61]:

votes['Black_County'] = np.where(votes.Black > 50.0, 1, 0)


# In[62]:

votes['Hispanic_County'] = np.where(votes.Hispanic > 50.0, 1, 0)


# In[10]:

votes_2016 = votes[['Trump_Win', 'FIPS', 'state_abbr', 'county_name', 'population_change', 'White', 'Black', 'Hispanic',
                   'Edu_highschool', 'Edu_batchelors', 'Income', 'Poverty', 'Women']]


# In[11]:

votes_2016.head()


# In[22]:

features_votes_2016 = votes[['White', 'Black', 'Hispanic', 'Women', 'Edu_highschool', 'Edu_batchelors', 'Income', 'Poverty']]


# In[27]:

trump_wins_2016 = votes['Trump_Win']


# In[29]:

X = features_votes_2016[['White', 'Black', 'Hispanic', 'Women', 'Edu_highschool', 'Edu_batchelors', 'Income', 'Poverty']]
y = trump_wins_2016


# In[30]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[31]:

from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)


# In[32]:

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


# In[33]:

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(penalty='l2', C=10)


# In[34]:

logreg.fit(X_train_std, y_train) 
zip(trump_wins_2016, logreg.coef_[0])


# In[40]:

import numpy as np
from sklearn.model_selection import GridSearchCV
# gridsearch for hyperparameters
# the parameters we want to search in a dictionary
# use the parameter name from sklearn as the key
# and the possible values you want to test as the values
parameters = {'C': np.linspace(1e-5, 1e5, 100), 'class_weight': [None, 'balanced']}
logreg2 = LogisticRegression()           
clf = GridSearchCV(logreg2, parameters, cv=cv)
clf.fit(X_train_std, y_train)


# In[41]:

clf.best_params_


# In[42]:

clf.best_score_


# In[43]:

X_train.head()


# In[45]:

best_log = clf.best_estimator_


# In[46]:

pd.DataFrame({'features': X.columns, 'coefficients': best_log.coef_[0]})


# In[48]:

import matplotlib.pyplot as plt
from sk_modelcurves.learning_curve import draw_learning_curve
get_ipython().magic(u'matplotlib inline')


# In[49]:

draw_learning_curve(best_log, X_train_std, y_train, scoring='accuracy', cv=cv);


# In[81]:

merged_data = pd.merge(votes_2016, aca, left_on = 'FIPS', right_on = 'fips')


# In[82]:

merged_data.columns


# Will look at random forest here.

# In[74]:

votes_2016.groupby(['Women'])[['Trump_Win']].mean() ## Women = 1 pop of 


# In[ ]:



