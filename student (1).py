#!/usr/bin/env python
# coding: utf-8

# ## Final Project Submission
# 
# Please fill out:
# * Student name: Fahmi Afri
# * Student pace: art time 
# * Scheduled project review date/time: 30/10/2022
# * Instructor name: Hardik Idnani
# * Blog post URL:https://github.com/fhmiafri/dsc-phase-2-project
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler,PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.formula.api import ols


# In[2]:


data=pd.read_csv(r'C:\Users\fahmi\Downloads\kc_house_data.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# 1.    ID - Identification Number
# 2.  Date - Date sold
# 3. price - Sale price
# 4. bedrooms - number of bedrooms
# 5. bathrooms - No of Bathrooms. 0.25 - Only Toilet 0.50 - Toilet and Sink 0.75 - Toilet, Sink and Shower 1 - Toilet, Sink, Shower and Bathtub
# 6.  sqft_liv - Total living/built up area in square feet.
# 7.  sqft_lot - Total lot area(includes living area and other structures such as garages, swimming pools, and sheds) in square feet.
# 8.  floors - Number of floors.
# 9. waterfront - '1' if the property has a waterfront, 'O' if not.
# 10. view - how good the view of the property on a index of (0 - min - 4 - max)
# 11. condition - Condition of the house, ranked from 1 to 5(1 min , 5 - max)
# 12. grade - classification of house based on the quality of construction and materials used measured on a index of(1-13)
# 13. sqft_above - area above the basemeant in square feet.
# 14. sqft basmt - area below the basemeant in square feet.
# 15. yr_built - year built
# 16. yr renov - year renovated (if never renovated it's zero)
# 17. zipcode - Zipcode of the house
# 18. lat - Latitude of the house.
# 19. long - longitude of the house
# 20. squft_liv15 - avearge of the living area of nearest 15 houses in square feet
# 21. squft lot15 - avearge of the lot area of nearest 15 houses in square feetID - Identification Number
# 

# In[5]:


#selecting only the importants
data2 = data [['price', 'bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition', 'grade' ]]


# In[6]:


data2


# In[7]:


data2.describe()


# In[8]:


#making sure they are not in scientific notation
pd.options.display.float_format = lambda x : '{:.0f}'.format(x) if int(x) == x else '{:,.2f}'.format(x)


# In[9]:


#checking for missing values
data2.isnull().sum()


# In[10]:


data2.describe()


# In[11]:


#sorting according to prices
data2.sort_values('price')


# In[12]:


plt.figure(figsize=(12,8))
sns.distplot(data2['price'])


# In[13]:


plt.figure(figsize=(12,8))
sns.distplot(data2['sqft_living'])


# In[14]:


data2.columns


# In[15]:


plt.figure(figsize=(10,6))
sns.countplot(data2['bedrooms'])


# In[16]:


plt.figure(figsize=(12,8))
sns.boxplot(x='bedrooms' , y='price',data=data2)


# In[17]:


plt.figure(figsize=(12,8))
sns.boxplot(x='bathrooms' , y='price',data=data2)


# In[18]:


fig = sns.boxplot(x='grade',y='price',data=data2)
fig.axis(ymin=0,ymax=5000000);


# In[19]:


data2.columns


# In[20]:


data2.info()


# In[21]:


sns.countplot(x='floors', data=data2, palette='Set2')


# In[22]:


plt.figure(figsize =(12,8))
sns.scatterplot(x='sqft_living', y='price', data=data2)


# In[23]:


plt.figure(figsize =(12,8))
sns.scatterplot(x='sqft_lot', y='price', data=data2)


# In[24]:


data2.columns


# In[25]:


data2.corr()


# In[26]:


#visualize the corr between all features
cor=data2.corr()
plt.figure(figsize=(20,15))
sns.heatmap(cor,annot=True)
plt.show


# In[27]:


data2.describe()


# In[28]:


#identifying std/zscore
zscore = np.abs(stats.zscore(data2))
zscore 


# In[29]:


#removing outlier as outlier boundaries from the mean is std=3
data3 = data2[(zscore < 3).all(axis=1)]
data3


# ## 2nd attempt after removing outlier
# 

# In[30]:


#after removing outlier
plt.figure(figsize=(12,8))
sns.distplot(data3['price'])


# In[31]:


plt.figure(figsize=(12,8))
sns.distplot(data3['sqft_living'])


# In[32]:


plt.figure(figsize=(10,6))
sns.countplot(data3['bedrooms'])


# In[33]:


plt.figure(figsize=(12,8))
sns.boxplot(x='bedrooms' , y='price',data=data3)


# In[34]:


plt.figure(figsize=(12,8))
sns.boxplot(x='bathrooms' , y='price',data=data3)


# In[35]:


fig = sns.boxplot(x='grade',y='price',data=data3)
fig.axis(ymin=0,ymax=5000000);


# In[36]:


sns.countplot(x='floors', data=data3, palette='Set2')


# In[37]:


plt.figure(figsize =(12,8))
sns.scatterplot(x='sqft_living', y='price', data=data3)


# In[38]:


plt.figure(figsize =(12,8))
sns.scatterplot(x='sqft_lot', y='price', data=data3)


# In[39]:


data3.corr()


# In[40]:


cor=data3.corr()
plt.figure(figsize=(20,15))
sns.heatmap(cor,annot=True)
plt.show


# In[41]:


data3.describe()


# In[42]:


#making the bathrooms dataset easy to access  
data3.loc[data3['bathrooms'] <= 1,'bathrooms'] = 1
data3.loc[(data3['bathrooms'] > 1) & (data3['bathrooms'] <= 2),'bathrooms'] = 2
data3.loc[(data3['bathrooms'] > 2) & (data3['bathrooms'] <= 3),'bathrooms'] = 3
data3.loc[(data3['bathrooms'] > 3) & (data3['bathrooms'] <= 4),'bathrooms'] = 4
data3.loc[(data3['bathrooms'] > 4) & (data3['bathrooms'] <= 5),'bathrooms'] = 5


# In[43]:


#transform data to near normal distribution for easy intepretation
data_log = pd.DataFrame([])
   
data_log['price_log'] = np.log(data3['price'])
data_log['bedrooms_log'] = (data3['bedrooms'])
data_log['bathrooms_log'] = (data3['bathrooms'])
data_log['sqft_living_log'] = np.log(data3['sqft_living'])
data_log['sqft_lot_log'] = np.log(data3['sqft_lot'])
data_log['condition_log'] = (data3['condition'])
data_log['grade_log'] = (data3['grade'])
data_log['floors_log'] = (data3['floors'])
data_log.hist(figsize  = [18,8]);


# In[44]:


data_log['bathrooms_log'] = data_log['bathrooms_log'].astype('Int64')


# In[45]:


#making sure all data is occupied
data_log = pd.get_dummies(data_log, columns=['bedrooms_log', 'bathrooms_log', 'condition_log','grade_log'])
data_log.columns


# In[46]:


data_log.info()


# In[47]:


y_result='price_log'
x_features = ['sqft_living_log', 'sqft_lot_log', 'floors_log',
       'bedrooms_log_1', 'bedrooms_log_2', 'bedrooms_log_3', 'bedrooms_log_4',
       'bedrooms_log_5', 'bedrooms_log_6', 'bathrooms_log_1',
       'bathrooms_log_2', 'bathrooms_log_3', 'bathrooms_log_4',
       'bathrooms_log_5', 'condition_log_2', 'condition_log_3',
       'condition_log_4', 'condition_log_5', 'grade_log_5', 'grade_log_6',
       'grade_log_7', 'grade_log_8', 'grade_log_9', 'grade_log_10',
       'grade_log_11']

x_features='+'.join(x_features)
formula=y_result + '~' + x_features
model=ols(formula=formula, data=data_log).fit()


# In[48]:


model.summary()


# In[49]:


x_features


# In[50]:


formula


# In[51]:


from sklearn.model_selection import train_test_split


# In[52]:


X= data_log.drop('price_log', axis=1)
y = data_log['price_log']


# In[53]:


len(y)


# In[54]:


##spliting datas for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[55]:


X_train


# In[56]:


from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score


# In[57]:


#apply and modeling the train test set
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)

#identifying the residuals
train_residuals = y_hat_train - y_train
test_residuals = y_hat_test - y_test

train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
print('Train Mean Squared Error:', train_mse)
print('Test Mean Squared Error:', test_mse)


# In[58]:


#performing cv test and comparing with train and test MSE
mse = make_scorer(mean_squared_error)
cv_10_results = cross_val_score(linreg, X, y, cv=10, scoring=mse)
cv_10_results.mean()


# In[59]:


model.summary()


# In[ ]:




