#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import time
import sys
import gc


# In[27]:


# Read all the given csv's
items = pd.read_csv('items.csv')
shops = pd.read_csv('shops.csv')
cats = pd.read_csv('item_categories.csv')
train = pd.read_csv('sales_train.csv')
# set index to ID to avoid droping it later
test  = pd.read_csv('test.csv').set_index('ID')


# In[28]:


# There is one entry when item count is very high.. 2000+.. Lets remove that
# Also there is one entry with a negative price and one with 3lakh! lets fill these with the mean
train = train[train.item_price<100000]
train = train[train.item_cnt_day<1001]
avg_price = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.mean()

# To replace a value at a particular column based on a filter
train.loc[train.item_price<0,'item_price'] = avg_price


# In[29]:


len(list(set(test.item_id) - set(test.item_id).intersection(set(train.item_id)))), len(list(set(test.item_id))), len(test), len(set(train.item_id)), len(set(test.item_id))


# In[30]:


# Lets build a matrix for training. Each shopid and itemid should be present for all months. So building a temp array like that
# The product function does the cartesian product of the combination
from itertools import product

matrix = []
cols = ['date_block_num','shop_id','item_id']
for i in range(34):
    sales = train[train.date_block_num==i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
    
len(matrix)


# In[31]:


matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols,inplace=True)


# In[32]:


len(matrix)


# In[33]:


# Lets get the training data in the monthly format

group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)


# In[34]:


len(group)


# In[35]:


# Now lets merge this group into the matrix we created and replace all values below 0 with 0 with clip
# Series.clip(lower=None, upper=None, axis=None, inplace=False, *args, **kwargs) - Assigns values outside boundary to boundary values

matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(lower=0)
                                .astype(np.float16))


# In[36]:


len(matrix)


# In[38]:


matrix.head(3)


# In[39]:


# change the data types in the test data
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)


# In[40]:


# We can now concat the test matrix array to the matrix
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) # 34 month


# In[41]:


matrix.tail(3)


# In[42]:


len(matrix)


# In[49]:





# In[50]:


# The items file has the item category id. So lets merge that with the matrix
items.drop(['item_name'],inplace=True,axis=1)
items.head(3)


# In[51]:


# Lets get some new features from shops and category files too
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
shops = shops[['shop_id','city_code']]

cats['split'] = cats['item_category_name'].str.split('-')
cats['type'] = cats['split'].map(lambda x: x[0].strip())
cats['type_code'] = LabelEncoder().fit_transform(cats['type'])
# if subtype is nan then type
cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
cats = cats[['item_category_id','type_code', 'subtype_code']]


# In[52]:


shops.head(3)


# In[53]:


cats.head(3)


# In[54]:


# Now merge all files with the matrix and change the data types
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
matrix = pd.merge(matrix, items, on=['item_id'], how='left')
matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
matrix['city_code'] = matrix['city_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['type_code'] = matrix['type_code'].astype(np.int8)
matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)


# In[55]:


matrix.head(3)


# In[56]:


matrix.tail(3)


# In[62]:


# Now we can create some lag features
tmp = matrix[['date_block_num','shop_id','item_id','item_cnt_month']]
tmp.tail(3)


# In[63]:


lags = [1,2,3,6]
for i in lags:
    shifted = tmp.copy()
    shifted.columns = ['date_block_num','shop_id','item_id', 'item_cnt_month_lag_'+str(i)]
    shifted['date_block_num'] += i
    matrix = pd.merge(matrix, shifted, on=['date_block_num','shop_id','item_id'], how='left')


# In[64]:


matrix.tail(3)


# In[65]:


# Lets check if there are any nulls
matrix[matrix.isnull().any(axis=1)]


# In[66]:


# Lot of null values.. lets fill them up with 0

def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)         
    return df

matrix = fill_na(matrix)


# In[67]:


matrix[matrix.isnull().any(axis=1)]


# In[83]:


# Now lets split the data into train and test for running the model on it and also validation

X_train = matrix[matrix.date_block_num < 34].drop(['item_cnt_month'], axis=1)
Y_train = matrix[matrix.date_block_num < 34]['item_cnt_month']


# In[84]:


X_train.tail(3)


# In[85]:


#X_validation = matrix[matrix.date_block_num == 33].drop('item_cnt_month',axis=1)
#Y_validation = matrix[matrix.date_block_num == 33]['item_cnt_month']


# In[86]:


#X_validation.tail(3)


# In[87]:


X_test = matrix[matrix.date_block_num == 34].drop(['item_cnt_month'], axis=1)


# In[88]:


X_test.tail(3)


# In[89]:


# Now that we have our train and test and validation datasets, lets run the algorithm

model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)


# In[91]:


model.fit(X_train,Y_train)


# In[92]:


predictions = model.predict(X_test)

submission_2_morefeatures = pd.DataFrame({'ID':test.index, 'item_cnt_month':predictions})

submission_2_morefeatures.to_csv('submission_GB_morefeatures.csv', index=False)


# In[93]:


len(submission_2_morefeatures)


# In[ ]:




