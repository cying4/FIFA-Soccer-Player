#%% [markdown]
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('classic')
dirpath = os.getcwd() 
filepath = os.path.join( dirpath ,'/Users/KJ/Desktop/6103 proj/data.csv')
fifa = pd.read_csv(filepath)  
fifa.head()

# %%
#To check whether index is unique and drop NA
print('\n',fifa.head(),'\n')
fifa.index
print(fifa.index.is_unique)
fifa.columns
fifa = fifa.rename(columns={'Unnamed: 0': 'id', 'Club Logo':'Club_Logo', 'Preferred Foot':'Preferred_Foot', 'International Reputation':'International_Reputation', 'Weak Foot':'Weak_Foot', 'Skill Moves':'Skill_Moves', 'Work Rate':'Work_Rate', 'Body Type':'Body_Type', 'Real Face':'Real_Face', 'Jersey Number':'Jersey_Number', 'Loaned From':'Loaned_From', 'Contract Valid Until':'Contract_Valid_Until', 'Release Clause':'Release_Clause'})
fifa = fifa.drop(columns=['Loaned_From'])
fifa = fifa.dropna()

# %%
fifa['Wage'] = fifa['Wage'].map(lambda x: x[1:][:-1])
fifa['Wage'] = pd.to_numeric(fifa['Wage'])


# %%
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score

#%%
##knn model on predict reputation.
##variables: skill scores
xfifa=fifa.iloc[:,53:87]
yfifa=fifa['International_Reputation']
xsfifa = pd.DataFrame( scale(xfifa), columns=xfifa.columns ) 
ysfifa = yfifa.copy()
X_train, X_test, y_train, y_test = train_test_split(xsfifa, ysfifa, test_size = 0.25, random_state=2019)


#%%
rmse_val = [] 
for K in range(15):
    K=K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train) 
    pred=model.predict(X_test) 
    error = sqrt(mean_squared_error(y_test,pred))
    rmse_val.append(error)
    print('RMSE value for k= ' , K , 'is:', error)


#%%
curve = pd.DataFrame(rmse_val)
curve.plot()



#%%
k=9
knn_scv = KNeighborsClassifier(n_neighbors=k)
scv_results = cross_val_score(knn_scv, xsfifa, ysfifa, cv=5)
print(scv_results) 
np.mean(scv_results)

#%%
import pandas as pd
y_true=y_test
knn_scv = neighbors.KNeighborsRegressor(n_neighbors = 15)
model.fit(X_train, y_train) 
y_pred=model.predict(X_test) 
pd.crosstab(y_true,round(pd.Series(y_pred)),rownames=['ACTUAL'],colnames=['PRED'])

#%%
##Preferred foot-ability
fifa['Weight'] = fifa['Weight'].map(lambda x: x[:][:-2])
fifa['Weight'] = pd.to_numeric(fifa['Weight'])

#%%
def P_foot (row):
   if row['Preferred_Foot'] == 'Left' :
      return 0
   if row['Preferred_Foot'] == 'Right':
      return 1

fifa['P_foot']=fifa.apply (lambda row: P_foot(row), axis=1)

#%%
xfifa1=fifa.iloc[:,53:87]
yfifa1=fifa['P_foot']
xsfifa1 = pd.DataFrame( scale(xfifa1), columns=xfifa1.columns ) 
ysfifa1 = yfifa1.copy()
X_train1, X_test1, y_train1, y_test1 = train_test_split(xsfifa1, ysfifa1, test_size = 0.25, random_state=2019)

#%%
rmse_val = [] 
for K in range(15):
    K=K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train1, y_train1) 
    pred=model.predict(X_test1) 
    error = sqrt(mean_squared_error(y_test1,pred))
    rmse_val.append(error)
    print('RMSE value for k= ' , K , 'is:', error)

#%%
curve = pd.DataFrame(rmse_val)
curve.plot()

#%%
##k=15
k=15
knn_scv1 = KNeighborsClassifier(n_neighbors=k)
scv_results1 = cross_val_score(knn_scv1, xsfifa1, ysfifa1, cv=5)
print(scv_results1) 
np.mean(scv_results1)

#%%
from sklearn.metrics import classification_report
y_true=y_test1
model = neighbors.KNeighborsRegressor(n_neighbors = 15)
model.fit(X_train1, y_train1) 
y_pred=model.predict(X_test1) 

print(classification_report(y_true, y_pred.round()))

#%%
import pandas as pd
y_true=y_test1
model = neighbors.KNeighborsRegressor(n_neighbors = 15)
model.fit(X_train1, y_train1) 
y_pred=model.predict(X_test1) 
pd.crosstab(y_true,round(pd.Series(y_pred)),rownames=['ACTUAL'],colnames=['PRED'])





#%%


























