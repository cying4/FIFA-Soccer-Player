#%% [markdown]
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('classic')
dirpath = os.getcwd() 
filepath = os.path.join( dirpath ,'data.csv')
fifa = pd.read_csv(filepath)  
fifa.head()

#%%
# Standard quick checks
def dfChkBasics(dframe): 
  cnt = 1  
  try:
    print(str(cnt)+': info(): ')
    print(dframe.info()) 
  except: pass

  cnt+=1
  print(str(cnt)+': describe(): ')
  print(dframe.describe())

  cnt+=1
  print(str(cnt)+': dtypes: ')
  print(dframe.dtypes)

  try:
    cnt+=1
    print(str(cnt)+': columns: ')
    print(dframe.columns)
  except: pass

  cnt+=1
  print(str(cnt)+': head() -- ')
  print(dframe.head())

  cnt+=1
  print(str(cnt)+': shape: ')
  print(dframe.shape)

  # cnt+=1
  # print(str(cnt)+': columns.value_counts(): ')
  # print(dframe.columns.value_counts())

def dfChkValueCnts(dframe):
  cnt = 1
  for i in dframe.columns :
    print(str(cnt)+':', i, 'value_counts(): ')
    print(dframe[i].value_counts())
    cnt +=1

# %%
#To check whether index is unique and drop NA
print('\n',fifa.head(),'\n')
fifa.index
print(fifa.index.is_unique)
fifa.columns
fifa = fifa.rename(columns={'Unnamed: 0': 'id', 'Club Logo':'Club_Logo', 'Preferred Foot':'Preferred_Foot', 'International Reputation':'International_Reputation', 'Weak Foot':'Weak_Foot', 'Skill Moves':'Skill_Moves', 'Work Rate':'Work_Rate', 'Body Type':'Body_Type', 'Real Face':'Real_Face', 'Jersey Number':'Jersey_Number', 'Loaned From':'Loaned_From', 'Contract Valid Until':'Contract_Valid_Until', 'Release Clause':'Release_Clause'})
fifa = fifa.drop(columns=['Loaned_From'])
fifa = fifa.dropna()

# %% Preprocess
from statsmodels.formula.api import ols

overall_skills = fifa.loc[:,'LS':'GKReflexes']
overall_skills.insert(0, 'Overall', fifa['Overall'])

fifa['Wage'] = fifa['Wage'].map(lambda x: x[1:][:-1])
fifa['Wage'] = pd.to_numeric(fifa['Wage'])

for i in overall_skills.columns:
    if type(overall_skills[i][0]) == str:
        overall_skills[i] = overall_skills[i].apply(lambda x: int(x.split('+')[0])+int(x.split('+')[1]))

# %%

# sns.set()
# sns.pairplot(overall_skills.iloc[:,2:])
# a='+'.join(overall_skills.columns)
a='LS+ST+RS+LW+LF+CF+RF+RW+LAM+CAM+RAM+LM+LCM+CM+RCM+RM+LWB+LDM+CDM+RDM+RWB+LB+LCB+CB+RCB+RB+Crossing+Finishing+HeadingAccuracy+ShortPassing+Volleys+Dribbling+Curve+FKAccuracy+LongPassing+BallControl+Acceleration+SprintSpeed+Agility+Reactions+Balance+ShotPower+Jumping+Stamina+Strength+LongShots+Aggression+Interceptions+Positioning+Vision+Penalties+Composure+Marking+StandingTackle+SlidingTackle+GKDiving+GKHandling+GKKicking+GKPositioning+GKReflexes'
# a= 'Crossing+Finishing+HeadingAccuracy+ShortPassing+Volleys+Dribbling+Curve+FKAccuracy+LongPassing+BallControl+Acceleration+SprintSpeed+Agility+Reactions+Balance+ShotPower+Jumping+Stamina+Strength+LongShots+Aggression+Interceptions+Positioning+Vision+Penalties+Composure+Marking+StandingTackle+SlidingTackle+GKDiving+GKHandling+GKKicking+GKPositioning+GKReflexes'
osModel = ols(formula='Overall ~' + a, data=overall_skills).fit()
print( osModel.summary() )

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Remove Variables that P-value lower than 0.05:

# %% Rebuild linear model
d= 'Crossing+Finishing+HeadingAccuracy+ShortPassing+Volleys+Dribbling+Curve+FKAccuracy+LongPassing+BallControl+Acceleration+SprintSpeed+Agility+Reactions+Balance+ShotPower+Jumping+Stamina+Strength+LongShots+Aggression+Interceptions+Positioning+Vision+Penalties+Composure+Marking+StandingTackle+SlidingTackle+GKDiving+GKHandling+GKKicking+GKPositioning+GKReflexes'
osModel4 = ols(formula='Overall ~' + d + '-1', data=overall_skills).fit()
print( osModel4.summary() )
# remove variable that P-value greater 0.05

# %% Rebuild linear model
e = 'Finishing+HeadingAccuracy+ShortPassing+Volleys+FKAccuracy+LongPassing+BallControl+Acceleration+SprintSpeed+Agility+Reactions+Balance+ShotPower+Jumping+Stamina+Strength+LongShots+Aggression+Interceptions+Positioning+Vision+Penalties+Composure+Marking+StandingTackle+SlidingTackle+GKDiving+GKHandling+GKKicking+GKPositioning+GKReflexes'
osModel5 = ols(formula='Overall ~' + e + '-1', data=overall_skills).fit()
print( osModel5.summary() )
# remove variable that P-value greater 0.05

# %% Rebuild linear model
f = 'Finishing+HeadingAccuracy+ShortPassing+Volleys+FKAccuracy+LongPassing+BallControl+Acceleration+SprintSpeed+Agility+Reactions+Balance+ShotPower+Jumping+Strength+LongShots+Aggression+Interceptions+Positioning+Vision+Penalties+Composure+Marking+StandingTackle+SlidingTackle+GKDiving+GKHandling+GKKicking+GKPositioning+GKReflexes'
osModel6 = ols(formula='Overall ~' + f +'-1', data=overall_skills).fit()
print( osModel6.summary() )

# %% Logistics Regression
# To predict the preferred foot by using logistics regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

xadmit = fifa[['Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy','LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower','Jumping','Stamina','Strength','LongShots','Aggression','Interceptions','Positioning','Vision','Penalties','Composure','Marking','StandingTackle','SlidingTackle','GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes']]
yadmit = fifa['Preferred_Foot']
print(type(xadmit))
print(type(yadmit))

x_trainAdmit, x_testAdmit, y_trainAdmit, y_testAdmit = train_test_split(xadmit, yadmit)

footlogit = LogisticRegression()
footlogit.fit(x_trainAdmit, y_trainAdmit)
footlogit.predict(x_testAdmit)
print('Logit model accuracy (with the test set):', footlogit.score(x_testAdmit, y_testAdmit))

print('x_trainAdmit type',type(x_trainAdmit))
print('x_trainAdmit shape',x_trainAdmit.shape)
print('x_testAdmit type',type(x_testAdmit))
print('x_testAdmit shape',x_testAdmit.shape)
print('y_trainAdmit type',type(y_trainAdmit))
print('y_trainAdmit shape',y_trainAdmit.shape)
print('y_testAdmit type',type(y_testAdmit))
print('y_testAdmit shape',y_testAdmit.shape)

# %%
from sklearn.linear_model import LogisticRegression

admitlogit = LogisticRegression()  # instantiate
admitlogit.fit(x_trainAdmit, y_trainAdmit)
admitlogit.predict(x_testAdmit)
print('Logit model accuracy (with the test set):', round(admitlogit.score(x_testAdmit, y_testAdmit),3))

#%%
print(admitlogit.predict_proba(x_trainAdmit[:8]))
print(admitlogit.predict_proba(x_testAdmit[:8]))

#%%
from sklearn.metrics import classification_report
y_true, y_pred = y_testAdmit, admitlogit.predict(x_testAdmit)
print(classification_report(y_true, y_pred))
a = classification_report(y_true, y_pred)
accuracy = a.split()[a.split().index('accuracy')+1]

print('The accuracy of logistics regression is', accuracy)
# %%
