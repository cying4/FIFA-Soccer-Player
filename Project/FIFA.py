#1.  Background and Data Description

# Our group chose to analyze data from the Fifa19. We used dataset
#  available on Kaggle,(https://www.kaggle.com/karangadiya/fifa19), 
#  which is scrapped from original source sofifa(https://sofifa.com/)
# The total size of the dataset is about 18206 rows with 88 fields.
# Fields include demographic information (name, age, height, nationality etc.) for 
# each player, the wage, evaluation(international reputation, overall, potential), 
# the Positions (LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, 
# RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB), the Scores for Attacking
#  (Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys), 
# the Scores for Skill (Dribbling, Curve, FKAccuracy, LongPassing, BallControl), 
# the Scores for Movement (Acceleration, SprintSpeed, Agility, Reactions, Balance),
# the Scores for Power (ShotPower, Jumping, Stamina, Strength, LongShots), 
# the Scores for Mentality (Aggression, Interceptions, Positioning, Vision, Penalties, Composure),
# the Scores for defending (Marking, StandingTackle, SlidingTackle), the Scores for
# Goalkeeping (GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes), Release Clause.

#2. SMART Question
# How is wage distributed? Can we model wage with soccer players’ features? 
# Is player’s overall rating calculated from a formula of abilities rating? Or is it determined experimentally?
# Is preferred foot affected by his abilities rating?
# Is the international reputation of players affected by theirs scores?
# Can we predict the preferred foot form score values?

#3. Data Preprocess and Cleaning

# %% Import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#%%
dirpath = os.getcwd() 
filepath = os.path.join( dirpath ,'data.csv')
fifa = pd.read_csv(filepath)  
fifa.head()

# %% Data Cleaning
#To check whether index is unique and drop NA
print('\n',fifa.head(),'\n')
fifa.index
print(fifa.index.is_unique)
fifa.columns
fifa = fifa.rename(columns={'Unnamed: 0': 'id', 'Club Logo':'Club_Logo', 'Preferred Foot':'Preferred_Foot', 'International Reputation':'International_Reputation', 'Weak Foot':'Weak_Foot', 'Skill Moves':'Skill_Moves', 'Work Rate':'Work_Rate', 'Body Type':'Body_Type', 'Real Face':'Real_Face', 'Jersey Number':'Jersey_Number', 'Loaned From':'Loaned_From', 'Contract Valid Until':'Contract_Valid_Until', 'Release Clause':'Release_Clause'})
fifa = fifa.drop(columns=['Loaned_From'])
fifa = fifa.dropna()

# %% 4. Linear model of wage
fifa['Wage'] = fifa['Wage'].map(lambda x: x[1:][:-1])
fifa['Wage'] = pd.to_numeric(fifa['Wage'])

#%%
plt.hist(fifa['Wage'], label='fifawage')
plt.xlabel('Wage')
plt.ylabel('Frequency')
plt.show()
#From the plot we can see salaries are right skewed

#%%
#from the plot we can see high-level salaries are much less than the lower-level salary, and first we look into the low-level salary which is less than 200K and 50K  and 5K
#Salaries are right-skewed distributed in each group, and the frequencies become less and less as salaries go up
plt.hist(fifa[fifa['Wage']<200]['Wage'], label='fifawage',edgecolor='black', linewidth=1.2)
plt.xlabel('Wage under 200')
plt.ylabel('Frequency')

#%%
plt.hist(fifa[fifa['Wage']<50]['Wage'], label='fifawage',edgecolor='black', linewidth=1.2)
plt.xlabel('Wage under 50')
plt.ylabel('Frequency')

#%%
plt.hist(fifa[fifa['Wage']<20]['Wage'], label='fifawage',edgecolor='black', linewidth=1.2)
plt.xlabel('Wage under 20')
plt.ylabel('Frequency')

# %%
#Then we look into the high salaries, we can see the frequencies in each level are sparse and scattered, and also some high outlier exist
plt.hist(fifa[fifa['Wage']>200]['Wage'], label='fifawage',edgecolor='black', linewidth=1.2)
plt.xlabel('Wage above 200')
plt.ylabel('Frequency')

# %%
#BMI is an indicator of health condition, and we increase a new variable bmi(=height(m)/weight(kg)^2)
#fh=open(filepath, encoding="utf8")
#for aline in fh.readlines(): # readlines creates a list of elements; each element is a line in the txt file, with an ending line return character. 
#  h = int(aline.split(',').split("'")[0])+int(aline.split(',').split("'")[1])*0.1

#fifa['Weight']=0.453592*int(fifa['Weight'].strip('lbs'))

w2=[0.453592*int(w.strip('lbs')) for w in fifa['Weight']]
h2=[int(h.split("'")[0])+int(h.split("'")[1])*0.1 for h in fifa['Height']]
# for i in range(len(fifa['Weight'])):
fifa['Weight'] = fifa['Weight'].apply(lambda x: 0.453592*int(x.strip('lbs')))
fifa['Height'] = fifa['Height'].apply(lambda x: int(x.split("'")[0]) + int(x.split("'")[1])*0.1)
fifa['BMI'] = fifa['Weight'] / pow(fifa['Height']*30.48/100, 2)
# for m in range(len(w2)):
#     bmi=w2[m]/pow(h2[m]*30.48/100,2)
#     fifa['BMI'][m]=bmi

# fifa['BMI'].head()

# %%
plt.hist(fifa['Age'], label='fifaage',edgecolor='black', linewidth=1.2)
#From the age distribution we can see age 20 to 30 are the majority
#We expect there will be 'golden ages' for athletes, which means below or above the certain range of age, soccer players would not be at their time of best performance
#fifa['Age^2']=None
#age=[age for age in fifa['Age']]
#for n in list(range(len(age))):
#    agesquare=pow(age[n],2)
#    fifa['Age^2'][n]=agesquare
#fifa['Age^2'].head()
#We can have a look at plot of Wage & Age. The plot is normal distributed.
fig, axis = plt.subplots()
axis.plot(fifa.Age, fifa.Wage,color='g', linestyle="", marker="o", markersize=3)
plt.xlabel("Age")
plt.ylabel("Wage")
plt.title("Wage vs Age")
plt.show()

# %%
plt.hist(fifa['Jersey_Number'], label='fifanumber',edgecolor='black', linewidth=1.2)
#We can see Jersey Number under 40 are the majority

#%%
#Many fans and soccer believe there are 'lucky numbers'. And there is a rumor going like that numbers in a range tend to be picked as lucky numbers
#To test whether it makes sense, we got jursey number squared  
#fifa['Jersey_Number_Squared']=None
#number=[number for number in fifa['Jersey_Number']]
#for j in list(range(len(number))):
#    numbersquare=pow(number[j],2)
#    fifa['Jersey_Number_Squared'][j]=numbersquare
#fifa['Jersey_Number_Squared'].head()
fig, axis = plt.subplots()
axis.plot(fifa.Jersey_Number, fifa.Wage,color='g', linestyle="", marker="o", markersize=3)
plt.xlabel("Jersey Number")
plt.ylabel("Wage")
plt.title("Wage vs Jersey Number")
plt.show()
#From the plot we can see valuable players are accumulated around 10

#%%
plt.hist(fifa[fifa['Jersey_Number']<40]['Jersey_Number'], label='fifanumber',edgecolor='black', linewidth=1.2)
fig, axis = plt.subplots()
axis.plot(fifa[fifa['Jersey_Number']<40].Jersey_Number, fifa[fifa['Jersey_Number']<40].Wage,color='g', linestyle="", marker="o", markersize=3)
plt.xlabel("Jersey Number")
plt.ylabel("Wage")
plt.title("Wage vs Jersey Number")
plt.show()

# %%
#Then we build the liner model to see which variables could have significant effect on wages
modelwage = ols(formula='Wage ~ Age+Overall+Potential+Special+C(International_Reputation)+C(Weak_Foot)+C(Skill_Moves)+C(Work_Rate)+C(Body_Type)+C(Position)+Jersey_Number+BMI+Crossing+Finishing+HeadingAccuracy+ShortPassing+Volleys+Dribbling+Curve+FKAccuracy+LongPassing+BallControl+Acceleration+SprintSpeed+Agility+Reactions+Balance+ShotPower+Jumping+Stamina+Strength+LongShots+Aggression+Interceptions+Positioning+Vision+Penalties+Composure+Marking+StandingTackle+SlidingTackle+GKDiving+GKHandling+GKKicking+GKPositioning+GKReflexes', data=fifa).fit()
print(modelwage.summary())

#%%
#We drop the variable with large p value until all p values are down to less than 5%
#(For categorical variables, though some dimesnions have large p value(>5%), we still keep them when other dimensions are signficant)
modelwage2 = ols(formula='Wage ~ Potential+Special+C(International_Reputation)+C(Skill_Moves)+C(Work_Rate)+Age+Overall+Finishing+Volleys+Reactions+Balance+Positioning+Vision+Composure+SlidingTackle+GKDiving', data=fifa).fit()
print(modelwage2.summary())


#%% 5. Linear model of overall rating 
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

# %% Preprocess

overall_skills = fifa.loc[:,'LS':'GKReflexes']
overall_skills.insert(0, 'Overall', fifa['Overall'])

for i in overall_skills.columns:
    if type(overall_skills[i][0]) == str:
        overall_skills[i] = overall_skills[i].apply(lambda x: int(x.split('+')[0])+int(x.split('+')[1]))

dfChkBasics(overall_skills)

# %%

# a='+'.join(overall_skills.columns)
a='LS+ST+RS+LW+LF+CF+RF+RW+LAM+CAM+RAM+LM+LCM+CM+RCM+RM+LWB+LDM+CDM+RDM+RWB+LB+LCB+CB+RCB+RB+Crossing+Finishing+HeadingAccuracy+ShortPassing+Volleys+Dribbling+Curve+FKAccuracy+LongPassing+BallControl+Acceleration+SprintSpeed+Agility+Reactions+Balance+ShotPower+Jumping+Stamina+Strength+LongShots+Aggression+Interceptions+Positioning+Vision+Penalties+Composure+Marking+StandingTackle+SlidingTackle+GKDiving+GKHandling+GKKicking+GKPositioning+GKReflexes'
# a= 'Crossing+Finishing+HeadingAccuracy+ShortPassing+Volleys+Dribbling+Curve+FKAccuracy+LongPassing+BallControl+Acceleration+SprintSpeed+Agility+Reactions+Balance+ShotPower+Jumping+Stamina+Strength+LongShots+Aggression+Interceptions+Positioning+Vision+Penalties+Composure+Marking+StandingTackle+SlidingTackle+GKDiving+GKHandling+GKKicking+GKPositioning+GKReflexes'
osModel = ols(formula='Overall ~' + a, data=overall_skills).fit()
print( osModel.summary() )

b='LS+ST+RS+LW+LF+CF+RF+RW+LAM+CAM+RAM+LM+LCM+CM+RCM+RM+LWB+LDM+CDM+RDM+RWB+LB+LCB+CB+RCB+RB+Crossing+Finishing+HeadingAccuracy+ShortPassing+Volleys+Dribbling+Curve+FKAccuracy+LongPassing+BallControl+Acceleration+SprintSpeed+Agility+Reactions+Balance+ShotPower+Jumping+Stamina+Strength+LongShots+Aggression+Interceptions+Positioning+Vision+Penalties+Composure+Marking+StandingTackle+SlidingTackle+GKDiving+GKHandling+GKKicking+GKPositioning+GKReflexes'
# a= 'Crossing+Finishing+HeadingAccuracy+ShortPassing+Volleys+Dribbling+Curve+FKAccuracy+LongPassing+BallControl+Acceleration+SprintSpeed+Agility+Reactions+Balance+ShotPower+Jumping+Stamina+Strength+LongShots+Aggression+Interceptions+Positioning+Vision+Penalties+Composure+Marking+StandingTackle+SlidingTackle+GKDiving+GKHandling+GKKicking+GKPositioning+GKReflexes'
osModel1 = ols(formula='Overall ~' + b + '-1', data=overall_skills).fit()
print( osModel1.summary() )

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

# %% 6. Logistics Regression of preferred foot
# To predict the preferred foot by using logistics regression

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

# %% logitstic regression
from sklearn.linear_model import LogisticRegression

admitlogit = LogisticRegression()  # instantiate
admitlogit.fit(x_trainAdmit, y_trainAdmit)
admitlogit.predict(x_testAdmit)
print('Logit model accuracy (with the test set):', round(admitlogit.score(x_testAdmit, y_testAdmit),3))

#%%
print(admitlogit.predict_proba(x_trainAdmit[:8]))
print(admitlogit.predict_proba(x_testAdmit[:8]))

#%%
y_true, y_pred = y_testAdmit, admitlogit.predict(x_testAdmit)
print(classification_report(y_true, y_pred))
a = classification_report(y_true, y_pred)
accuracy = a.split()[a.split().index('accuracy')+1]

print('The accuracy of logistics regression is', accuracy)

cof = admitlogit.coef_.tolist()[0]
low2highc = sorted(cof)
name = []
coff = []
for i in range(len(low2highc)):
  x = len(low2highc)-i-1
  name.append(xadmit.columns[cof.index(low2highc[x])])
  coff.append(str(round(low2highc[x]*100, 3)) + '%')

log = pd.DataFrame(name, coff)
  # print(xadmit.columns[cof.index(low2highc[x])], str(round(low2highc[x]*100, 3)) + '%')


#%% 7. KNN of reputation.
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

#%% 8. KNN of Preferred foot
##Preferred foot-ability

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


# %%
