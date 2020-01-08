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
#%% [markdown]
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
plt.style.use('classic')
dirpath = os.getcwd() 
filepath = os.path.join(dirpath ,'data.csv')
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

