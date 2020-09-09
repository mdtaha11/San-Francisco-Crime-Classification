# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 17:00:31 2020

@author: Taha
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

sns.FacetGrid(train,hue="Category",size=5).map(plt.plot,"X","Y").add_legend()

top_crimes=train.Category.value_counts()
pos=np.arange(len(top_crimes))
plt.barh(pos,top_crimes.values)
plt.yticks(pos,top_crimes.index)


weekday_data=train["DayOfWeek"].value_counts()
order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
sns.boxplot(x=weekday_data.index,y=weekday_data.values,order=order)

train['Date']=train['Dates'].apply(lambda x:int(x.split("-")[2].split(" ")[0]))

#Graph plot to see which date has most number of crimes and which has least
plt.figure(figsize=[10,7])
plt.bar(train['Date'].value_counts().index,train['Date'].value_counts().values)

#Plotting boxplot to see crimes distributed in months
train['Month']=train['Dates'].apply(lambda x:int(x.split("-")[1]))
months_data=train["Month"].value_counts()
order = ["01", "02", "03", "04", "05", "06", "07", "08",  "09", "10",  "11" , "12" ]
sns.boxplot(x=months_data.index,y=months_data.values,order=order)

#
train['Hour']=train['Dates'].apply(lambda x:int(x.split(" ")[1].split(":")[0]))
plt.figure(figsize=[10,7])
plt.bar(train['Hour'].value_counts().index,train['Hour'].value_counts().values)

nigh_hours=['18','19','20','21','22','23','00','01','02','03','04','05']
f={True:1,False:0}
train['Night']=train['Hour'].apply(lambda x:int(x in nigh_hours )).map(f)
train.head()

train.Night.value_counts()
#Bivariate Analysis

#Label Encoding
data=[train,test]
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()

non_numeric_features = ['DayOfWeek', 'PdDistrict']

for feature in non_numeric_features:
    train[feature] = enc.fit_transform(list(train[feature]))



#Category vs Day of crime
new_data=pd.DataFrame(index=["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],columns=train.Category.unique())
new=train.iloc[:,1:4].drop("Descript",axis=1)
for j in train.Category.unique():
    for i in range(0,7):
        t=(train['Category']==j) & (train['DayOfWeek']==i)
        new_data[j][i]=(t==True).sum()
plt.figure(figsize=[20,14])
new_data.plot.bar(figsize=[20,14],stacked=True)
 
p={1:0,2:0,3:0,4:0,5:1,6:1,0:1}
train['zone']=train['DayOfWeek'].map(p)

train["Address"]=train["Address"].apply(lambda x: "Intersection" if "/" in x else ("Block"))


top_crimes=train.Category.value_counts()[:10]
plt.figure(figsize=(10,7))
pos=np.arange(len(top_crimes))
plt.barh(pos,top_crimes.values)
plt.yticks(pos,top_crimes.index)

top_addresses = train.Address.value_counts()[:20]
subset=train[train.Address.isin(top_addresses.index)&train.Category.isin(top_crimes.index)]

pos=np.arange(len(top_addresses)) 
plt.bar(pos,top_addresses.values)


threat_to_individuals = ['ARSON', 'ASSAULT', 'EXTORTION', 'KIDNAPPING', 'LARCENY/THEFT', 'BURGLARY', 'MISSING PERSON', 'ROBBERY', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE', 'VANDALISM', 'VEHICLE THEFT', 'WEAPON LAWS', 'FAMILY OFFENSES', 'OTHER OFFENSES']
violation_of_law = ['BAD CHECKS', 'BRIBERY', 'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'LIQUOR LAWS', 'LOITERING', 'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'RUNAWAY', 'SECONDARY CODES', 'SUSPICIOUS OCC', 'TRESPASS', 'WARRANTS']

train['Category'] = train['Category'].apply(lambda x: 'THREAT TO OTHERS LIFE' if x in threat_to_individuals else ('VIOLATION OF LAW' if x in violation_of_law  else 'NON-CRIMINAL'))    



features = ['Address','Month','Date','Hour','Minute','DayOfWeek','PdDistrict','X','Y']

 
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df =pd.DataFrame(enc.fit_transform(train[['Address']]).toarray())

train=train.join(enc_df)

train=train.drop(["Dates","Descript","Resolution","Address"],axis=1)
from sklearn.model_selection import train_test_split
train_set, valid_set, train_labels, valid_labels = train_test_split(
    train.drop("Category",axis=1), train['Category'], test_size=0.4, random_state=4327)
 

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(train_set, train_labels)
print(accuracy_score(valid_labels, xgb_classifier.predict(valid_set)))

 



