# Dated : 02-06-2019
#<==================================================================================================>
#<-----------------------------Import Libraries------------------------------------------------------>
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#====================================================================================================>
#<-----------------------------Load the Dataset------------------------------------------------------->

data = pd.read_csv('train.csv',encoding = 'latin-1')
data = data.rename(columns = lambda x:x.strip().lower())
data.head()

#<===================================================================================================>
#----------------------------Data Wrangling----------------------------------------------------------->

data = data[['pclass','sex','age','sibsp','fare','embarked','survived']]
data['sex'] = data['sex'].map({'male':0,'female':1})
data['age'] = pd.to_numeric(data['age'], errors='coerce')
data['age'] = data['age'].fillna(np.mean(data['age']))
#====================================================================================================>
#--------------------Remove the Dummy Variable------------------------------------------------------->
dummies = pd.get_dummies(data['embarked'])
data = pd.concat([data,dummies], axis=1)
data = data.drop(['embarked'],axis = 1)

#<===================================================================================================>
#---------------------Data Separation---------------------------------------------------------------->
X = data.drop(['survived'],axis = 1)
y = data['survived']

#=====================================================================================================>
#---------------------------Rescaling the Independent Feature Variables------------------------------->

sc = MinMaxScaler(feature_range=(0,1))
X_scaled = sc.fit_transform(X)

#======================================================================================================>
#---------------------------Model Training------------------------------------------------------------->

model = LogisticRegression(C=1)
model.fit(X_scaled,y)

#<=====================================================================================================>
#-------------------Save the Model--------------------------------------------------------------------->
pickle.dump(model, open("Titanic_Survial_Prediction_Web/Titanic_Survial_Prediction_Web/Titanic_Survival_ML_Model.sav", 'wb'))
pickle.dump(sc, open("Titanic_Survial_Prediction_Web/Titanic_Survial_Prediction_Web/scaler.sav", "wb"))