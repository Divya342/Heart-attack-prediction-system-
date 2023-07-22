# Heart-attack-prediction-system

# Abstract
In India, close to regarding 2 hundredths of the whole population loses their lives because of an interrupted health observance system I.e., in most of the hospitals, doctor visits rarely. What happens if patient health becomes critical between that interval or if the doctor was unavailable to treat the patient? A patient might lose his/her life. So the main theme of our heart attack detection system using DE10 Nano FPGA, machine learning, and API Twilio is an innovative approach to monitor heart health and provide timely alerts to individuals who may be experiencing a heart attack.The system uses advanced machine learning algorithms to analyze real-time data from the DE10 Nano FPGA, which is capable of monitoring various vital signs such as heart rate, glucose levels, cholesterol levels, and ST-T wave abnormality. The system can then detect any abnormal patterns and send an alert through Twilio (API) to the user’s phone, allowing them to seek immediate medical attention.This technology has the potential to save lives by providing early detection and intervention in cases of heart attack, making it an important advancement in the field of healthcare by intimating the level of the disease (i.e., initial stage, intermediate...) In this way, we can reduce the mortality rate of a heart attack patients.
# Problem statement
Heart attack is a critical medical emergency that requires prompt diagnosis and treatment. Early detection of heart attack symptoms can help improve patient outcomes and reduce mortality rates. However, identifying symptoms of a heart attack can be challenging for individuals, and delayed diagnosis can result in severe consequences. Thus, there is a need for an accurate and reliable heart attack detection system that can quickly and efficiently identify symptoms and alert medical professionals.
# Solution proposed
A heart attack is a critical medical emergency that requires prompt diagnosis and treatment. Early detection of heart attack symptoms can help improve patient outcomes and reduce mortality rates. However, identifying symptoms of a heart attack can be challenging for individuals, and delayed diagnosis can result in severe consequences.
Thus, there is a need for an accurate and reliable heart attack detection system that can quickly and efficiently identify symptoms and alert medical professionals.we have proposed a solution to reduce the mortality rate of heart disease patients.We have implemented this project on DE10 nano FPGA board as the board offers HPS(hard processor system) which lets us to make communication with sensors and enables to writecode in python.We used LXDE(pre installer) OS to access the board.We get readings from sensors includes( Heart rate , IR , Gas and ECG sensors ) and compare the readings with the data-set Using Machine learning technology .we have the system with all the attributes(I.,e heart rate, ST-T wave abnormality,cholesterol,glucose ) and message will be sent to relatives or doctors mobile as output.
# Introduction of project
The real-time monitoring of heart failure patients, especially people with cardio-vascular diseases, is a very important task. Continuous monitoring can help minimize the need for human supervision in hospitals, and ensure the medical monitoring of individuals at risk for cardiac seizures without requiring heavy and costly hospital management.Therefore, the development of an embedded monitoring system is needed. It is possible toperform real-time monitoring of this kind of patient.To create a machine learning-based heart attack detection system using the DE10 Nano FPGA, you would need to first collect a large amount of patient data. This data would need to include variety of patient characteristics, such as age, gender, glucose levels, cholesterol level, ECG rest. You would also need to collect data on the patient's heart rate, blood pressure, cholesterol levels, and other relevant factors.Once you have collected the data, you can use machine learning algorithms to analyze it and identify patterns that are indicative of an increased risk of a heart attack. You would need to train the machine learning algorithm using a variety of techniques, such as supervised learning, unsupervised learning, or reinforcement learning, depending on the specific algorithm you are using.Once the machine learning algorithm has been trained, you can deploy it on the DE10 Nano FPGA to create a real-time heart attack detection system. DE10 nano FPGA board as the board offers HPS(hard processor system) which lets us to make serial communication with Ardunio.DE10 nano serial communication, is a protocol used for serial communication between devices. It allows for data to be sent in a bit-by-bit fashion over a single communication line.Arduino is a popular open-source hardware and software platform used for building digital devices and interactive objects. It is commonly used in the development of embedded systems and IoT applications.To implement this system, you can use an Arduino board with sensors to collect data from the patient, and a DE10 nano board to perform serial communication with the Arduino board. The DE 10 nano board can then be used to transmit the collected data to a computer or other device running the machine learning algorithm.We used LXDE(pre installer) OS as environment .We get readings from sensors includes( Heart rate , IR , Gas and ECG sensors ) and compare the readings with the dataset Using Machine learning technology. The machine learning algorithms would then analyze this data in real-time to identify any patterns or anomalies that could indicate an impending heart attack. The system would be designed to provide early warning to the patient and their healthcare provider, allowing them to take preventative measures and potentially save the patient's life.To ensure that the patient receives timely alerts, the system would be integrated with the Twilio platform, which provides a programmable API for sending SMS messages and voice calls. The system would be configured to automatically send alerts to the patient'shealthcare provider and emergency contacts in the event of an impending heart attack.

# Block diagram

![IMG_202307192_104753599](https://github.com/Divya342/Heart-attack-prediction-system-/assets/114659084/2840efe9-de45-4191-acbd-e13b8b41f6df)

# circuit/Schematic diagram
![IMG_202307192_104259147](https://github.com/Divya342/Heart-attack-prediction-system-/assets/114659084/0817baa2-ae66-4039-9bf5-e63f535648ce)
# Machine learning code
# -*- coding: utf-8 -*-
"""heartattack.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ULk8A3OncoxJ9MpuU-WkrsGNlA7wKeSE
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import plotly as plot
import plotly.express as px
import plotly.graph_objs as go

import cufflinks as cf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score,mean_squared_error
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot

path="/content/drive/MyDrive/dileep.csv"
dileep=pd.read_csv(path)
dileep

info = ["age","1: male, 0: female"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved"]



for i in range(len(info)):
    print(dileep.columns[i]+":\t\t\t"+info[i])

dileep['target']t

dileep.groupby('target').size()

dileep.groupby('target').sum()

dileep.shape

dileep.size

dileep.describe()

dileep.info()

dileep['target'].unique()

#Visualization

dileep.hist(figsize=(14,14))
plt.show()

plt.bar(x=dileep['sex'],height=dileep['age'])
plt.show()

sns.barplot(x="restecg", y="target", data=dileep)
plt.show()

sns.barplot(x=dileep['sex'],y=dileep['age'],hue=dileep['target'])

px.bar(dileep,dileep['sex'],dileep['target'])

sns.distplot(dileep["chol"])

sns.pairplot(dileep,hue='target')

dileep

numeric_columns=['fbs','chol','thalach','age']

sns.pairplot(dileep[numeric_columns])

dileep['target']

y = dileep["target"]

sns.countplot(y)

target_temp = dileep.target.value_counts()

print(target_temp)

# create a correlation heatmap
sns.heatmap(dileep[numeric_columns].corr(),annot=True, cmap='terrain', linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.show()

# create four distplots
plt.figure(figsize=(12,10))
plt.subplot(221)
sns.distplot(dileep[dileep['target']==0].age)
plt.title('Age of patients without heart disease')
plt.subplot(222)
sns.distplot(dileep[dileep['target']==1].age)
plt.title('Age of patients with heart disease')
plt.subplot(223)
sns.distplot(dileep[dileep['target']==0].thalach )
plt.title('Max heart rate of patients without heart disease')
plt.subplot(224)
sns.distplot(dileep[dileep['target']==1].thalach )
plt.title('Max heart rate of patients with heart disease')
plt.show()

plt.figure(figsize=(13,6))
plt.subplot(121)
sns.violinplot(x="target", y="thalach", data=dileep, inner=None)
sns.swarmplot(x="target", y="thalach", data=dileep, color='w', alpha=0.5)


plt.subplot(122)
sns.swarmplot(x="target", y="thalach", data=dileep)
plt.show()

dileep

# create pairplot and two barplots
plt.figure(figsize=(16,6))
plt.subplot(131)
sns.pointplot(x="sex", y="target", data=dileep)
plt.legend(['male = 1', 'female = 0'])
plt.subplot(132)
sns.barplot(x="restecg", y="target", data=dileep)
plt.legend(['yes = 1', 'no = 0'])
plt.subplot(133)
sns.countplot(x="thalach", hue='target', data=dileep)
plt.show()

dileep['target'].value_counts()

dileep['target'].isnull()

dileep['target'].sum()

dileep['target'].unique()

dileep.isnull().sum()

X,y=dileep,dileep.target

X.drop('target',axis=1,inplace=True)

y

####Or X, y = heart.iloc[:, :-1], heart.iloc[:, -1]

X.shape

y.shape

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10,test_size=0.3,shuffle=True)

X_test

y_test

print ("train_set_x shape: " + str(X_train.shape))
print ("train_set_y shape: " + str(y_train.shape))
print ("test_set_x shape: " + str(X_test.shape))
print ("test_set_y shape: " + str(y_test.shape))

# Model

# Decision Tree Classifier
scores_dict = {}

Catagory=['No','Yes you have Heart Disease']

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)

prediction=dt.predict(X_test)
accuracy_dt=accuracy_score(y_test,prediction)*100

scores_dict['DecisionTreeClassifier'] = accuracy_dt
print(accuracy_dt)

print("Accuracy on training set: {:.3f}".format(dt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, y_test)))

prediction

X_DT=np.array([[63,1,140,212,1,168]])
X_DT_prediction=dt.predict(X_DT)

X_DT_prediction[0]

Catagory[int(X_DT_prediction[0])]

#Feature Importance in Decision Trees

print("Feature importances:\n{}".format(dt.feature_importances_))

prediction=dt.predict(X_test)
accuracy_dt=accuracy_score(y_test,prediction)*100

scores_dict['DecisionTreeClassifier'] = accuracy_dt
print(accuracy_dt)

#Models
import pickle
pickle.dump(dt,open('model.pkl','wb'))
pickle.dump(sc,open('sc.pkl','wb'))

######Accuracy
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test,prediction)
confusion_matrix

import pickle
import numpy as np
from twilio.rest import Client

path1='/content/drive/MyDrive/sc (2).pkl'
path2='/content/drive/MyDrive/model (2).pkl'
sc = pickle.load(open(path1, 'rb'))
model = pickle.load(open(path2, 'rb'))
print("\n\n")
age = int(input("Enter Age : "))
se = int(input(" Enter Gender (1 or 0): "))
fbs = int(input("Enter fasting glucose level : "))
chol = int(input("Enter Serum Cholesterol : "))
restecg = int(input("Enter resting electrocardiographic results : "))
thalach = int(input("Enter maximum heart rate achieved :"))


# Test Set
new = [age, se,tresbbp, chol, restecg,
       thalach]
'''# Another Sets
x = [52, 1, 0, 125, 212, 0, 1, 168,	0, 1.0, 2,	2, 3]
inputs = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]'''

input = np.array([new])
input = sc.transform(input)
output = model.predict(input)
if(int(output == 0)):
    print("\n------Normal State ")
else:
    print("\n-------Abnormal")

accoun_sid='ACfc86854138e118fdf81366f3faec8c39'
auth_token='cb2c3a37204f3a03db1336c0d4c40cc1'
client=Client(accoun_sid,auth_token)
if(int(output ==0)):
  message=client.messages.create(body=("----Normal State",output),from_='+15855802827', to='+918688747438')
else:
  message=client.messages.create(body=("Abnormal",output),from_='+15855802827', to='+918688747438')



from google.colab import files
files.download('sc.pkl')

pip install twilio
# Conclusion
In conclusion, a heart attack detection system can be a valuable tool in identifying potential heart attacks and providing timely medical assistance to those in need. By using various data sources such as ECG signals, heart rate, glucose levels, cholesterol levels, and other vital signs, such a system can help detect and diagnose heart attacks early, potentially reducing the risk of serious complications or even death.
# Referenes
## REFERENCE 1
FPGA-based system for heart rate monitoring
Link:https://www.researcgate.net/publication/332079912_FPGA-basedSystem_for_Heart_Rate_Monitoring
## REFERENCE 2
Non invasive cholesterol meter using Near Infrared Sensor 
Link:https://ieeeexplore.ieee.org/document/7449581
## REFERENCE 3
A real time non-invasive cholesterol monitoring system 
Link:https://www.researchgate.net/publication/347525970_A_real_time_non_invasive_chol
esterol_monitoring_system

