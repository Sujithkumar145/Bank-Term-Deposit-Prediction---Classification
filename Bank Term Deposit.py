#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('bank.csv - bank.csv.csv')


# In[3]:


data


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# There is no null values in the data

# In[6]:


data.columns


# In[7]:


data.describe()


# In[8]:


data['y'].value_counts()


# In[9]:


for col in data.select_dtypes(include='object').columns:
  print(col)
  print(data[col].unique())


# The Categorical Features have unknown values,It may affect our model.
# 

# In[10]:


categorical_features=[feature for feature in data.columns if ((data[feature].dtype=='O') and (feature not in ['y']))]
print(categorical_features)


# In[11]:


for col in data.select_dtypes(include='object').columns:
  print(col)
  print(data[col].value_counts())


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[13]:


for i in categorical_features :
  fig, ax=plt.subplots()
  sns.countplot(x=data[i],data=data)
  plt.xticks(rotation=90)


#     Insights:
#    * According to jobs category Admin,Bluecollar and technical peoples are mostly contacted
#    * In Education category those who completed a high school and also Degree holders are mostly contacted
#    * Those who already have credit are lessly contacted
#    * Those people who don't have personal loans are mostly contacted
#    * Most of the campaigns were conducted in end of april or start of may, Beacuse almost 30 % of people contacted in may.
#    
#     

# In[14]:


for i in categorical_features:
  sns.catplot(x='y',col=i,kind='count',data=data)
plt.show()


#     Insights:
#    * Retired people and students are shown more interest in bank deposit
#    * In the month of March,October and Septmeber more people have became depositor compared to other months.
#    * Those people having housing loan are less interested in bank deposit
#    * Despite more people contacted in may month, few people had became depositor.
#    * Successful campaigns adds more people in depositor team    
#    
#    

# In[15]:


for i in categorical_features:
  print(data.groupby(['y',i]).size())


# In[16]:


numerical_features=[feature for feature in data.columns if ((data[feature].dtype !='O') and (feature not in ['y']))]
print(numerical_features)


# In[17]:


data[numerical_features].head()


# In[18]:


data['pdays'].value_counts()


# In[19]:


data.replace(999,0,inplace=True)


# In[20]:


data


# In[21]:


discrete_feature = [feature for feature in numerical_features if len(data[feature].unique())<25]
print(len(discrete_feature))
print(discrete_feature)


# In[22]:


sns.countplot(x='previous',data=data)


# In[23]:


sns.displot(y='previous',x='y',data=data)


# In[24]:


sns.countplot(x='nr.employed',data=data)


# In[25]:


sns.countplot(x='emp.var.rate',data=data)


# In[26]:


sns.displot(data['age'],kde=True)
plt.show()
sns.displot(data['campaign'],kde=True)
plt.show()
sns.displot(data['pdays'],kde=True)
plt.show()
sns.displot(data['previous'],kde=True)
plt.show()
sns.displot(data['emp.var.rate'],kde=True)
plt.show()
sns.displot(data['cons.price.idx'],kde=True)
plt.show()
sns.displot(data['cons.conf.idx'],kde=True)
plt.show()
sns.displot(data['euribor3m'],kde=True)
plt.show()
sns.displot(data['nr.employed'],kde=True)
plt.show()


# In[27]:


plt.figure(figsize=(20,60))
pltno=1
for feature in numerical_features:
  ax=plt.subplot(12,3,pltno)
  sns.boxplot(x='y',y=data[feature],data=data)
  plt.xlabel(feature)
  pltno+=1
plt.show()


# In[28]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[29]:


plt.figure(figsize=(20,60))
pltno=1
for x in numerical_features:
  ax=plt.subplot(12,3,pltno)
  sns.boxplot(data[x])
  plt.xlabel(x)
  pltno+=1
plt.show()


# In[30]:


data.groupby(['y','campaign']).size()


# In[31]:


data.groupby(['y','pdays']).size()


# In[32]:


cor_df=data.corr()
print(cor_df)


# In[33]:


sns.heatmap(cor_df)


#     Handling Outliers
#  * Unknown records in each features are removed
#  * People whose age greater than 75 are removed due to more outliers
#  * campaigns counts upto 11 covers most of people, so the remaining samples are removed
#  * 4 features are heavily correalated and 1 feature holds around 20 % od unknown data.
#    Totally 5 features are dropped.
#  * There are around 2744 duplicate records in the dataset.Those duplicated are removed.   

# In[34]:


df=data.copy()


# In[35]:


df.groupby('age',sort=True)['age'].count().tail(20)


# In[36]:


df=df[df['age']<76]


# In[37]:


df=df[df['campaign']<12]


# In[38]:


df.drop(['default','emp.var.rate','cons.price.idx','euribor3m','nr.employed'],axis=1,inplace=True)


# In[39]:


df=df.loc[df['housing']!= 'unknown']


# In[40]:


df=df.loc[df['job']!='unknown']


# In[41]:


df=df.loc[df['marital']!='unknown']


# In[42]:


df=df.loc[df['education']!='unknown']


# In[43]:


df=df[df['pdays']<7]


# In[44]:


df=df[df['previous']<4]


# In[45]:


dup=df[df.duplicated(keep="last")]
dup


# In[46]:


df=df.drop_duplicates()


# In[47]:


df.shape


# In[48]:


df.describe()


# In[49]:


sns.boxplot(df['age'])
plt.show()


# In[50]:


df['y'].value_counts()


# In[51]:


categorical_features.remove('default')


# In[52]:


df=pd.get_dummies(df,columns=categorical_features,drop_first=True)


# In[53]:


df


# In[54]:


from sklearn import preprocessing
label=preprocessing.LabelEncoder()
df['y']=label.fit_transform(df['y'])


#     Data Preprocessing
#    * Target variable is transformed into binary with use of Label Encoder
#    * Remaining categorical variables are converted to binary values using OneHot Coder
#    * Due to OneHot Coding more columns are created

# In[55]:


# Model Building
from sklearn.model_selection import train_test_split
X=df.drop('y',axis=1)
Y=df['y']


# In[56]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)


# In[57]:


x_sca=pd.DataFrame(x_scaled, columns=X.columns)


#      Principal component analysis is used to reduce columns in the dataset.

# In[58]:


from sklearn.decomposition import PCA
pca=PCA(n_components=10)
pca.fit(x_sca)
pca_train=pca.transform(x_sca)
x_reduced=pd.DataFrame(pca_train)


# In[60]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix,roc_auc_score,classification_report
log=LogisticRegression()
tree=DecisionTreeClassifier()
knn=KNeighborsClassifier()
nb=GaussianNB()
rf=RandomForestClassifier()


# In[61]:


x_train,x_test,y_train,y_test=train_test_split(x_reduced,Y,test_size=.20,random_state =8,stratify=Y)


# In[62]:


rf.fit(x_train,y_train)
y_prf=rf.predict(x_test)
print(accuracy_score(y_test,y_prf))
print(roc_auc_score(y_test,y_prf))
print(recall_score(y_test,y_prf))
print(precision_score(y_test,y_prf))
print(confusion_matrix(y_test,y_prf))
print(classification_report(y_test,y_prf))
print(f1_score(y_test,y_prf))


# In[63]:


from imblearn.over_sampling import SMOTE
# creating an instance
sm = SMOTE(random_state=27)
# applying it to the training set
x_train_smote, y_train_smote = sm.fit_resample(x_train, y_train)


# In[64]:


rf.fit(x_train_smote,y_train_smote)
y_prf=rf.predict(x_test)
print(accuracy_score(y_test,y_prf))
print(roc_auc_score(y_test,y_prf))
print(recall_score(y_test,y_prf))
print(precision_score(y_test,y_prf))
print(confusion_matrix(y_test,y_prf))
print(classification_report(y_test,y_prf))
print(f1_score(y_test,y_prf))


# In[65]:


gb=GradientBoostingClassifier()
gb.fit(x_train_smote,y_train_smote)
y_pred=gb.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(f1_score(y_test,y_pred))


# In[66]:


from xgboost import XGBClassifier


# In[67]:


xgb=XGBClassifier()
xgb.fit(x_train_smote,y_train_smote)
y_pred=xgb.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(f1_score(y_test,y_pred))


# In[68]:


rf = RandomForestClassifier(criterion='gini', max_features=None, bootstrap=True, random_state=0,max_depth = 9,  
                            min_samples_leaf=6, min_samples_split=2,
                                  class_weight={0: 0.58, 1: 0.42},n_estimators=10)
rf.fit(x_train_smote,y_train_smote)
y_prf=rf.predict(x_test)
print(accuracy_score(y_test,y_prf))
print(roc_auc_score(y_test,y_prf))
print(recall_score(y_test,y_prf))
print(precision_score(y_test,y_prf))
print(confusion_matrix(y_test,y_prf))
print(classification_report(y_test,y_prf))
print(f1_score(y_test,y_prf))


# In[69]:


rf = RandomForestClassifier(criterion='gini', max_features=None, bootstrap=True, random_state=0,max_depth = 9, 
                            min_samples_leaf=6, min_samples_split=2,
                                  class_weight={0: 0.58, 1: 0.42},n_estimators=100)
rf.fit(x_train_smote,y_train_smote)
y_prf=rf.predict(x_test)
print(accuracy_score(y_test,y_prf))
print(roc_auc_score(y_test,y_prf))
print(recall_score(y_test,y_prf))
print(precision_score(y_test,y_prf))
print(confusion_matrix(y_test,y_prf))
print(classification_report(y_test,y_prf))
print(f1_score(y_test,y_prf))


# In[70]:


from sklearn.model_selection import cross_val_score
model_rf=cross_val_score(rf,x_train_smote,y_train_smote,scoring='f1',cv=20)
print(model_rf.mean())


# In[71]:


bgr=BaggingClassifier(rf, max_samples=0.5, max_features = 1, n_estimators = 10)
bgr.fit(x_train_smote,y_train_smote)
y_bgr=bgr.predict(x_test)
print(f1_score(y_test,y_bgr))


# In[72]:


adbr=AdaBoostClassifier(rf, n_estimators=5, learning_rate=1)
adbr.fit(x_train_smote,y_train_smote)
y_adbr=adbr.predict(x_test)
print(f1_score(y_test,y_adbr))

