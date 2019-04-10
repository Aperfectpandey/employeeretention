
# coding: utf-8

# In[37]:


import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import sklearn 
from sklearn.linear_model import LogisticRegression
ds=pd.read_csv("datasets_HR/HR_comma_sep.csv")
ds.head(5)
ds.tail(5)
ds.dtypes
ds.describe().T
ds.describe(include=['object'])
ds.info()


# In[2]:


ds['left'].value_counts()


# In[3]:


ds.groupby('left').mean()


# In[ ]:


#The average satisfaction level of employees who stayed with the company is higher than that of the employees who left.
#The average monthly work hours of employees who left the company is more than that of the employees who stayed.
#The employees who had workplace accidents are less likely to leave than that of the employee who did not have workplace accidents.
#The employees who were promoted in the last five years are less likely to leave than those who did not get a promotion in the last five years


# In[4]:


ds.groupby('Department').mean()   #in department we found thathr and technical portion left the company
                                   # mostly


# In[5]:


ds.groupby('salary').mean()  #low salary portion left the most


# In[10]:


leftds=ds[ds['left']==1]
notleftds=ds[ds['left']==0]
leftds
print(leftds.shape)
print(notleftds.shape)


# In[11]:


sns.distplot(leftds['satisfaction_level'])


# In[12]:


corr=leftds.corr()      #used to build the correlation between the aatributes value and left valuesS
sns.heatmap(corr)


# In[13]:


corr=leftds.drop(labels='left',axis=1).corr()
corr


# In[14]:


sns.heatmap(corr)     #by heatmap(corr) we found that lastevaluation,number of project and promotionlast5yrs are major factor of employee company leaving 


# In[15]:


ds.salary=ds.salary.map({'low':0,'medium':1,'high':2})
ds.salary.head(5)
x=ds.drop(['left','Department'],axis=1)
y=ds['left']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
X_test
from sklearn.linear_model import LogisticRegression        #use of logistic  regression
model=LogisticRegression()
model.fit(X_train,y_train)
ypred=model.predict(X_test)
print(model.score(X_test,y_test))
model.predict_proba(X_test)


# In[24]:


ds.salary=ds.salary.map({'low':0,'medium':1,'high':2})
print(ds.salary.head(5))
from sklearn.preprocessing import LabelEncoder
le_salary=LabelEncoder()
ds['salary_n']=le_salary.fit_transform(ds['salary'])
ds_n=ds.drop(['salary'],axis=1)
print(ds_n.head(4))
x=ds_n.drop(['left','Department'],axis=1)
print(x.head(4))
y=ds_n['left']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
X_test

from sklearn.ensemble import RandomForestClassifier                  #use of randomforest method
model=RandomForestClassifier(n_estimators=30)
model.fit(X_train,y_train)
model.score(X_test,y_test)
ypred=model.predict(X_test)


# In[25]:


ds.salary=ds.salary.map({'low':0,'medium':1,'high':2})
print(ds.salary.head(5))
from sklearn.preprocessing import LabelEncoder
le_salary=LabelEncoder()
ds['salary_n']=le_salary.fit_transform(ds['salary'])
ds_n=ds.drop(['salary'],axis=1)
print(ds_n.head(4))
x=ds_n.drop(['left','Department'],axis=1)
print(x.head(4))
y=ds_n['left']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
X_test
from sklearn import tree
model=tree.DecisionTreeClassifier()    #use of decisiontree
model.fit(X_train,y_train)
model.score(X_test,y_test)
ypred=model.predict(X_test)


# In[39]:


from sklearn.preprocessing import LabelEncoder
le_salary=LabelEncoder()
ds['salary_n']=le_salary.fit_transform(ds['salary'])
ds_n=ds.drop(['salary'],axis=1)
print(ds_n.head(4))
x=ds_n.drop(['left','Department'],axis=1)
print(x.head(4))
y=ds_n['left']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
X_test
from sklearn.svm import SVC          #use of svm
model=SVC()
model.fit(X_train,y_train)
model.score(X_test,y_test)
ypred=model.predict(X_test)


# In[7]:


#Here classification_report  and accuracy_score.
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import sklearn 
from sklearn.linear_model import LogisticRegression
ds=pd.read_csv("datasets_HR/HR_comma_sep.csv")
ds.head(5)
ds.tail(5)
ds.dtypes
ds.describe().T
ds.describe(include=['object'])
ds.info()
ds.salary=ds.salary.map({'low':0,'medium':1,'high':2})
ds.salary.head(5)
x=ds.drop(['left','Department'],axis=1)
y=ds['left']
ypred=[]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
X_test
from sklearn.linear_model import LogisticRegression        #use of logistic  regression
model=LogisticRegression()
model.fit(X_train,y_train)
ypred=model.predict(X_test)
print("Logisticregression Model score:",model.score(X_test,y_test))
model.predict_proba(X_test)

from sklearn.preprocessing import LabelEncoder
le_salary=LabelEncoder()
ds['salary_n']=le_salary.fit_transform(ds['salary'])
ds_n=ds.drop(['salary'],axis=1)
print(ds_n.head(4))
x=ds_n.drop(['left','Department'],axis=1)
print(x.head(4))
y=ds_n['left']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
X_test
from sklearn.ensemble import RandomForestClassifier                  #use of randomforest method
model=RandomForestClassifier(n_estimators=30)
model.fit(X_train,y_train)
print("RandomForest Model score:",model.score(X_test,y_test))
ypred=model.predict(X_test)

from sklearn.preprocessing import LabelEncoder
le_salary=LabelEncoder()
ds['salary_n']=le_salary.fit_transform(ds['salary'])
ds_n=ds.drop(['salary'],axis=1)
print(ds_n.head(4))
x=ds_n.drop(['left','Department'],axis=1)
print(x.head(4))
y=ds_n['left']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
X_test
from sklearn import tree
model=tree.DecisionTreeClassifier()                             #use of decisiontree
model.fit(X_train,y_train)
print("Decisiontree Model score:",model.score(X_test,y_test))
ypred=model.predict(X_test)

from sklearn.preprocessing import LabelEncoder
le_salary=LabelEncoder()
ds['salary_n']=le_salary.fit_transform(ds['salary'])
ds_n=ds.drop(['salary'],axis=1)
print(ds_n.head(4))
x=ds_n.drop(['left','Department'],axis=1)
print(x.head(4))
y=ds_n['left']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
X_test
from sklearn.svm import SVC                                          #use of svm
model=SVC()
model.fit(X_train,y_train)
print("SVM Model score:",model.score(X_test,y_test))
ypred=model.predict(X_test)


from sklearn.metrics import classification_report,accuracy_score
print("accuracyscore is:",accuracy_score(y_test,ypred))
print("classificationreport is:",classification_report(y_test,ypred))

