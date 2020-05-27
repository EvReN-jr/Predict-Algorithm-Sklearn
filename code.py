import pandas as pd#1
from sklearn.model_selection import train_test_split #2
from sklearn.ensemble import RandomForestClassifier#3
from sklearn.metrics import confusion_matrix#4
# import libs

datas=pd.read_csv("datas.csv")# read datas
#1

x=datas.iloc[:,3:-3].values  
y=datas.iloc[:,-2].values
# split values

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.10, random_state=0)# 90% for train, %10 for test
#2

rfc=RandomForestClassifier()
#3
rfc.max_depth=100
rfc.criterion="entropy"#select criterion,other criterion is 'gini'
rfc.n_estimators=1
rfc.fit(x_train,y_train)

y_pred=rfc.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
#4
print("RFC")
print(cm)
