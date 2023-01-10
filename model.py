import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# get the data 
data=pd.read_csv("melb_data.csv")
print(data.columns)
y=data.Price
x=data.drop(["Price"],axis=1)

# split data into testing and training sets
X_train,X_valid,y_train,y_valid= train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)

# get the categorical and numerical variables
categorical_cols=[]
numerical_cols=[]
for x in X_train.columns:
    if data[x].dtype == "object" and data[x].nunique()<10:
        categorical_cols.append(x)
    elif data[x].dtype in ["int64","float64"]:
        numerical_cols.append(x)

my_cols=numerical_cols+categorical_cols
X_train=X_train[my_cols].copy()
X_valid=X_valid[my_cols].copy()

print(X_train.head())



"""
Pipelines and Pre Processing

"""
# preprocessing for numerical variables
numerical_transformer=SimpleImputer(strategy="constant")

# preprocessing for categorical variables
categorical_transformer=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

# bundle categorical and numerical variables
bundled_process=ColumnTransformer(transformers=[
    ("num",numerical_transformer,numerical_cols),
    ("cat",categorical_transformer,categorical_cols)
])

# define the machine learning model
model=RandomForestRegressor(n_estimators=100,random_state=0,criterion="absolute_error")

# bundle model and preprocessing
my_pipeline=Pipeline(steps=[
    ("preprocessor",bundled_process),
    ("model",model)
])

# fit
my_pipeline.fit(X_train,y_train)
# predict
predictions=my_pipeline.predict(X_valid)
# get the validation
mae=mean_absolute_error(y_valid,predictions)
print(mae)

