

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


df = pd.read_csv('sip.csv')

print(df.nunique(dropna=False))

print(df.columns)

df = pd.get_dummies(df,columns=['Category'])
print(df.head())

df_dtype = [df[col].dtype for col in df.columns]

print(df_dtype)
print(df.isnull().sum(axis=0))


df['Return 1-Year'].hist()

df['Return 3-Year'].hist()

df['Return 5-Year'].hist()

df['Return 1-Year'].describe()

df['Return 1-Year'][df['Return 1-Year']>26].count()

df['Return 1-Year'][df['Return 1-Year']>65].count()

df[df['Return 1-Year']>65]   #both belong to Category_EQ-SC(Small Cap equity sips)


print(pd.DataFrame(df[df['Return 1-Year']>25].sum(axis = 0)))

print(pd.DataFrame(df[df['Return 5-Year']>25].sum(axis = 0)))

print(pd.DataFrame(df[df['Return 3-Year']>25].sum(axis = 0)))

print(pd.DataFrame(df[df['Return 20-Year']>15].sum(axis = 0)))

print(pd.DataFrame(df['Fund'][ (df['Return 20-Year']>15) & (df['Category_EQ-MLC    ']==1)]))


print(pd.DataFrame(df['Fund'][ (df['Return 20-Year']>15) & (df['Category_EQ-MC     ']==1)]))


# We are observing a general trend that all the Multi Cap and Mid Cap Equities are showing a relatively better return (above  
# 25 %)all other (even if combined ) whether its a 1-year,5-year or 20-year

pd.DataFrame(df[df['Return 5-Year']>15].sum(axis = 0))

pd.DataFrame(df[df['Return 1-Year']>15].sum(axis = 0))

pd.DataFrame(df[df['Return 20-Year']>15].sum(axis = 0))

pd.DataFrame(df[df['Return 5-Year']>15].sum(axis = 0))




#When it comes to return upto 15% or above the Multi Cap ,MidCap and Large Cap equities perform as well. 
#Large Cap performs exceptionally well when it comes to moderate return for one year
#For more return above 25% mutli cap performs the best
#Large Caps performs the best in the long term investmet for returns 25% or above


df['Return 1-Year'][df['Return 1-Year']>25].count()

df['Return 1-Year'][df['Return 3-Year']>25].count()

df['Return 1-Year'][df['Return 5-Year']>25].count()

df['Return 1-Year'][df['Return 20-Year']>26].count()

df['Return 1-Year'][(df['Return 1-Year']>15) & (df['Return 1-Year']<25)].count()

df['Return 1-Year'][(df['Return 3-Year']>15) &((df['Return 3-Year']<25))].count()

df['Return 1-Year'][(df['Return 5-Year']>15) & ((df['Return 5-Year']<25))].count()

df['Return 1-Year'][(df['Return 20-Year']>15) & ((df['Return 20-Year']<25))].count()



#Short term investmets(less than 3 years) : one can get very large returns .
#Returns decrease as the investment increase 3 years

# As the size of investmentt increases the maximum return one can get tends to decrease but the minimum return tends to 
# increase.This means that once the investment period increaes the risk of loosing money decreases.
# Also the returns of most of the SIPs(75 percentile) decreases as the size of investment increase

print(df['Return 1-Year'][df['Return 1-Year']<0].count())

#Some SIPs can tend to loose you your money
print(pd.DataFrame(df[df['Return 1-Year']<0].sum(axis=0)))

 

#Creating Model and training data


training_cols = ['Rating', 'Return 1-Year', 'Return 3-Year',
       'Category_DT-CO     ', 'Category_DT-DB     ', 'Category_DT-INC    ',
       'Category_DT-LIQ    ', 'Category_DT-ST     ', 'Category_DT-UST    ',
       'Category_EQ-BANK   ', 'Category_EQ-INFRA  ', 'Category_EQ-LC     ',
       'Category_EQ-MC     ', 'Category_EQ-MLC    ', 'Category_EQ-SC     ',
       'Category_EQ-TS     ', 'Category_GL-MLT    ', 'Category_GL-ST     ',
       'Category_HY-AA     ', 'Category_HY-AR     ', 'Category_HY-DA     ',
       'Category_HY-DC     ', 'Category_HY-EQ     ', 'Return 5-Year']


df_train = df[training_cols]

print(df_train.head())

print(df_train.isnull().sum(axis=0))


# Since missing values are very less , we can try to impute the missing data in col:'return 1-year' and col2:'return 3-year
# with the mean of the col

one_year_mean = df_train['Return 1-Year'].mean()

three_year_mean = df_train['Return 3-Year'].mean()

for i in range(len(df_train)):
    if np.isnan(df_train['Return 1-Year'].iloc[i]):
        df_train['Return 1-Year']=one_year_mean
    if np.isnan(df_train['Return 3-Year'].iloc[i]):
        df_train['Return 3-Year']=three_year_mean

print(df_train.isnull().sum(axis=0))
print(len(df_train))


#Making the data having a entry for ['Return 5-Year'] column as the training set and the rest as train set.
#Actually , I am trying to predict those missing entries.

df_testset = df_train[df_train['Return 5-Year'].isnull()==True]
print(len(df_testset))

df_testset.head()
df_trainset = df_train[df_train['Return 5-Year'].isnull()==False]
print(df_trainset.head())
print(len(df_trainset))

y_train = df_trainset['Return 5-Year']

df_testset.drop('Return 5-Year',inplace=True,axis=1)

X_train = df_trainset.drop('Return 5-Year',axis=1)

features = X_train.columns
target = 'Return 5-Year'

print(len(y_train))


#I am trying to predict the values for ['Return 5-Year'] column. 



#Fisrt of all applying supervised learning on the data to predict the unlabelled data
#Sincce the train dataset is very small,supervised learning doesn't prove to be that much effective


from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


model_set =  [RandomForestRegressor(),Ridge(),BayesianRidge(),ExtraTreesRegressor(),ElasticNet(),KNeighborsRegressor(),GradientBoostingRegressor()]



#Finding the mean squared error over various models 

for model in model_set:
    model.seed = 42
    num_folds = 3
    scores = cross_val_score(model,X_train,y_train,scoring='mean_squared_error',cv=num_folds,n_jobs=-1)
    score_mean = (np.sqrt(scores.mean()*-1))
    print('{model:25} ->    RMSE:   {score}'.format(model=model.__class__.__name__,score=score_mean))


#Now I am using semi-supervised learning for predicting unlabelled data
#Applying Pseudo Labelling for the unlabelled data to make decisions boundary more accurate


from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin



class PseudoLabeler(BaseEstimator, RegressorMixin):
    
    def __init__(self, model, unlabled_data, features, target, sample_rate=0.2, seed=42):
        assert sample_rate <= 1.0, 'Sample_rate should be between 0.0 and 1.0.'
        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed
        self.unlabled_data = unlabled_data
        self.features = features
        self.target = target

    def get_params(self, deep=True):
        return {"sample_rate": self.sample_rate,"seed": self.seed,"model": self.model,"unlabled_data": self.unlabled_data,"features": self.features,"target": self.target}
    

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        augemented_train = self.__create_augmented_train(X, y)
        self.model.fit(
        augemented_train[self.features],
        augemented_train[self.target]
        )
        return self



    def __create_augmented_train(self, X, y):

        num_of_samples = int(len(self.unlabled_data) * self.sample_rate)

        # Train the model and creat the pseudo-labels
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.unlabled_data[self.features])

        # Add the pseudo-labels to the test set
        pseudo_data = self.unlabled_data.copy(deep=True)
        pseudo_data[self.target] = pseudo_labels

        # Take a subset of the test set with pseudo-labels and append in onto
         # the training set
        sampled_pseudo_data = pseudo_data.sample(n=num_of_samples)
        temp_train = pd.concat([X, y], axis=1)
        augemented_train = pd.concat([sampled_pseudo_data, temp_train])

        return shuffle(augemented_train)

    def predict(self, X):
        return self.model.predict(X)

    def get_model_name(self):
        return self.model.__class__.__name__

model_set_pseudo  = [PseudoLabeler(RandomForestRegressor(),df_testset,features,target,sample_rate=0.3)]

for model in model_set_pseudo:
    model.seed = 42
    num_folds = 2
    scores = cross_val_score(model,X_train,y_train,scoring='mean_squared_error',cv=num_folds,n_jobs=-1)
    score_mean = (np.sqrt(scores.mean()*-1))
    print('{model:25} ->    RMSE:  {score}'.format(model=model.__class__.__name__,score=score_mean))






