#### Importing Required Libraries
import os
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn import ensemble
from sklearn import model_selection

#### --------------------------------------------------------------------- ####
#### --------------------------------------------------------------------- ####

### Required Functions

#### Drop unnecessary columns
def drop_col_not_req(df, cols):
    df.drop(cols, axis = 1, inplace = True)

#### Create a Fare category
def fare_category(fare):
    if (fare <= 8):
        return 'Low_Fare'
    elif (fare <= 10.5):
        return 'Med_Fare'
    elif (fare <= 100):
        return 'High_Fare'
    else:
        return 'Very_High_Fare'

#### Create a Family Size Category
def family_size_category(family_size):
    if (family_size <= 1):
        return 'Single'
    elif (family_size <= 3):
        return 'Small_Family'
    else:
        return 'Large_Family'
        
#### Function to fill missing 'Age' values
def fill_missing_age(missing_age_train, missing_age_test):
    missing_age_X_train = missing_age_train.drop(['Age'], axis = 1)
    missing_age_y_train = missing_age_train['Age']

    missing_age_X_test = missing_age_test.drop(['Age'], axis = 1)
    
    gbm_reg = ensemble.GradientBoostingRegressor()
    
    gbm_reg_param_grid = {'n_estimators': [1000], 'max_depth': [3], 'learning_rate': [0.1]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv = 10, n_jobs = 10, verbose = 0)
    gbm_reg_grid.fit(missing_age_X_train, missing_age_y_train)
    gbm_reg_grid.best_score_
    print("Avg CV Score for 'Age' Feature Regressor: " + str(gbm_reg_grid.score(missing_age_X_train, missing_age_y_train)))
    
    missing_age_test['Age'] = gbm_reg_grid.predict(missing_age_X_test)
    
    return missing_age_test


#### --------------------------------------------------------------------- ####
#### --------------------------------------------------------------------- ####

#### Read the Train and Test Data

train_data_orig = pd.read_csv('train.csv')
test_data_orig = pd.read_csv('test.csv')

#### Basic info of Train data

train_data_orig.shape
train_data_orig.info()
train_data_orig.describe()
train_data_orig.head()

#### More info of Train data - Analyze the data in each feature

#### Basic info of Test data

test_data_orig.shape
test_data_orig.info()
test_data_orig.describe()
test_data_orig.head()

#### More info of Test data - Analyze the data in each feature

#### Combine Train and Test data to fill missing values

combined_train_test = train_data_orig.append(test_data_orig)
combined_train_test.shape
combined_train_test.info()
combined_train_test.describe()

#### More info of Train and Test data combined - Analyze the data in each feature


#### --------------------------------------------------------------------- ####
#### --------------------------------------------------------------------- ####

## Feature Engineering

#### Pclass
print(combined_train_test['Pclass'].groupby(by = combined_train_test['Pclass']).count().sort_values(ascending = False))

pclass_dummies_df = pd.get_dummies(combined_train_test['Pclass'],
                                   prefix = combined_train_test[['Pclass']].columns[0])
combined_train_test = pd.concat([combined_train_test, pclass_dummies_df], axis = 1)

#### Fill basic missing values for 'Embarked' feature and convert it in to dummy variable
#### Fill missing Embarked values with Mode
if (combined_train_test['Embarked'].isnull().sum() != 0):
    combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
    combined_train_test.info()

#### Convert Embarked categorical variables into dummy/indicator variables
emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'],
                                prefix = combined_train_test[['Embarked']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis = 1)

#### Convert feature variable 'Sex' into dummy variable
lb_sex = preprocessing.LabelBinarizer()
lb_sex.fit(np.array(['male', 'female']))
combined_train_test['Sex'] = lb_sex.transform(combined_train_test['Sex'])

#### Extract Titles from Name feature and create a new column
combined_train_test['Title'] = combined_train_test['Name'].str.extract('.+,(.+)').str.extract('^(.+?)\.').str.strip()
titles_uniq = combined_train_test['Title'].unique()
print(titles_uniq)
print(combined_train_test['Title'].groupby(by = combined_train_test['Title']).count().sort_values(ascending = False))

title_Dictionary = {"Capt":       "Other", "Col":        "Other",
                    "Major":      "Other", "Jonkheer":   "Other",
                    "Don":        "Other", "Sir" :       "Other",
                    "Dr":         "Other", "Rev":        "Other",
                    "the Countess":"Other","Dona":       "Other",
                    "Mme":        "Mrs", "Mlle":       "Miss",
                    "Ms":         "Mrs", "Mr" :        "Mr",
                    "Mrs" :       "Mrs", "Miss" :      "Miss",
                    "Master" :    "Master", "Lady" :      "Other"
}

combined_train_test['Title'] = combined_train_test['Title'].map(title_Dictionary)
print(combined_train_test['Title'].groupby(by = combined_train_test['Title']).count().sort_values(ascending = False))

#### Fare
if (combined_train_test['Fare'].isnull().sum() != 0):
    combined_train_test['Fare'].fillna(combined_train_test['Fare'].mean(), inplace=True)
    combined_train_test.info()

combined_train_test['Fare'].describe()

combined_train_test['Fare_Category'] = combined_train_test['Fare'].map(fare_category)
le_fare = LabelEncoder()
le_fare.fit(np.array(['Low_Fare', 'Med_Fare', 'High_Fare', 'Very_High_Fare']))
combined_train_test['Fare_Category'] = le_fare.transform(combined_train_test['Fare_Category'])

#### Parch and SibSp
combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
print(combined_train_test['Family_Size'].groupby(by = combined_train_test['Family_Size']).count().sort_values(ascending = False))

combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(family_size_category)

le_family = LabelEncoder()
le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
combined_train_test['Family_Size_Category'] = le_family.transform(combined_train_test['Family_Size_Category'])

#### Age
#### Fill Missing values for 'Age' using relevant features - Name, Sex, Parch and SibSp

combined_train_test.info()
missing_age_df = pd.DataFrame(combined_train_test[['PassengerId', 'Age', 'Parch', 'Sex', 'SibSp', 'Family_Size', 'Family_Size_Category', 'Title']])
missing_age_df = pd.get_dummies(missing_age_df, columns = ['Title'])
missing_age_df.shape
missing_age_df.info()

missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test  = missing_age_df[missing_age_df['Age'].isnull()]

age_new_df = fill_missing_age(missing_age_train, missing_age_test)
age_new_df_with_pid = pd.DataFrame(age_new_df[['PassengerId', 'Age']])

new_df = combined_train_test[['PassengerId', 'Age']].set_index('PassengerId')
new_df.update(age_new_df_with_pid.set_index('PassengerId'))
combined_train_test['Age'] = new_df.values

combined_train_test.shape
combined_train_test.info()


#### --------------------------------------------------------------------- ####
#### --------------------------------------------------------------------- ####



#### --------------------------------------------------------------------- ####
#### --------------------------------------------------------------------- ####



#### --------------------------------------------------------------------- ####
#### --------------------------------------------------------------------- ####
