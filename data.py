# step 1: import necessary libraries
# data handling, analysis, and visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# data preprocessing, modeling, and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# step 2: load and clean data
# Read CSV file into Python using pandas
df = pd.read_csv("data.csv")

# Check value counts in EDUCATION for unreasonable or unknown variables
print(df.EDUCATION.value_counts())
## combine the "unknown" categories in the 'EDUCATION' variable
# from data description, we know 4=others, 5=unknown, 6=unknown
# so I replace categories 0, 5, and 6 with 4 (indicating "others")
df['EDUCATION'].replace({0:4, 5:4, 6:4}, inplace=True)
print(df.EDUCATION.value_counts())
# Similarly, I replace category 0 with 3 (the "other" category) for 'MARRIAGE'
print(df['MARRIAGE'].value_counts())
df['MARRIAGE'].replace({0:3}, inplace=True)
print(df['MARRIAGE'].value_counts())

# calculate the mean default rate for each group
plt.figure(figsize=(10,7))
sns.barplot(x='EDUCATION', y='default.payment.next.month', data=df)
plt.title('Default Rates by Education Level')
plt.show()


# one-hot encoding categorical variables
df_encoded = pd.get_dummies(df, columns=['EDUCATION', 'MARRIAGE'])

# scaling numerical variable
scaler = StandardScaler()
df[['LIMIT_BAL', 'AGE']] = scaler.fit_transform(df[['LIMIT_BAL', 'AGE']])

# separate features (X) and target (y)
X = df.drop('default.payment.next.month', axis=1)
y = df['default.payment.next.month']

# step 3: train-test split and balance data
# train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# balance data
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# step 4: logistic regression model
# Initialize logistic regression model
log_reg = LogisticRegression()

# Train the model
log_reg.fit(X_train, y_train)

# Predict on test data
y_pred_lr = log_reg.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred_lr))

# Get the coefficients
coefficients = log_reg.coef_
# Convert the coefficients into one-dimensional 1darray with corresponding df column names as axis labels
coeff_series = pd.Series(coefficients[0], df.columns[:-1])
# Show the coefficients
print(coeff_series)

# Probability of default (PD)
y_pred_proba_lr = log_reg.predict_proba(X_test)

# y_pred_proba_lr is a 2D array with probabilities for "non-default" and "default"
# We keep only the probabilities of default
pd_lr = y_pred_proba_lr[:, 1]

# Now pd_lr contains the probability of default for each instance in the test set
print(pd_lr)


# step 5: random forest model
# Initialize random forest model
rf = RandomForestClassifier()

# train the model
rf.fit(X_train, y_train)

# predict on test data
y_pred_rf = rf.predict(X_test)

# evaluate the model
print(classification_report(y_test, y_pred_rf))

# how the probability of default payment varies by demographic variables?
# get feature importances in Random Forest model
importances = rf.feature_importances_
# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, df.columns[:-1])
# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)
# make the bar plot from f_importances 
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16,9), rot=45, fontsize=15)
# Show the plot
plt.tight_layout()
plt.show()

# Probability of default
y_pred_proba_rf = rf.predict_proba(X_test)

# y_pred_proba_rf is a 2D array with probabilities for "non-default" and "default"
# We keep only the probabilities of default
pd_rf = y_pred_proba_rf[:, 1]

# Now pd_rf contains the probability of default for each instance in the test set
print(pd_rf)


### Question 1 ###
# How does the probability of default payment 
# vary by categories of different demographic variables?

# I use use groupby() function to group the data by different categories
# and calculate the mean default rate in each category

print(df.groupby('EDUCATION')['default.payment.next.month'].mean())

### Question 2 ###   
# Which variables are the strongest predictors of default payment?

# I use a machine learning model--random forest--to check feature importances
# define the features and target
X = df.drop('default.payment.next.month', axis=1)
y = df['default.payment.next.month']

# create and fit the model
model = RandomForestClassifier(random_state=1)
model.fit(X, y)

# print the feature importances
importances = model.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f'{feature}: {importance}')

### ### ### ### ### ### ### ### ### ### ### ### 