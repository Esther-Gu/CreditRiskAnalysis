# Credit Risk Analysis

This Python script is used to perform credit risk analysis.

## Libraries Used

- pandas
- numpy
- seaborn
- matplotlib
- sklearn
- imblearn

## Steps

The script is divided into the following sections:

1. **Importing Necessary Libraries**: The script begins by importing necessary libraries for data handling, analysis, visualization, and modeling.

2. **Data Loading and Cleaning**: The data is loaded from a CSV file, and initial cleaning operations are performed.

3. **Exploratory Data Analysis (EDA)**: Basic EDA is performed to understand the data. This includes checking statistics, histograms, and correlations.

4. **Data Preprocessing**: This step includes one-hot encoding of categorical variables and scaling of numerical variables.

5. **Train-Test Split and Balancing Data**: The data is split into training and test sets, and the training set is balanced using SMOTE.

6. **Logistic Regression Model**: A Logistic Regression model is trained, and predictions are made on the test data. The performance of the model is evaluated.

7. **Random Forest Model**: Similarly, a Random Forest model is trained and evaluated.

8. **Calculation of Expected Loss**: Finally, the expected loss is calculated using the formula EL = PD * EAD * LGD.
