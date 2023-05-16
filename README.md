# Credit Risk Analysis

This project is an implementation of credit risk analysis using Machine Learning algorithms such as Logistic Regression and Random Forest in Python.

## Project Description

The main goal of this project is to predict the probability of default for each instance in the dataset. The project also calculates expected loss using the formula EL = PD * EAD * LGD, where PD is Probability of Default, EAD is Exposure at Default, and LGD is Loss Given Default.

The original dataset, data.csv, comes from
Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

## Libraries Used

- pandas
- numpy
- seaborn
- matplotlib
- sklearn
- imblearn

## Steps

The project follows the following steps:

1. Data Loading and Cleaning
2. Exploratory Data Analysis
3. Data Preprocessing (One-hot encoding and scaling)
4. Train-Test Split and Balancing Data
5. Logistic Regression Model
6. Random Forest Model
7. Calculation of Expected Loss

## Usage

1. Ensure that the required libraries are installed in your Python environment.
2. Run the Python script in the project folder.
3. The script reads "data.csv" in the same folder. 
4. The script prints various statistics and information about the data, as well as model performance metrics.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


