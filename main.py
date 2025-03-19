import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define file path (Modify if needed)
file_path = "C:/Users/user/Documents/Lincoln-Cloud-Computing/Research-Methods-2425/Assessment/insurance.csv"

# Load the dataset
data = pd.read_csv(file_path)

# Display the first few rows
print("Dataset Preview:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Encode categorical variables (Ensure all categorical features are converted to numeric)
print("\nEncoding categorical variables...")
data_encoded = pd.get_dummies(data, drop_first=True)

# Ensure all columns are numeric
data_encoded = data_encoded.apply(pd.to_numeric, errors='coerce')

# Descriptive statistics
print("\nDescriptive Statistics:")
print(data_encoded.describe())

# --- Outlier Detection and Removal ---
# Select only numerical columns
numeric_columns = data_encoded.select_dtypes(include=np.number)

# Compute IQR
Q1 = numeric_columns.quantile(0.25)
Q3 = numeric_columns.quantile(0.75)
IQR = Q3 - Q1

# Filter the dataset to remove outliers
filtered_data = data_encoded[~((numeric_columns < (Q1 - 1.5 * IQR)) | (numeric_columns > (Q3 + 1.5 * IQR))).any(axis=1)]
print(f"\nNumber of rows before outlier removal: {len(data_encoded)}")
print(f"Number of rows after outlier removal: {len(filtered_data)}")

# --- Additional Visualizations ---
# Histogram of Charges
plt.figure(figsize=(8, 5))
sns.histplot(data['charges'], kde=True)
plt.title('Distribution of Medical Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()

# Boxplot of Charges by Smoker Status
plt.figure(figsize=(8, 5))
sns.boxplot(x=data['smoker'], y=data['charges'])
plt.title('Boxplot of Charges by Smoking Status')
plt.xlabel('Smoker')
plt.ylabel('Charges')
plt.show()

# Scatter Plot with Regression Line (BMI vs Charges)
plt.figure(figsize=(8, 5))
sns.regplot(x=data['bmi'], y=data['charges'])
plt.title('Scatter Plot of BMI vs. Charges with Regression Line')
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.show()

# --- Hypothesis Testing ---

# Independent t-test: Charges for smokers vs. non-smokers
print("\nPerforming Independent t-test...")
smokers = data[data['smoker'] == 'yes']['charges']
non_smokers = data[data['smoker'] == 'no']['charges']
t_stat, p_value = stats.ttest_ind(smokers, non_smokers)
print(f"T-test Results:\n - t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Significant difference in charges between smokers and non-smokers (p < 0.05).")
else:
    print("No significant difference in charges between smokers and non-smokers (p >= 0.05).")

# Pearson correlation: BMI and charges
print("\nCalculating Pearson Correlation...")
correlation, p_value = stats.pearsonr(data['bmi'], data['charges'])
print(f"Correlation coefficient: {correlation:.4f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Significant correlation between BMI and charges (p < 0.05).")
else:
    print("No significant correlation between BMI and charges (p >= 0.05).")

# One-way ANOVA: Charges across regions
print("\nPerforming One-way ANOVA...")
anova_result = stats.f_oneway(
    data[data['region'] == 'northeast']['charges'],
    data[data['region'] == 'southeast']['charges'],
    data[data['region'] == 'southwest']['charges'],
    data[data['region'] == 'northwest']['charges']
)
print(f"F-statistic: {anova_result.statistic:.4f}, p-value: {anova_result.pvalue:.4f}")

if anova_result.pvalue < 0.05:
    print("Significant difference in charges across regions (p < 0.05).")
else:
    print("No significant difference in charges across regions (p >= 0.05).")

# --- Multiple Linear Regression ---
print("\nRunning Multiple Linear Regression...")

# Ensure all columns in X are numeric
X = filtered_data.drop(columns=['charges'])
y = filtered_data['charges']

# Debugging: Check initial data types of X
print("\nChecking Data Types of X:")
print(X.dtypes)

# Debugging: Check target variable type
print("\nChecking Data Type of y (Target Variable):", y.dtype)

# Convert boolean columns to integers (fixing dtype issue)
X = X.astype(int)

# Debugging: Check final data types before regression
print("\nFinal Data Types of X (After Fix):")
print(X.dtypes)

# Add intercept
X = sm.add_constant(X)

# Fit the model
try:
    model = sm.OLS(y, X).fit()
    print("\nRegression Model Summary:")
    print(model.summary())
except Exception as e:
    print("\nError in running regression model:", e)

# Model Performance Metrics
try:
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("\nModel Performance:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-Squared (R²): {r2:.4f}")
except Exception as e:
    print("\nError in calculating model performance metrics:", e)

print("\nAnalysis Complete!")
