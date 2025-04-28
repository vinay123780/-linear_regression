# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
# For this example, let's create a simple dataset of 'X' (independent variable) and 'Y' (dependent variable)
data = {
    'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Y': [1, 2.2, 2.9, 4.1, 5.0, 5.9, 7.1, 8.0, 9.1, 10.2]
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Display basic statistics of the dataset
print("Data Summary:")
print(df.describe())

# Visualizing the data with a scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='X', y='Y', data=df)
plt.title("Scatter Plot of X vs Y")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Prepare the data for modeling
# X (independent variable) and Y (dependent variable)
X = df[['X']]  # Double brackets to ensure it is a 2D array
Y = df['Y']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()


model.fit(X_train, Y_train)


Y_pred = model.predict(X_test)


mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


plt.figure(figsize=(8, 6))
sns.scatterplot(x='X', y='Y', data=df, color='blue', label='Actual data')
plt.plot(X_test, Y_pred, color='red', linewidth=2, label='Regression line')
plt.title("Linear Regression: X vs Y")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Slope (Coefficient of X): {model.coef_[0]}")
                        
