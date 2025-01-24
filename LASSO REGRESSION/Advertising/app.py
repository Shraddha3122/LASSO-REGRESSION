# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from flask import Flask, jsonify

# Load the data
data = pd.read_csv('D:/WebiSoftTech/LASSO REGRESSION/Advertising/Advertising.csv')

# Check for missing values
if data.isnull().sum().any():
    print("Data contains missing values.")
else:
    print("No missing values in the data.")

# Define features and target variable
X = data[['TV', 'radio', 'newspaper']]
y = data['sales']

# Split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_stateN=42)

# Create and fit the Lasso regression model
lasso = Lasso(alpha=0.1) 
lasso.fit(X_train, y_train)

coefficients = pd.DataFrame(lasso.coef_, X.columns, columns=['Coefficient'])

# Print the coefficients
print(coefficients)

#Plot the coefficients
coefficients.plot(kind='bar')
plt.title('Lasso Regression Coefficients')
plt.ylabel('Coefficient Value')
plt.xlabel('Media Type')
plt.show()

# Flask application
app = Flask(__name__)

@app.route('/analyze', methods=['GET'])
def analyze():
    # Return the coefficients as JSON
    return jsonify(coefficients.to_dict())

if __name__ == '__main__':
    app.run(debug=True)