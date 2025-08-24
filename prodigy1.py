# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import zipfile
import os

# Load the dataset from the zip file
zip_path = "/content/house-prices-advanced-regression-techniques.zip"
csv_file_name = "train.csv" # Specify the name of the CSV file within the zip

# Extract the CSV file from the zip archive
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extract(csv_file_name, "/content/")

# Read the extracted CSV file into a pandas DataFrame
df = pd.read_csv(f"/content/{csv_file_name}")


# Features and target
# Assuming 'SalePrice' is the target variable and other relevant columns are features.
# You might need to adjust these based on the actual columns in your dataset.
# Let's select some numerical features for simplicity.
# You may need to perform data cleaning and feature engineering based on your dataset.
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'GarageCars', 'TotalBsmtSF']]  # Independent variables
y = df['SalePrice']  # Dependent variable

# Handle potential missing values by filling with the mean (a simple approach)
X = X.fillna(X.mean())


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

# Display the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict price for a new house - using example values based on the new features
# Prompt the user for house details
grlivarea = float(input("Enter the above ground living area (sq ft): "))
bedroomabvgr = int(input("Enter the number of bedrooms above ground: "))
fullbath = int(input("Enter the number of full bathrooms: "))
garagecars = int(input("Enter the number of garage cars capacity: "))
totalbsmtsf = float(input("Enter the total basement square footage: "))


new_house = [[grlivarea, bedroomabvgr, fullbath, garagecars, totalbsmtsf]]
predicted_price = model.predict(new_house)
print("Predicted Price for the new house:", predicted_price[0])