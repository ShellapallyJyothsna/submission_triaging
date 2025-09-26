# # Step 1: Import necessary libraries
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# import joblib

# # Step 2: Load the dataset
# # Replace 'sample_submission_data.csv' with the path to your actual dataset
# df = pd.read_csv('Triaging_Data_Expanded_Complete.csv')  # Adjust path accordingly

# # Step 3: Preprocess the data
# # Drop non-predictor columns and prepare feature set (X) and target variable (y)
# X = df.drop(columns=["Submission ID", "Bind Propensity Score", "Submission Complete", "CAT Zone"])
# y = df["Bind Propensity Score"]

# # Apply One-Hot Encoding to categorical features
# X_encoded = pd.get_dummies(X, drop_first=True)

# # Step 4: Train the models
# # Split the data into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# # Initialize the models
# lr_model = LinearRegression()  # Linear Regression model
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Random Forest model

# # Train the Linear Regression model
# lr_model.fit(X_train, y_train)
# y_pred_lr = lr_model.predict(X_test)

# # Train the Random Forest model
# rf_model.fit(X_train, y_train)
# y_pred_rf = rf_model.predict(X_test)

# # Step 5: Evaluate the models
# # Calculate RMSE (Root Mean Squared Error) for both models
# rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
# rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# # Calculate R² (Coefficient of Determination) for both models
# r2_lr = r2_score(y_test, y_pred_lr)
# r2_rf = r2_score(y_test, y_pred_rf)

# # Print the performance of the models
# print(f"Linear Regression: RMSE = {rmse_lr:.4f}, R² = {r2_lr:.4f}")
# print(f"Random Forest: RMSE = {rmse_rf:.4f}, R² = {r2_rf:.4f}")

# # Step 6: Save the models to disk
# # Save both models as .pkl files using joblib
# joblib.dump(lr_model, 'linear_regression_model.pkl')
# joblib.dump(rf_model, 'random_forest_model.pkl')

# print("Models saved as 'linear_regression_model.pkl' and 'random_forest_model.pkl'")














#############3wworking

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# Step 2: Load the dataset
# Make sure the CSV file is in the same directory or provide the full path
try:
    df = pd.read_csv('Triaging_Data_Expanded_Complete.csv')
except FileNotFoundError:
    print("Error: 'Triaging_Data_Expanded_Complete.csv' not found. Please check the file path.")
    exit()

# Step 3: Preprocess the data
# Define the target variable
y = df["Bind Propensity Score"]

# Define the features (X) by dropping the target and other non-predictor columns
# We are now INCLUDING 'Historical Bind Rate', 'Submission Complete', 'CAT Zone', 'Days to Quote', and 'Prior Claims'
# We are DROPPING 'Bind_Flag' as it's a direct outcome variable (to prevent data leakage)
X = df.drop(columns=["Submission ID", "Bind Propensity Score", "Bind_Flag", "Total Insured Value ($)", "Expected Value"])

# Apply One-Hot Encoding to categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Step 4: Train the models
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize the models
lr_model = LinearRegression()  # Linear Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Random Forest model

# Train the Linear Regression model
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Train the Random Forest model
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Step 5: Evaluate the models
# Calculate RMSE (Root Mean Squared Error) for both models
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Calculate R² (Coefficient of Determination) for both models
r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)

# Print the performance of the models
print("--- Model Evaluation Results ---")
print(f"Linear Regression: RMSE = {rmse_lr:.4f}, R² = {r2_lr:.4f}")
print(f"Random Forest:     RMSE = {rmse_rf:.4f}, R² = {r2_rf:.4f}")
print("--------------------------------")


# Step 6: Save the models to disk
# Save both models as .pkl files using joblib
joblib.dump(lr_model, 'linear_regression_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')

print("\nModels saved successfully as 'linear_regression_model.pkl' and 'random_forest_model.pkl'")
print("You can now use these updated models in your Streamlit application.")










