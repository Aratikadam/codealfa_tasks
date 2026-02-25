import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv("car data.csv")

print("First 5 rows:\n", data.head())
print("\nDataset Info:\n")
print(data.info())

# Data Preprocessing
# Remove Car_Name column (not useful for prediction)
if 'Car_Name' in data.columns:
    data = data.drop('Car_Name', axis=1)

# Convert categorical columns to numeric
le = LabelEncoder()

categorical_columns = ['Fuel_Type', 'Selling_type', 'Transmission']

for col in categorical_columns:
    if col in data.columns:
        data[col] = le.fit_transform(data[col])

# Define Features & Target
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("\nModel Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Visualization
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices")
plt.show()
