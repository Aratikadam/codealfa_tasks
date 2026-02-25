import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load Dataset
data = pd.read_csv("Advertising.csv")

# Remove unnamed column if exists
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)
print("First 5 rows:\n", data.head())

# Define Features & Target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("\nModel Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Feature Importance
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Impact (Coefficient)": model.coef_
})
print("\nAdvertising Impact on Sales:\n")
print(coefficients)

# Visualization
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()


# Predict Future Sales
# Example: New advertising plan
new_data = pd.DataFrame({
    'TV': [200],
    'Radio': [40],
    'Newspaper': [50]
})

future_sales = model.predict(new_data)
print("\nPredicted Future Sales for new advertising plan:", future_sales[0])
