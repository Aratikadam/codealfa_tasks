import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load main dataset
data = pd.read_csv("Unemployment in India.csv")

# Display column names
print("Columns in dataset:\n", data.columns)

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Rename columns (if needed)
data.rename(columns={
    'Date': 'Date',
    'Estimated Unemployment Rate (%)': 'Unemployment Rate'
}, inplace=True)

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Sort by Date
data = data.sort_values('Date')

# Remove missing values
data = data.dropna()
print("\nFirst 5 rows:\n", data.head())

# Overall Trend
plt.figure()
plt.plot(data['Date'], data['Unemployment Rate'])
plt.title("Unemployment Rate Over Time (India)")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Covid-19 Impact (2020)
covid_data = data[data['Date'].dt.year == 2020]
plt.figure()
plt.plot(covid_data['Date'], covid_data['Unemployment Rate'])
plt.title("Unemployment Rate During Covid-19 (2020)")
plt.xlabel("Month (2020)")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Monthly Seasonal Trend
data['Month'] = data['Date'].dt.month
monthly_avg = data.groupby('Month')['Unemployment Rate'].mean()
plt.figure()
monthly_avg.plot(kind='bar')
plt.title("Average Monthly Unemployment Rate")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# Summary Statistics
print("\nSummary Statistics:")
print(data['Unemployment Rate'].describe())
