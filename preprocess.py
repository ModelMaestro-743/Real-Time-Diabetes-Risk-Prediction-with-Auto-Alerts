import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Check for missing values
print(data.isnull().sum())

# Split into features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print('Data preprocessing complete.')
