import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Loading the Dataset
data = pd.read_csv("house_data.csv")

# Separate input features
X = data[['Size', 'Bedrooms', 'Age']]

# Target Variable
y = data['Price']

# Create Model
model = LinearRegression()

# Train Model
model.fit(X, y)

# Save trained model
joblib.dump(model,'house_model.pkl')
print("Model Trained and saved successfully!")