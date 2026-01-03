# 2026-ML-PROJECT 1 (BBLEARN)
#  Creating fake data to simulate oil wells
#  Linear regression assumes a linear relationship between features and production

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generating synthetic oil and gas data
# Features: Well depth (feet), Reservoir pressure (psi), Temperature (°F)
# Target: Oil production volume (barrels per day)

# Set a "magic number" so the random data is the same every time (like a seed in a game for fair play)
np.random.seed(42)  # This makes sure I get the same fake data if I run the code again—great for testing!

# Decide how many examples I want (like 1000 stories or data points)
n_samples = 1000  # I create 1000 fake measurements

# Make up random "depth" values (in feet, like how deep a hole is dug)
# Imagine picking numbers between 5000 and 15000 randomly, like rolling dice
depth = np.random.uniform(5000, 15000, n_samples)  # Each depth is a random number in that range

# Make up random "pressure" values (in psi, like how much force is pushing)
# Again, random picks between 2000 and 5000
pressure = np.random.uniform(2000, 5000, n_samples)  # This is what I might want to predict later

# Make up random "temperature" values (in °F, like how hot it is)
# Random picks between 100 and 200
temperature = np.random.uniform(100, 200, n_samples)  # Another clue for predictions

# Simulating the production: Higher depth/pressure might correlate with higher production (simplified)
production = 50 + 0.005 * depth + 0.01 * pressure - 0.1 * temperature + np.random.normal(0, 10, n_samples)

# Creating a DataFrame
data = pd.DataFrame({
    'Depth': depth,
    'Pressure': pressure,
    'Temperature': temperature,
    'Production': production
})

#  Preparing data for ML #Features(x)  #Target(y)
X = data[['Depth', 'Pressure', 'Temperature']]  
y = data['Production'] 

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

#  Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")  # Closer to 1.0 is better

# Example prediction for a new well
new_well = pd.DataFrame({
    'Depth': [10000],
    'Pressure': [3500],
    'Temperature': [150]
})
predicted_production = model.predict(new_well)
print(f"\nPredicted production for new well: {predicted_production[0]:.2f} barrels per day")


#RESULT -> Model Performance:
Mean Squared Error: 111.02
R² Score: 0.72
Predicted production for new well: 120.15 barrels per day
