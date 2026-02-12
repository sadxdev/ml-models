import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Step 1: Create Dataset
data = {
    "size_sqft": [500, 800, 1000, 1200, 1500, 1800, 2000, 2200],
    "price": [100000, 150000, 200000, 230000, 300000, 350000, 400000, 420000]
}

df = pd.DataFrame(data)

# Step 2: Define Features (X) and Target (y)
X = df[["size_sqft"]]
y = df["price"]

# Step 3: Split Data (Training & Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create Model
model = LinearRegression()

# Step 5: Train Model
model.fit(X_train, y_train)

# Step 6: Predict
predictions = model.predict(X_test)

# Step 7: Evaluate
error = mean_absolute_error(y_test, predictions)

print("Predictions:", predictions)
print("Actual:", y_test.values)
print("Mean Absolute Error:", error)

# Step 8: Predict New Value
new_size = np.array([[1600]])
predicted_price = model.predict(new_size)

print("Predicted price for 1600 sqft:", predicted_price[0])
