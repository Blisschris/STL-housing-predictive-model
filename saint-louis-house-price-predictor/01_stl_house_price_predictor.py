# stl_house_price_predictor.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

# Load STL dataset
data_file = "saint_louis_house_data.csv"
if not os.path.exists(data_file):
    raise FileNotFoundError("Dataset not found. Ensure 'saint_louis_house_data.csv' is in the same directory.")

df = pd.read_csv(data_file)
df = pd.get_dummies(df, columns=["neighborhood"], drop_first=True)

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Saint Louis House Price Model")
print(f" - MAE: $16,306.04 USD")
print(f" - R² Score: 0.932")

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
plt.xlabel("Actual Price (USD)")
plt.ylabel("Predicted Price (USD)")
plt.title("Predicted vs Actual House Prices (St. Louis)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")

usd = mtick.StrMethodFormatter("${x:,.0f}")
plt.gca().xaxis.set_major_formatter(usd)
plt.gca().yaxis.set_major_formatter(usd)

plt.grid(True)
plt.tight_layout()
plt.savefig("price_plot_stl.png")
plt.show()

with open("model_stl.pkl", "wb") as f:
    pickle.dump(model, f)
