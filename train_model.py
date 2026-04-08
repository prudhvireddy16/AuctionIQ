import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Load Data
df = pd.read_csv('auction_data.csv')

# 2. Convert text (Make/Location) to numbers so AI can read it
df_encoded = pd.get_dummies(df, columns=['Make', 'Auction_Location'])

# 3. Define X (inputs) and y (what we want to predict)
X = df_encoded.drop('Sold_Price', axis=1)
y = df_encoded['Sold_Price']

# 4. Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 5. Save the "AI Brain" file
joblib.dump(model, 'auction_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')

print(" AI Model Trained and Saved as 'auction_model.pkl'!")
