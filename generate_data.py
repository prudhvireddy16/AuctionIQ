import pandas as pd
import numpy as np

# Creating 5,000 rows of fake auction data
np.random.seed(42)
num_records = 5000

data = {
    'Vehicle_Year': np.random.randint(1960, 2024, num_records),
    'Mileage': np.random.randint(100, 150000, num_records),
    'Condition_Score': np.random.uniform(1, 10, num_records),
    'Make': np.random.choice(['Ford', 'Chevy', 'Porsche', 'Ferrari', 'Toyota'], num_records),
    'Auction_Location': np.random.choice(['Scottsdale', 'Palm Beach', 'Las Vegas'], num_records),
}

df = pd.DataFrame(data)

# Logic for Price: Older/Exotic cars + Good Condition = High Price
df['Sold_Price'] = (df['Vehicle_Year'] - 1960) * 500 + (df['Condition_Score'] * 5000) - (df['Mileage'] * 0.1) + 20000
df.loc[df['Make'] == 'Ferrari', 'Sold_Price'] += 100000 # Brand premium

# Save it
df.to_csv('auction_data.csv', index=False)
print("✅ Created auction_data.csv successfully!")