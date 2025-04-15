import pandas as pd

df = pd.read_csv('./data/sales.csv')

df.to_csv('./data/sales.csv', index=False)

print("✅ Download stage done.")
