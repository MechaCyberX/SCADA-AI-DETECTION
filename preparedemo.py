import pandas as pd
import numpy as np

df = pd.read_csv("data/Demonstration.csv")
# 1) Replace infinities with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# 2) Drop or fill NaN
df.dropna(inplace=True)
# df.fillna(0, inplace=True)  # alternatively fill with 0
df.reset_index(drop=True, inplace=True)

# 3) Save new CSV
df.to_csv("cleaned_file.csv", index=False)
print(" Cleaned and saved!")
