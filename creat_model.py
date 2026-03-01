import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# 1. Load the dataset 
df = pd.read_csv('medical_insurance.csv') 

# 2.Convert text data into numbers (Encoding)
df['Sex'] = df['sex'].map({'female': 1, 'male': 0})
df['Smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df['Region'] = df['region'].map({'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3})

# 3. Separating Features (X) and the Target Variable (y)
X = df.drop('charges', axis=1)
y = df['charges']

# 4. Train the Model
model = RandomForestRegressor()
model.fit(X, y)

# 5.create .pkl file
# 'wb':write binary
with open('medical_insurance_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Congratulations...! medical_insurance_model.pkl file has been succesfully created")