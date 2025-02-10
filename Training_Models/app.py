import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load your dataset
# Replace 'your_dataset.csv' with your actual dataset path
df = pd.read_csv('D:\Interactive Dashboard\Training_Models\parkinsons.csv')

# Step 2: Preprocessing
# Identify categorical columns (non-numeric)
non_numeric_columns = df.select_dtypes(exclude=['number']).columns

# Encode categorical columns using Label Encoding
encoder = LabelEncoder()
for column in non_numeric_columns:
    df[column] = encoder.fit_transform(df[column])

# Separate features (X) and target (y)
# Replace 'target_column' with your actual target column name
X = df.drop('status', axis=1)  # Features
y = df['status']               # Target

# Step 3: Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Step 5: Train the model (RandomForestClassifier example)
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(x_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Save the model using pickle

 
filename='parkinson_model.sav'
pickle.dump(model,open(filename,'wb'))
print("Model and scaler have been saved as pickle files.")
