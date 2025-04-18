import sys
import io
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import os

# Change the default encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Step 1: Data Preprocessing
# Load the dataset
df = pd.read_csv('Dataset.csv')

# Drop any unnamed or irrelevant columns
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col or '#NAME?' in col])

# Convert columns to numeric, coercing errors to NaN
numeric_columns = ['Temp_Min', 'Temp_Max', 'Pressure_Min', 'Pressure_Max', 'Performance', 'Selectivity']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill NaN values in numeric columns with the mean of the column
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Combine Temp_Min and Temp_Max into a single Temp column
df['Temp'] = (df['Temp_Min'] + df['Temp_Max']) / 2

# Combine Pressure_Min and Pressure_Max into a single Pressure column
df['Pressure'] = (df['Pressure_Min'] + df['Pressure_Max']) / 2

# Drop the old columns
df = df.drop(columns=['Temp_Min', 'Temp_Max', 'Pressure_Min', 'Pressure_Max'])

# Fill missing values in categorical columns with 'missing'
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna('missing')

# Define features and target
X = df.drop(columns=['Best_Catalyst', 'Notes'])
y = df['Best_Catalyst']

# Encode the target variable (y) into numeric labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Identify categorical and numeric columns
categorical_columns = X.select_dtypes(include=['object']).columns
numeric_columns = X.select_dtypes(include=['number']).columns

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist'), categorical_columns)
    ])

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y_encoded, test_size=0.2, random_state=42)

# Step 2: Model Design
class CatalystOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CatalystOptimizer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define the model
input_size = X_train.shape[1]
hidden_size = 64
output_size = len(label_encoder.classes_)  # Number of unique catalyst classes
model = CatalystOptimizer(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 3: Training
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)  # Use .toarray() if X_train is sparse
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)  # Use .toarray() if X_val is sparse
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Save the trained model
model_save_path = 'catalyst_optimizer_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Step 4: Inference
def predict_catalyst(reactant_1, reactant_2, product, temp=None, pressure=None):
    # Load the saved model
    model = CatalystOptimizer(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Reactant_1': [reactant_1],
        'Reactant_2': [reactant_2],
        'Product': [product],
        'Temp': [temp if temp is not None else df['Temp'].mean()],
        'Pressure': [pressure if pressure is not None else df['Pressure'].mean()]
    })

    # Fill missing values for categorical columns
    for col in categorical_columns:
        if col in input_data.columns:
            input_data[col] = input_data[col].fillna('missing')
        else:
            input_data[col] = 'missing'
    
    # Add missing columns with default values
    for col in X.columns:
        if col not in input_data.columns:
            if col in numeric_columns:
                input_data[col] = df[col].mean()  # Fill numeric columns with mean
            else:
                input_data[col] = 'missing'  # Fill categorical columns with 'missing'
    
    # Ensure the columns are in the same order as during training
    input_data = input_data[X.columns]
    
    # Apply preprocessing
    input_data_preprocessed = preprocessor.transform(input_data)
    
    # Convert to tensor
    input_tensor = torch.tensor(input_data_preprocessed.toarray(), dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    # Decode the predicted label back to the original catalyst name
    predicted_catalyst = label_encoder.inverse_transform([predicted.item()])[0]
    
    # Check if notes are available for the predicted catalyst
    note = df[df['Best_Catalyst'] == predicted_catalyst]['Notes'].values
    note = note[0] if len(note) > 0 else "No specific note available."
    
    return {
        'Optimal Catalyst': predicted_catalyst,
        'Note': note
    }

# Ask for user input
def get_user_input():
    print("Please provide the following inputs:")
    reactant_1 = input("Reactant 1: ")
    reactant_2 = input("Reactant 2: ")
    product = input("Product: ")
    temp = input("Temperature (optional, press Enter to skip): ")
    pressure = input("Pressure (optional, press Enter to skip): ")

    # Convert temperature and pressure to float if provided
    temp = float(temp) if temp else None
    pressure = float(pressure) if pressure else None

    return reactant_1, reactant_2, product, temp, pressure

# Main program
if __name__ == "__main__":
    # Ask for user input
    reactant_1, reactant_2, product, temp, pressure = get_user_input()

    # Make prediction
    result = predict_catalyst(reactant_1, reactant_2, product, temp, pressure)
    print("\nPrediction Result:")
    print(result)