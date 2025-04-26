import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os

# Load the datasets
dataset_path_por = r'C:\Users\Jeeban Jyoti\Downloads\student-performance-predictor\data\student-por.csv'
dataset_path_mat = r'C:\Users\Jeeban Jyoti\Downloads\student-performance-predictor\data\student-mat.csv'

if not os.path.exists(dataset_path_por) or not os.path.exists(dataset_path_mat):
    print(f"One or both datasets not found at {dataset_path_por} or {dataset_path_mat}")
    exit()

# Load both datasets
df_por = pd.read_csv(dataset_path_por, delimiter=';')
df_mat = pd.read_csv(dataset_path_mat, delimiter=';')

# Combine the datasets
df = pd.concat([df_por, df_mat], ignore_index=True)

# Encode categorical columns
df['sex'] = df['sex'].map({'F': 0, 'M': 1})
df['address'] = df['address'].map({'U': 0, 'R': 1})

# One-hot encode 'Mjob' (Parent's Job)
df = pd.get_dummies(df, columns=['Mjob'], prefix='Pjob', drop_first=True)

# Convert G3 into binary classes: Pass (1) if G3 >= 10, else Fail (0)
df['G3'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Feature columns including "goout" and excluding "absences"
features = ['age', 'sex', 'address', 'Medu', 'Dalc', 'studytime', 'failures', 'G1', 'G2', 'goout'] + \
           [col for col in df.columns if col.startswith('Pjob_')]

X = df[features]
y = df['G3']

print(f"Training features: {features}")
print(f"Number of training features: {len(features)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=200, max_depth=15)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Plot feature importance
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Model")
plt.show()

# Save model and scaler
os.makedirs("model", exist_ok=True)
joblib.dump(model, 'model/student_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("✅ Model and Scaler have been saved successfully!")