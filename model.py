from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Import preprocessed data
from preprocess import X_train, X_test, y_train, y_test, scaler

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
