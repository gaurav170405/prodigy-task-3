import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Generate synthetic time series data with hills and valleys
np.random.seed(0)
n = 100
t = np.linspace(0, 10, n)
signal = np.sin(2 * np.pi * t / 5)  # Example signal with a period of 5

# Add noise to create variations
noise = 10.0 * np.random.normal(size=len(t))
data = signal + noise

# Label hills (peaks) and valleys (troughs) based on local maxima and minima
local_maxima = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1  # Indices of local maxima
local_minima = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1  # Indices of local minima

y = np.zeros(len(t))  # Initialize labels
y[local_maxima] = 1  # Peaks labeled as 1
y[local_minima] = -1  # Troughs labeled as -1

# Plot the synthetic time series data with peaks and troughs
plt.figure(figsize=(10, 5))
plt.plot(t, data, label='Synthetic Data')
plt.plot(t[local_maxima], data[local_maxima], 'ro', label='Peaks')
plt.plot(t[local_minima], data[local_minima], 'go', label='Troughs')
plt.title('Synthetic Time Series Data with Peaks and Troughs')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Prepare features and labels for logistic regression
X = np.column_stack([t, np.gradient(data)])  # Feature matrix data and its gradient
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
