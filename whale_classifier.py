import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load simulated data
data = pd.read_csv('simulated_sst.csv')
X = data[['SST']].values
y = data['Label'].values

# Augment with fake data (more samples)
np.random.seed(42)
whale_sst = np.random.normal(8.04, 0.1, 50)  # Whale avg + noise
no_whale_sst = np.random.normal(8.00, 0.1, 50)  # No whale
X = np.concatenate([whale_sst, no_whale_sst]).reshape(-1, 1)
y = ['Whale'] * 50 + ['No Whale'] * 50

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Score
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Plot
plt.scatter(X_train, y_train, c='blue', label='Train')
plt.scatter(X_test, y_test, c='red', label='Test')
plt.axvline(x=8.02, color='green', linestyle='--', label='Decision Boundary')
plt.xlabel('SST (°C)')
plt.ylabel('Class')
plt.title('Whale vs. No Whale Classification')
plt.legend()
plt.savefig('classification_plot.png')
plt.show()