from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

data_dict = pickle.load(open('data.pkl', 'rb'))


for i, item in enumerate(data_dict['data']):
    if len(item) != len(data_dict['data'][0]):
        print(f'Inconsistancy at index {i}: {len(item)}')

# Converting data and labels into numpy array for training
data = np.asarray(data_dict['data'])
label = np.asarray(data_dict['label'])

# Split dataset and train data using RandomForest
model = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=42, stratify=label)

model.fit(X_train, y_train)

# Test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)

print(f'Accuracy = {accuracy*100}%')

# Save model in a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
    print('Model saved successfully.')
