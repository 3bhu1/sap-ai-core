# %%
import os
import pandas as pd

# %%
# Variables
DATA_PATH = '/app/data/irisdataset.csv'
CLASS_LABEL= int(os.getenv('CLASS_LABEL'))
KERNEL= os.getenv('KERNEL')
MODEL_PATH = 'model.pkl'

# %%
# Load Datasets
df = pd.read_csv(DATA_PATH)
X = df.drop('target', axis=1)
y = df['target']

# %%
# Partition into Train and test dataset
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

# %%

# Init model - Support Vector Machine 
from sklearn import svm
model = svm.SVC(C=CLASS_LABEL, kernel=KERNEL)

# %%
# Train model
model.fit(train_x, train_y)

# %%
# Test model
score = model.score(test_x, test_y)
print(f'Model Accuracy: {score:.2f}')

# %%
# Save model
import pickle
pickle.dump(model, open(MODEL_PATH, 'wb'))

