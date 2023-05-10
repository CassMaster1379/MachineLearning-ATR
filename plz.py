import nibabel as nib
import pandas as pd
import numpy as np
import warnings
from sklearn import svm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
warnings.filterwarnings('ignore')

# Load labels and features from Excel file
labels_df = pd.read_excel('ATR_training.xlsx')
labels = labels_df['label'].values
subject_ids = labels_df['subject_id'].values

# Feature extraction function
def extract_features(image):
    return np.mean(image.get_fdata())

# Load nii.gz images and extract features
features = []
for subject_id in subject_ids:
    img_path = f'ATR_final/{subject_id}.nii.gz'
    img = nib.load(img_path)
    features.append(extract_features(img))

features = np.array(features)

# Data preprocessing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 1))
X_test = scaler.transform(X_test.reshape(-1, 1))


# SVM classification
clf = svm.SVC(kernel='linear', C=1)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"The model is {accuracy*100}% accurate")
print("Confusion Matrix:\n", conf_matrix)
print(classification_report(y_test, y_pred, target_names=['0', '1', '2' , '3']))

#Testing the new images
