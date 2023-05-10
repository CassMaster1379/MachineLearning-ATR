import os
import nibabel as nib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def extract_features(image_path, num_bins=100):
    nii_image =nib.load(image_path)
    
    image_data = nii_image.get_fdata()
    
    flat_data = image_data.ravel()
    
    hist, _ = np.histogram(flat_data, bins=num_bins)
    
    histo_norm = hist / hist.sum()
    
    return histo_norm

#define directorys

input_dir = 'ATR_smoth'
excel_file = 'ATR_training.xlsx'
num_bins = 50

#read labels
labels_df = pd.read_excel(excel_file)
labels = labels_df['label'].values

# Extract histo features for all images

feature_list = []
file_list = sorted(os.listdir(input_dir))
for filename in file_list:
    if filename.endswith('.nii.gz'):
        input_path = os.path.join(input_dir, filename)
        features = extract_features(input_path, num_bins)
        feature_list.append(features)
        
        
#convert list to array
X = np.array(feature_list)

#labels
y = np.array(labels)

#standarize the features 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#train the svm classifier
svm_class = SVC(kernel='linear', C=5)
cv = StratifiedKFold(n_splits=10)
scores = cross_val_score(svm_class,X_scaled, y, cv=cv)

print(f'Cross-validation scores: {scores}')
print(f'Average cross-validation score: {scores.mean()}')

        

#Testing the new images


