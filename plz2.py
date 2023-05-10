import os
import glob
import nibabel as nib
from scipy.ndimage import zoom

def load_nifti_image(file_path):
    return nib.load(file_path)

def resize_image(image_data, new_shape):
    factors = [new_shape[i] / image_data.shape[i] for i in range(3)]
    return zoom(image_data, factors, order=1)

def save_resized_image(input_path, output_path, new_shape):
    image = load_nifti_image(input_path)
    image_data = image.get_fdata()
    resized_data = resize_image(image_data, new_shape)
    resized_image = nib.Nifti1Image(resized_data, image.affine)
    nib.save(resized_image, output_path)

data_dir = 'ATR_data'
output_dir = 'ATR_final'
new_shape = (128, 128, 128)

file_paths = glob.glob(os.path.join(data_dir, '*.nii.gz'))
for file_path in file_paths:
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, filename)
    save_resized_image(file_path, output_path, new_shape)
