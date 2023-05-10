import os
import nibabel as nib
from scipy.ndimage import gaussian_filter

def apply_gaussian_filter(input_dir,output_dir, sigma):
    
    nii_image = nib.load(input_dir)
    
    im_data = nii_image.get_fdata()
    
    smoth_data = gaussian_filter(im_data, sigma)
    
    smoth_data[smoth_data != 0] = 0
    
    mod_nii_image = nib.Nifti1Image(smoth_data, nii_image.affine)
    
    nib.save(mod_nii_image, output_dir)
    

#define var
input_dir = 'ATR_data'
output_dir = 'ATR_smoth'
sigma = 10.65

for filename in os.listdir(input_dir):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f'smoth_{filename}')

            apply_gaussian_filter(input_path, output_path , sigma)
