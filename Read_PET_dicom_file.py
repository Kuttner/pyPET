# Example of how to read a PET DICOM file from a Siemens mMR scanner
# The code reads all DICOM files in a folder, sorts them by InstanceNumber and creates a 3D array with the pixel data.
# The code also scales the pixel data by the RescaleSlope and RescaleIntercept values to convert the data into the correct units.

import numpy as np
from pydicom import dcmread
from matplotlib import pyplot as plt
import os

PATH_PET = "/Users/sku014/Library/CloudStorage/OneDrive-UiTOffice365/Research/Projekt/PSMA_PET_ML_Virtuell_Biopsi/Bilder/PELVIS_PETACQUISITION_AC_IMAGES_30003/"

# Read all DICOM slices in the folder
dcm_files = os.listdir(PATH_PET)
files = []
for file in dcm_files:
    if file.endswith(".IMA"):
        file_path = PATH_PET + file
        files.append(dcmread(file_path))

# Sort the files by InstanceNumber
files.sort(key=lambda x: int(x.InstanceNumber))

# Create a 3D array with the pixel data
image = np.stack([f.pixel_array for f in files])

# Scale the image by the slope and intercept
slope = files[0].RescaleSlope
intercept = files[0].RescaleIntercept
image_scl = image * slope + intercept

# Plot the central slice of the image
plt.imshow(image_scl[image_scl.shape[0] // 2], cmap="hot")

# Print the max and min values of the scaled image
print(image_scl.max(), image_scl.min())
print(files[0].Units)

# Scale the image to SUV units
injecected_dose = (
    files[0].RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
)
patient_weight = files[0].PatientWeight
image_suv = image_scl * patient_weight / injecected_dose / 1000

# Print the max and min values of the SUV image
print(image_suv.max(), image_suv.min())
