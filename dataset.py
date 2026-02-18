import torch #core deep learning library (tensors + GPU ops)
from torch.utils.data import Dataset #base class to create custom datasets
import torch.nn.functional as F # an alias for PyTorch's functional API 
import nibabel as nib #loads medical brain files (.mgz, .nii)
import numpy as np #numerical array operations
import os #file system operations (paths, listing files)
import pandas as pd
import os
import re
from torch.utils.data import DataLoader



# Load CSV
df = pd.read_csv("oasis_cross-sectional.csv")
df = df[df["ID"].str.contains("_MR1")] #take only the participants part of the MR1 group
df = df.dropna(subset=["CDR"]) #drop where whether they have alzheimers or not is not mentioned

# Extract subject ID only
df["Subject"] = df["ID"].str.extract("(OAS1_\\d+)") #remove stuff beyond the ID bc we want the text to exactly match the format of the csv
df["label"] = df["CDR"].apply(lambda x: 0 if x == 0 else 1) #we are making it into a binary classifier, 0-> healthy, greater than 0, dementia

# Get all mgz files
brain_files = [f for f in os.listdir(".") if f.endswith(".mgz")] #get all the files in this directory which end with .mgz

files = []

for file in brain_files:
    match = re.search(r"OAS1_\d+", file) #get just the oas1_id out
    if match:
        subject = match.group() #pulling the subject ID out of the filename.

        row = df[df["Subject"] == subject] #find matching row in csv
        if not row.empty:
            label = row["label"].values[0] #get the cdr value
            files.append((file, label)) #append to files as a tuple of file name and label of cdr

print("Usable subjects:", len(files))
class BrainDataset(Dataset):
    def __init__(self, file_list): #stores a list of file and label
        self.file_list = file_list
    
    def __len__(self): #tells pytorch how many samples exist
        return len(self.file_list)

    def __getitem__(self,idx): #for  a give index, load .mgz, convert to numpy, normalise, resize, convert to torch tensor, return(tensor, label)
        file_name, label = self.file_list[idx]
        image = nib.load(file_name)
        volume = image.get_fdata() #inbuilt function,converts the brain scan file into a 3D NumPy array of voxel intensity values

        volume = (volume -  np.mean(volume)) / np.std(volume) #normalise

        volume = torch.tensor(volume, dtype= torch.float32) #convert to tensor

        volume = volume.unsqueeze(0) #adds a new dimension at index 0 

        #resize to fixed shape:

        volume = F.interpolate( #expects 5D input for 3D volumes (Batch size, chanels, D, H, W)
            volume.unsqueeze(0), #temporarily add batch
            size = (128,128,128),
            mode = 'trilinear', #When resizing the 3D MRI scan, smoothly estimate new voxel values by blending nearby voxels in all three directions (depth, height, width).
            align_corners = False
        ).squeeze(0) #remove batch


        return volume, torch.tensor(label, dtype=torch.long)
    
dataset = BrainDataset(files)

print("Dataset size:", len(dataset))

x, y = dataset[0]

print("Volume shape:", x.shape)   # should be (1,128,128,128)
print("Label:", y)
print("Mean:", x.mean().item())
print("Std:", x.std().item())
        
loader = DataLoader(
    dataset, #How many MRI scans the model looks at before updating itself once. number of samples processed before one weight update.
    batch_size=4,
    shuffle = True #shuffle up the ordering
)
for x, y in loader:
    print("Batch shape:", x.shape)
    print("Batch labels:", y.shape)
    break

    
