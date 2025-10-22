import torch

# Load the file
pt_file = torch.load("/data/output/28s3y60y/features/00017_slide_H&E_0.pt", map_location=torch.device('cpu'))

# Print the head of the file
print(pt_file.shape)