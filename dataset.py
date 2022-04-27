import pandas as pd
import torch
import os
import cv2
import numpy as np

class CardiacDataset(torch.utils.data.Dataset):

    def __init__(self, path_to_labels_csv, directory,root, augs):
        
        self.labels = pd.read_csv(path_to_labels_csv)
        self.bbox = []
        self.root_path = root
        self.directory = directory
        self.patients = []
        self.augment = augs

    def set_up_dataset(self):
        for root, dirs, files in os.walk(self.directory):
            for filename in files:
                path = os.path.join(root, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                y, x = img.shape
                img = cv2.resize(img,(360,360))
                patient = filename.replace('.',' ').split()[0]
                data = self.labels[self.labels["image_id"]==patient]
        
                # Get entries of given patient
                # Extract coordinates
                x_min = data["x"]*(360/x)
                y_min = data["y"]*(360/y)
                x_max = x_min + data["w"]*(360/x)  # get xmax from width
                y_max = y_min + data["h"]*(360/y)  # get ymax from height
                try:
                    max_arg = y_max.argmax()
                    bbox = [x_min.to_list()[max_arg],y_min.to_list()[max_arg],x_max.to_list()[max_arg],y_max.to_list()[max_arg]]
                except:
                    max_arg = 0
                    bbox = [np.NaN, np.NaN, np.NaN, np.NaN]
                
                self.patients.append(img)
                self.bbox.append(bbox)
                if len(self.patients) == 1000:
                    break
            if len(self.patients) == 1000:
                break
    def  __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.patients)
        
    def __getitem__(self, idx):
        """
        Returns an image paired with bbox around the heart
        """
        patient = self.patients[idx]
        # Get data according to index
        bbox = self.bbox[idx]

        # # Load file and convert to float32
        # file_path = self.root_path/patient  # Create the path to the file
        # img = np.load(f"{file_path}.npy").astype(np.float32)
        img = patient.astype(np.float32)
        
        # # Apply imgaug augmentations to image and bounding box
        # if self.augment:
            
        #     bb = BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
            
        #     ###################IMPORTANT###################
        #     # Fix for https://discuss.pytorch.org/t/dataloader-workers-generate-the-same-random-augmentations/28830/2
        #     # https://github.com/pytorch/pytorch/issues/5059
        #     random_seed = torch.randint(0, 1000000, (1,)).item()
        #     imgaug.seed(random_seed)
        #     #####################################################

        #     img, aug_bbox  = self.augment(image=img, bounding_boxes=bb)
        #     bbox = aug_bbox[0][0], aug_bbox[0][1], aug_bbox[1][0], aug_bbox[1][1]
            
            
        # Normalize the image according to the values computed in Preprocessing
        img = (img - 0.494) / 0.252

        img = torch.tensor(img).unsqueeze(0)
        bbox = torch.tensor(bbox)
            
        return img, bbox
