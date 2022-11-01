from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import pandas as pd
import random

class NIH_Dataset(Dataset):
    def __init__(self,
                 imgpath,
                 csvpath="Data_Entry_2017.csv",
                 bbox_list_path="BBox_List_2017.csv",
                 split="train",
                 views=["PA"],
                 transform=None,
                 nrows=None,
                 seed=0,
                 unique_patients=True,
                 gender=None,
                 pathology_masks=False
    ):
        super(NIH_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.root = "/content/drive/MyDrive/Datasets/NIH"
        self.imgpath = os.path.join(self.root, imgpath)
        self.csvpath = os.path.join(self.root, csvpath)
        self.transform = transform
        self.pathology_masks = pathology_masks
        self.split = split
        self.gender = gender

        self.train_valid_path = os.path.join(self.root, "train_val_list.txt")
        self.test_path = os.path.join(self.root, "test_list.txt")

        with open (self.train_valid_path, "r") as myfile:
            self.train_val_list = myfile.read().splitlines()

        with open (self.test_path, "r") as myfile:
            self.test_list = myfile.read().splitlines()
        
        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia"]

        self.pathologies = sorted(self.pathologies)

        # Load data
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)

        # Split data

        train_valid_data = self.csv[self.csv['Image Index'].isin(self.train_val_list)]
        test_data = self.csv[self.csv['Image Index'].isin(self.test_list)]
        
        if self.gender is not None:
            if self.gender in train_valid_data['Patient Gender'].unique():
                train_valid_data = train_valid_data[train_valid_data['Patient Gender'] == self.gender]
                test_data = test_data[test_data['Patient Gender'] == self.gender]
            else:
                raise ValueError(f'Gender could be only "F" for female and "M" for male, but got {self.gender}')
                
        if self.split == "train":
            self.csv = train_valid_data # self.csv[self.csv['Image Index'].isin(self.train_val_list)]
        elif self.split == "test":
            self.csv = test_data # self.csv[self.csv['Image Index'].isin(self.test_list)] # ~20%
        else:
            raise ValueError("You have to specifiy the split whether 'train' or 'valid' or 'test'")

        # Remove images with view position other than specified
        self.csv.loc[:, "view"] = self.csv.loc[:, 'View Position']
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first()

        self.csv = self.csv.reset_index()

        ####### pathology masks ########
        # load nih pathology masks
#         self.pathology_maskscsv = pd.read_csv(os.path.join(self.root, bbox_list_path),
#                 names=["Image Index","Finding Label","x","y","w","h","_1","_2","_3"],
#                skiprows=1)

#         train_valid_data_masks = self.pathology_maskscsv[self.pathology_maskscsv['Image Index'].isin(self.train_val_list)]
#         if self.split == "train":
#             self.pathology_maskscsv = train_valid_data_masks[:int(len(train_valid_data_masks) * 0.9)] # ~70% from data according to the official paper
#         elif self.split == "valid":
#             self.pathology_maskscsv = valid_data = train_valid_data_masks[int(len(train_valid_data_masks) * 0.9):] # ~10%
#         elif self.split == "test":
#             self.pathology_maskscsv = self.pathology_maskscsv[self.pathology_maskscsv['Image Index'].isin(self.test_list)] # ~20%

        # change label name to match
        # self.pathology_maskscsv.loc[self.pathology_maskscsv["Finding Label"] == "Infiltrate", "Finding Label"] = "Infiltration"
        # self.csv["has_masks"] = self.csv["Image Index"].isin(self.pathology_maskscsv["Image Index"])

        ####### pathology masks ########
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # patientid
        self.csv["patientid"] = self.csv["Patient ID"].astype(str)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]

        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        image = cv2.imread(img_path)
        if image.shape[-1] != 3:
            raise ValueError(f'Expected 3 but got {image.shape[-1]}')
            
        transform_seed = np.random.randint(2147483647)

        if self.transform is not None:
            random.seed(transform_seed)
            image = self.transform(image)
        return image, label

    
    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the 
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)] # Select the view


class CheX_Dataset(Dataset):
    def __init__(self,
                 imgpath,
                 csvpath,
                 views="Frontal",
                 uncertain='ones',
                 transform=None,
                 flat_dir=True,
                 seed=0,
                 unique_patients=False
    ):

        super(CheX_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        
        self.root = "C:\\Users\\marouane.tliba\\MedicalImaging\\swinCXR"
        self.imgpath = imgpath
        self.transform = transform
        self.csvpath = os.path.join(self.imgpath, csvpath)
        self.csv = pd.read_csv(self.csvpath).fillna(0)
        self.views = views
        self.uncertain = uncertain

        self.pathologies = list(self.csv.columns)[6:]

        if views in ['Frontal', 'Lateral']:
            self.csv = self.csv[self.csv["Frontal/Lateral"] == views].reset_index()


        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(pat = r'(patient\d+)')
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # # Get our classes.
        self.labels = np.array(self.csv[self.pathologies], dtype=np.float32)


        # Make all the -1 values into nans to keep things simple
        if self.uncertain == "zeros":
            self.labels[self.labels == -1] = 0
        elif self.uncertain == "ones":
            self.labels[self.labels == -1] = 1
        else:
            raise ValueError("uncertain must be 'ones' or 'zeros'")

        # Rename pathologies
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))

        # patientid
        if 'train' in csvpath:
            patientid = self.csv.Path.str.split("train/", expand=True)[1]
        elif 'valid' in csvpath:
            patientid = self.csv.Path.str.split("valid/", expand=True)[1]
        else:
            raise NotImplemented

        patientid = patientid.str.split("/study", expand=True)[0]
        patientid = patientid.str.replace("patient","")
        self.csv["patientid"] = patientid

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={}".format(len(self), self.views)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = os.path.join(self.root, self.csv['Path'].iloc[idx])
        image = cv2.imread(img_path)
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        if image is None:
            raise ValueError(f'None Image @ {img_path}')
            
        if image.shape[-1] != 3:
            raise ValueError(f'Expected 3 but got {image.shape[-1]}, {img_path}')

        transform_seed = np.random.randint(2147483647)

        if self.transform is not None:
            random.seed(transform_seed)
            image = self.transform(image)

        return image, label