from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import pandas as pd
import numpy as np
import pickle


def f(x):
    if np.isnan(x):
        return 0
    else:
        return(x)
        
def g(x):
    if isinstance(x, float):
        return 'dropthis'
    else:
        return x

class PlotsterDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file='./poster_after_CLIP/posters/movie_data_rating_after_CLIP',text_file="list_clear_id_test_after_CLIP.txt", img_dir = "./poster_after_CLIP/posters/", transform1=Compose([ToTensor(),Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),]), transform2=Compose([ToTensor(),Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),]), transform3=Compose([ToTensor(),Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])):
        """
        Args:
            text_file (string): Path to the txt file with labels and titles.
            img_dir (string): Directory with all the images.
            transform (callable): Optional transform to be applied
                on a sample.
        """
        
        movie_dat = pd.read_csv(open(csv_file,'rU'), encoding='utf-8', engine='c')     
        movie_dat['rating'] = movie_dat.rating.apply(lambda x:f(x))
        movie_dat = movie_dat.rename(columns={"plot": "Plot"})
        movie_dat['Plot'] = movie_dat.Plot.apply(lambda x:g(x))
        movie_dat = movie_dat.drop(movie_dat[movie_dat.genre == '[]'].index)
        movie_dat = movie_dat.drop(movie_dat[movie_dat.Plot == 'dropthis'].index)
        self.movie_id = movie_dat.set_index('id')
        self.img_dir = img_dir
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        with open(text_file, "rb") as fp:
            self.file_id = pickle.load(fp)

    def __len__(self):
        return len(self.file_id)

    def __getitem__(self, idx):
        im_id = str(self.file_id[idx])
        im_name = "poster_AC_" + im_id + ".jpg"
        img_path = os.path.join(self.img_dir,
                                im_name)
        image = Image.open(img_path).convert('RGB')
        
        label = self.movie_id.loc[im_id].genre
        plot = self.movie_id.loc[im_id].Plot
        title = self.movie_id.loc[im_id].title     
       
        image_t1 = self.transform1(image)
        image_t2 = self.transform2(image)
        image_t3 = self.transform3(image)

        return image_t1, image_t2, image_t3, label, title.lower(), plot.lower(), im_id
        
def add_one(genre_list, idx):
    genre_list[idx] = 1
    return genre_list
    
def id_to_index(genre_id):
    if genre_id == 28:
        return 0
    if genre_id == 12:
        return 1
    if genre_id == 16:
        return 2
    if genre_id == 35:
        return 3
    if genre_id == 80:
        return 4
    if genre_id == 99:
        return 5
    if genre_id == 18:
        return 6
    if genre_id == 10751:
        return 7
    if genre_id == 14:
        return 8
    if genre_id == 36:
        return 9
    if genre_id == 27:
        return 10
    if genre_id == 10402:
        return 11
    if genre_id == 9648:
        return 12
    if genre_id == 10749:
        return 13
    if genre_id == 878:
        return 14
    if genre_id == 10770:
        return 15
    if genre_id == 53:
        return 16
    if genre_id == 10752:
        return 17
    if genre_id == 37:
        return 18