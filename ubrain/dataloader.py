import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import decode_image


class UpScalingDataset(Dataset):
    def __init__(self, path="data/single_imgs/"):
        self.path = path
        file_paths = os.listdir(path + "hq/")
        data = []
        for file in file_paths:
            if not(".png" in file):
                continue
            row = {"HQ": path + "hq/" + file,
                   "LQ": path + "lq/" + file}
            data.append(row)
        self.df = pd.DataFrame(data)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc(idx)
        image_lq = decode_image(row["LQ"])
        image_hq = decode_image(row["HQ"])
        return image_lq, image_hq
