from PIL import Image
import os
import nibabel as nb
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    path_data = "data/IXI-T2/"
    path_processed = "data/single_imgs/"
    new_size = (125, 125)
    files = os.listdir(path_data)
    data_files = []
    for file in files:
        if ".nii.gz" in file:
            data_files.append(file)

    print("there is", len(data_files), "files detected")

    for i, data in enumerate(tqdm(data_files)):
        ni_file = nb.load(path_data + data)
        ni_data = ni_file.get_fdata().T
        for j, img in enumerate(ni_data):
            img_array = np.array(img)
            img_pil = Image.fromarray(img_array)
            img_pil = img_pil.convert("L")

            # save high quality images
            path_img = path_processed + "lq/" + str(i) + "_" + str(j) + ".png"
            img_pil.save(path_img)

            # save low quality images
            img_pil = img_pil.resize(new_size)
            path_img = path_processed + "hq/" + str(i) + "_" + str(j) + ".png"
            img_pil.save(path_img)
