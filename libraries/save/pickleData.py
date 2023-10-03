import pickle
import os
import stat
import pandas as pd
import numpy as np

def load(path:str):
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    return []

def load_dict(path:str):
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    return {}

def save(path:str, data):
    pickle.dump(data, open(path, "wb"))
    os.chmod(path, os.stat(path).st_mode | stat.S_IRWXO)

def save_mask(path:str, data):
    dst_path = f'{path}.npz'
    np.savez_compressed(dst_path, image=data)
    os.chmod(dst_path, os.stat(dst_path).st_mode | stat.S_IRWXO)
    # np.savez(dst_path, image=data)  # without compressed

def load_df(path:str):
    return pd.read_pickle(path)

def save_df(path:str, data:pd.DataFrame):
    data.to_pickle(path)

if __name__ == "__main__":
    # data = [
    #     {'PatientID': None, 'Name': 'NCKUH', 'Gender': 'M', 'Date_of_Birth': None, 'Modality': 'CT', 'Data_of_Study': None, 'Path': 'D:/Master/02_Project/NoduleDetection_v4/data/NCKUH/0003'},
    #     {'PatientID': None, 'Name': 'NCKUH', 'Gender': 'M', 'Date_of_Birth': None, 'Modality': 'CT', 'Data_of_Study': None, 'Path': 'D:/Master/02_Project/NoduleDetection_v4/data/NCKUH/0003'}
    # ]
    # save(r"./history/history.dat", data)
    import cv2 as cv
    image_data = load(r"./data/image.pickle")
    for key, data in image_data.items():
        cv.imshow("Nodule", data)
        cv.waitKey(100)