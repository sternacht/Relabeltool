import os
import SimpleITK as sitk
from glob import glob
from .database import gen_dicom_path

PATH_DICOM = r"../Database\Dicom"
PATH_LOG = r"Database/Log"

def read_patient_infor(path):
    reader = sitk.ReadImage(os.path.join(path,os.listdir(path)[1]))
    dirname = os.path.dirname(path)
    patient_infor = {
        "PatientID": "0010|0020",
        "Name": "0010|0010",
        "Gender": "0010|0040",
        "Date_of_Birth": "0010|0030",
        "Modality": "0008|0060",
        "Date_of_Study": "0008|0020" 
    }
    for key, value in patient_infor.items():
        if reader.HasMetaDataKey(value):
            patient_infor[key] = reader.GetMetaData(value)
        else:
            patient_infor[key] = None
    patient_infor["Path"] = dirname
    if patient_infor["PatientID"] is None:
        patient_infor["PatientID"] = os.path.split(os.path.dirname(path))[-1]
    if patient_infor["Date_of_Birth"] is not None:
        date = patient_infor["Date_of_Birth"]
        patient_infor["Date_of_Birth"] = "{}/{}/{}".format(date[:4], date[4:6], date[6:])
    if patient_infor["Date_of_Study"] is not None:
        date = patient_infor["Date_of_Study"]
        patient_infor["Date_of_Study"] = "{}/{}/{}".format(date[:4], date[4:6], date[6:])
    if patient_infor["Name"] is not None:
        patient_infor["Name"] = patient_infor["Name"].replace(" ", "")
    if patient_infor["Gender"] is not None:
        patient_infor["Gender"] = patient_infor["Gender"].replace(" ", "")

    patient_infor["Style"] = "DICOM"
    return patient_infor

def checkExt(files, extension=".dcm"):
    for file in files:
        if extension in file:
            return True, file
    return False, None

def check_dir(dirname):
    dirname = dirname.replace('inference','')
    if 'mask' not in os.listdir(dirname):
        return True, dirname
    return False, None

def refresh(history, dicom_db, path_dicom= PATH_DICOM, loaded_path=[]):
    path_dicom = os.path.abspath(path_dicom)
    can_relabel = dicom_db.get_can_do_relabel()
    for dir_can_relabel in can_relabel.values():
        dirname = gen_dicom_path(*(dir_can_relabel[0]))
        if dirname not in loaded_path:
            patient_info = read_patient_infor(os.path.join(path_dicom, dirname, 'dicom'))
            patient_info['LOG'] = dirname
            loaded_path.append(dirname)
            history.append(patient_info)
        else:
            patient_info = list(filter(lambda x:x["LOG"] == dirname, history))[0]
        patient_info['Confirmed'] = 'V' if dir_can_relabel[1] else None
    history.sort(key=lambda x:x["PatientID"])
    # print(history)
    return history, loaded_path

if __name__ == "__main__":
    history = refresh()

