import os
import SimpleITK as sitk
from glob import glob
from typing import List, Dict, Tuple, Any
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

def refresh(history: List[Dict[str, Any]], 
            dicom_db,
            path_dicom: str = PATH_DICOM, 
            loaded_path: List[Dict[str, Any]] = []) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Refresh the table of patients' information
    Args:
        history: List[Dict[str, Any]
            list of patients' information, e.g.: [{'PatientID': '000569}, {'PatientID': '000570'}]
        dicom_db: DicomDatabaseAPI
            database API
        path_dicom: str
            A path to dicom folder
        loaded_path: List[Dict[str, Any]
            A list of loaded path, this is used to avoid loading the same path twice
    Returns: Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        A tuple of (history, loaded_path)
    """
    path_dicom = os.path.abspath(path_dicom)
    
    series_path_and_relabel_status = dicom_db.get_all_series_path_and_relabel_status()
    for folder_info, relabel_status in series_path_and_relabel_status.values(): # folder_info: (patient_id, study_id, series_id)
        dirname = gen_dicom_path(*folder_info)
        if dirname not in loaded_path:
            patient_info = read_patient_infor(os.path.join(path_dicom, dirname, 'dicom'))
            loaded_path.append(dirname)
            history.append(patient_info)
        else:
            patient_info = list(filter(lambda x:dirname in x["Path"], history))[0]
        
        # Add relabel status    
        is_relabel, relabel_user_name = relabel_status
        if is_relabel:
            patient_info['Confirmed_User'] = relabel_user_name
            patient_info['Confirmed'] = 'V'
        else:
            patient_info['Confirmed_User'] = None
            patient_info['Confirmed'] = None
        
    history.sort(key = lambda x: x["PatientID"])
    # print(history)
    return history, loaded_path

if __name__ == "__main__":
    history = refresh()

