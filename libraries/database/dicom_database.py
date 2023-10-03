import os
import logging
from datetime import datetime
from typing import Tuple, Dict, Optional
from .constant import dicom_database_setting, PATIENT_ID_FORMAT, STUDY_ID_FORMAT, SERIES_ID_FORMAT, INSTANCE_NUM_FORMAT 
from .database import Database
# from openfl_system.utilities import get_local_time_in_taiwan
import re

logger = logging.getLogger(__name__)

def gen_dicom_path(patient_id: int, study_folder_index: int, series_folder_index: int) ->str:
    """Generate path for dicom file.

    The path is based on patient_id, study_folder_index, series_folder_index, e.g. ID-000001\\std-0001\\Ser-001, ID-000001\\std-0001\\Ser-001
    
    Args:
        patient_id: the id of the patient
        study_folder_index: the folder index of the study for the paitent
        series_folder_index: the folder index of the series for the study
    """
    patitent_folder = PATIENT_ID_FORMAT.format(patient_id)
    study_folder = STUDY_ID_FORMAT.format(study_folder_index)
    series_folder = SERIES_ID_FORMAT.format(series_folder_index)
    path = os.path.join(patitent_folder, study_folder, series_folder)
    return path

def gen_dicom_file_name(patient_id: int, study_folder_index: int, series_folder_index: int, instance_num:int=-1) -> str:
    """Generate the file name for dicom file (Not Contain extension!)

    Return:
        the name of dicom file based on give arguments, e.g 'ID-000001_Std-0001_Ser-001'
    """
    file_name_format = "{}_{}_{}".format(PATIENT_ID_FORMAT, STUDY_ID_FORMAT, SERIES_ID_FORMAT)
    file_name = file_name_format.format(patient_id, study_folder_index, series_folder_index)
    # if instance_num != -1:
    #     file_name += '_' + INSTANCE_NUM_FORMAT.format(instance_num)
    return file_name

def gen_dicom_file_name_from_path(path: str) -> str:
    """Generate the file name for dicom file from path string(Not Contain extension!)

    Return:
        the name of dicom file based on give arguments, e.g 'ID-000001_Std-0001_Ser-001'
        a list of 'int string' that store id of patient, study and series, e.g ['000001','0001','001']
    """
    id_pattern = r'ID-(\d+)'
    study_pattern = r'Std-(\d+)'
    series_pattern = r'Ser-(\d+)'
    
    id_matching = re.search(id_pattern, path)
    study_matching = re.search(study_pattern, path)
    series_matching = re.search(series_pattern, path)
    if id_matching == None or study_matching == None or series_matching == None:
        return None, None
    else:
        infos = [int(id_matching.group(1)), int(study_matching.group(1)), int(series_matching.group(1))]
        return gen_dicom_file_name(*infos), infos

class DicomDatabaseAPI(object):
    def __init__(self, db_path:str='./data/DICOM.db') -> None:
        self.db = Database(db_path, dicom_database_setting)
        self.cache = dict()
    
    def connect(self):
        self.db.connect()

    def __enter__(self):
        self.db.connect()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.db.close()
     
    def add_patient(self, patient_uid: str, patient_id: int) -> int:
        """Add patient ID into patient table
        
        Args:
            patient_uid: string, the real patient UID for this dicom file
            patien_id: int, the patient ID for database
        Return: 
            integer, the index of patient
        """
        # Write into database
        attrs = ['patient_id', 'patient_uid', 'is_removed']            
        values = (patient_id, patient_uid, 0)
        self.db.insert("patient", values=values, attrs=attrs)
        return patient_id

       
    def add_study(self, patient_id:int, study_uid:str) -> Tuple[int, int]:
        """Add study UID into study table
        
        Args:
            patient_id: integer, the index for the patient
            study_uid: string, the study UID for this dicom file
        Return: 
            tuple(study_id, folder_index for this study)
        """
        cond = {'patient_id':patient_id, }
        attrs = ['study_id', 'study_uid', 'folder_index']
        order = {'folder_index': 'down', }
        rs = self.db.select("study", attrs, cond, order)

        study_id = -1
        folder_index = -1
        # Found some rows having same patient_id
        if len(rs) != 0:
            for row in rs:
                if study_uid == row[1]:
                    study_id = row[0]
                    folder_index = row[2]
                    return study_id, folder_index
            # If not found in these row, then set folder index to the largest folder_index + 1
            folder_index = rs[0][2] + 1
        # Not Found
        else:
            folder_index = 1
        study_id = self.db.get_length_of_table("study") + 1

        # Insert new study
        attrs = ['study_uid', 'folder_index', 'patient_id']
        values = (study_uid, folder_index, patient_id)
        self.db.insert("study", values, attrs)

        return study_id, folder_index
    
    def add_series(self, patient_id:int, study_id:int, series_uid:str) -> Tuple[int, int]:
        """Add series UID into series table
        
        Args:
            patient_id: integer, the patient index
            study_id: integer, the study index
            series_uid: str, the series UID for this dicom
        Return: 
            tuple(series_id, folder_index) for this series
        """
        cond = {'patient_id':patient_id, 'study_id':study_id}
        attrs = ['series_id', 'series_uid', 'folder_index']
        order = {'folder_index': 'down', }
        rs = self.db.select("series", attrs, cond, order)

        series_id = -1
        folder_index = -1
        # Found some records having same patient_id and study_id
        if len(rs) != 0:
            for row in rs:
                if series_uid == row[1]:
                    series_id = row[0]
                    folder_index = row[2]
                    return series_id, folder_index
            # Not Found in these records, then use (the highest folder_index + 1)
            folder_index = rs[0][2] + 1
        # Not Found
        else:
            folder_index = 1
        series_id = self.db.get_length_of_table("series") + 1
        # Insert new series
        attrs = ['series_uid', 'folder_index', 'patient_id', 'study_id']
        values = (series_uid, folder_index, patient_id, study_id)
        self.db.insert("series", values, attrs)

        return series_id, folder_index

    def add_folder_path(self, series_id:int, patient_id:int, study_folder_index:int, series_folder_index:int)-> None:
        cond = {'series_id':series_id, }
        rs = self.db.select("series_path", cond=cond)
        # This series_id is already in the series_path table 
        if len(rs) != 0:
            return

        attrs = ['series_id', 'patient_id', 'study_folder_index', 'series_folder_index']
        values = (series_id, patient_id, study_folder_index, series_folder_index)
        self.db.insert("series_path", values, attrs)

    # def add_empty_series_status(self, series_id:int) -> None:
    #     cond = {'series_id':series_id, }
    #     rs = self.db.select("series_status", cond=cond)
    #     # This series_id is already in the seriesStatus 
    #     if len(rs) != 0:
    #         return
    #     # Insert new record into series_status table
    #     taiwan_now = get_local_time_in_taiwan()
    #     cur_time = taiwan_now.strftime("%Y/%m/%d")
    #     attrs = ['series_id', 'is_continual', 'is_npy', 'is_inference', 'is_relabel', 'created_date']
    #     values = (series_id, 0, 0, 0, 0, cur_time)
    #     self.db.insert("series_status", values, attrs)

    def remove_oldest_dicom_file_and_get_patient_id(self) -> int:
        # self.
        pass


    def add_new_dicom_file(self, patient_uid:str, study_uid:str, series_uid:str, num_limit_of_series: int) -> Tuple[int, int, int]:
        # Because the Instance from same series will have same patient_uid, study_uid and 
        # series_uid, storing them will decrease the times for searching 
        key = "{}/{}/{}".format(patient_uid, study_uid, series_uid)
        indicies = self.cache.get(key)
        if indicies != None:
            return indicies

        patient_id = self.get_patient_id(patient_uid)
        num_series_in_db = self.db.get_length_of_table("patient")
        if patient_id == None:
            if num_series_in_db < num_limit_of_series:
                patient_id = num_limit_of_series + 1
            else:
                patient_id = self.remove_oldest_dicom_file_and_get_patient_id()
            self.add_patient(patient_uid, patient_id)
        study_id, study_folder_index = self.add_study(patient_id, study_uid)
        series_id, series_folder_index = self.add_series(patient_id, study_id, series_uid)
        self.add_folder_path(series_id, patient_id, study_folder_index, series_folder_index)
        self.add_empty_series_status(series_id)

        self.cache[key] = (patient_id, study_folder_index, series_folder_index)
        return patient_id, study_folder_index, series_folder_index

    def get_not_continual(self) -> Dict[int, Tuple[int, int, int]]:
        """Get the folder information which 'is_continual' is 0  
       
        Search series_status table and find the series_id of the row which 'is_continual' is 0, then use series_id to search series_path table to get folder

        Returns: 
            dict, pair(series_id : tuple(patient_id, study_folder_index, series_folder_index))
        """
#        folders = dict()

        cond = {'is_continual':0, }
        return self._get_folder_infos(cond)
    
    def get_can_do_inference(self) -> Dict[int, Tuple[int, int, int]]:
        """Get the information of series folder which is continual, already converted into npy file and not be inference
        """
        cond = {'is_npy':1, 'is_continual':1, 'is_inference': 0}
        return self._get_folder_infos(cond)

    def get_can_do_relabel(self) -> Dict[int, Tuple[int, int, int]]:
        """Get the information of series folder which is continual, already converted into npy file and already inferenced.
        """
        cond = {'is_npy':1, 'is_continual':1, 'is_inference':1}#, 'is_relabel':0}
        return self._get_folder_infos(cond)

    def get_can_do_npy(self) -> Dict[int, Tuple[int, int, int]]:
        """Get the information of series folder which is continual and not be converted into npy file.
        """
        cond = {'is_npy':0, 'is_continual':1}
        return self._get_folder_infos(cond)

    def get_series_id_by_dicom_meta(self, patient_uid: str, study_uid: str, series_uid: str) -> Optional[int]:
        rs = self.db.select('patient', 'patient_id', cond={'patient_uid':patient_uid})
        if len(rs) == 0:
            return None
        patient_id = rs[0][0]

        cond = {'patient_id':patient_id, 
                'study_uid': study_uid}
        rs = self.db.select('study', 'study_id', cond=cond)
        if len(rs) == 0:
            return None
        study_id = rs[0][0]

        cond ={'patient_id':patient_id,
                'study_id': study_id,
                'series_uid': series_uid}
        rs = self.db.select('series', 'series_id', cond=cond)
        if len(rs) == 0:
            return None
        series_id = rs[0][0]

        return series_id
    
    def get_series_id_by_folder_info(self, patient_id: int, study_folder_idx: int, series_folder_idx: int) -> Optional[int]:
        cond = {'patient_id': patient_id, 
                'study_folder_index': study_folder_idx, 
                'series_folder_index': series_folder_idx}

        result = self.db.select("series_path", 'series_id', cond=cond)
        if len(result) == 0:
            return None

        return result[0][0]

    def get_patient_id(self, patient_uid: str) -> int:
        """Get patient id in database for given patient_uid
        Returns: int
            patient id in database

        """
        cond = {'patient_uid':patient_uid, 'is_removed':0}
        rs = self.db.select("patient", cond=cond)
        # Not Found
        if len(rs) == 0:
            return None
        else:
            return int(rs[0][0])

    def update_is_npy(self, series_id: int, status:bool) -> None:
        """Update is_npy for given series_id for series_status table 

        Args:
            series_id: A index of the series
            status: the new status for column 'is_npy'
        """
        if not isinstance(status, bool):
            raise ValueError("Status should be bool.")
        self._update_seriesStatus(series_id, is_npy=status)

    def update_is_relabel(self, series_id: int, status:bool) -> None:
        """Update is_relabel for given series_id for series_status table 

        Args:
            series_id: A index of the series
            status: the new status for column 'is_relabel'
        """
        if not isinstance(status, bool):
            raise ValueError("Status should be bool.")
        self._update_seriesStatus(series_id, is_relabel=status)

    def update_is_continual(self, series_id: int, status:bool) -> None:
        """Update is_continual for given series_id for series_status table 

        Args:
            series_id: the index of the series.
            status: the new status for column 'is_continual'
        """
        if not isinstance(status, bool):
            raise ValueError("Status should be bool.")
        self._update_seriesStatus(series_id, is_continual=status)
      
    def update_is_inference(self, series_id: int, status:bool) -> None:
        """Update is_inference for given series_id for series_status table 

        Args:
            series_id: A index of the series
            status: the new status for column 'is_inference'
        """
        if not isinstance(status, bool):
            raise ValueError("Status should be bool.")
        self._update_seriesStatus(series_id, is_inference=status)

    def update_can_train(self, series_id: int) -> None:
        self._update_seriesStatus(series_id, is_continual=True, is_inference=True, is_relabel=True, is_npy=True)

    def _update_seriesStatus(self, series_id: int, 
                                    is_continual: Optional[bool]=None, 
                                    is_npy: Optional[bool]=None,
                                    is_inference: Optional[bool]=None,
                                    is_relabel: Optional[bool]=None):

        cond = {'series_id':series_id, }
        attrs = []
        values = []
        # column for 'is_continual' 
        if is_continual != None:
            attrs.append('is_continual')
            if is_continual == True:
                values.append(1)
            else:
                values.append(0)
        # column for 'is_npy' 
        if is_npy != None:
            attrs.append('is_npy')
            if is_npy == True:
                values.append(1)
            else:
                values.append(0)
        # column for 'is_inference' 
        if is_inference != None:
            attrs.append('is_inference')
            if is_inference == True:
                values.append(1)
            else:
                values.append(0)
        # column for 'is_relabel' 
        if is_relabel != None:
            attrs.append('is_relabel')
            if is_relabel == True:
                values.append(1)
            else:
                values.append(0)
        self.db.update("series_status", attrs, values=values, cond=cond)

    def _get_folder_infos(self, cond:dict) -> Dict[int, Tuple[int, int, int]]:
        """Get folder information from seriesStatus table
        Args:
            cond: A dict contain condition for this search
        """
        folders = dict()
        attrs = ['series_id', 'is_relabel']
        statuses = self.db.select("series_status", attrs=attrs, cond=cond)

        attrs =  ['patient_id','study_folder_index','series_folder_index']
        for row in statuses:
            series_id = row[0]
            is_relabel = row[1]
            cond = {'series_id': series_id, }
            rs = self.db.select("series_path", attrs=attrs, cond=cond)
            folders[series_id] = [rs[0], is_relabel]
        return folders

    def get_folder_info_by_series_id(self, series_id: int) -> Optional[Tuple[int, int, int]]:
        attrs =  ['patient_id','study_folder_index','series_folder_index']
        rs = self.db.select('series_path', attrs=attrs, cond={'series_id': series_id})
        if len(rs) == 0:
            return None

        return rs[0]
    
    def get_training_available(self) -> Dict[int, Tuple[int, int, int, datetime]]:
        """Get the information of series, which is relabeled by doctors. 

        Returns: dict, pair(series_id : (patient_id, study_folder_index, series_folder_index, add_time))
        """
        folder_infos = dict()
        attrs = ['series_id', 'created_date']
        cond = {'is_relabel':1,}        
        # logger.warning("Collect data matching is_npy=1 for testing")
        # cond = {'is_npy':1,} # for testing 
        statuses = self.db.select("series_status", attrs=attrs, cond=cond)

        attrs =  ['patient_id','study_folder_index','series_folder_index']
        for row in statuses:
            series_id = row[0]
            created_date = row[1]
            created_date = datetime.strptime(created_date, "%Y/%m/%d")
            cond = {'series_id': series_id}
            # get tuple of (patient_id, study_folder_index, series_folder_index)
            rs = self.db.select("series_path", attrs=attrs, cond=cond) 
            folder_infos[series_id] = list(rs[0]) + [created_date]
        return folder_infos
    
    def close(self):
        self.db.close()