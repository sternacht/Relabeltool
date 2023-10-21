import os
import logging
import re
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional, Union, List
from .constant import dicom_database_setting, PATIENT_ID_FORMAT, STUDY_ID_FORMAT, SERIES_ID_FORMAT, INSTANCE_NUM_FORMAT 
from .database import Database

logger = logging.getLogger(__name__)

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
    

def get_local_time_in_taiwan() -> datetime:
    utc_now = datetime.utcnow()
    taiwan_now = utc_now + timedelta(hours=8) # Taiwan in UTC+8
    return taiwan_now
    
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
    if instance_num != -1:
        file_name += '_' + INSTANCE_NUM_FORMAT.format(instance_num)
    return file_name

def gen_uids_key(patient_uid: str, stduy_uid: str, series_uid: str):
    return '{}/{}/{}'.format(patient_uid, stduy_uid, series_uid)

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
     
    def add_patient(self, patient_uid: str) -> int:
        """Add patient ID into patient table
        
        Args:
            patient_uid: string, the real patient UID for this dicom file
        Return: int
            An index of patient in patient table
        """
        patient_id = self.get_patient_id(patient_uid)
        if patient_id != None:
            return patient_id
        # Add new paitent
        attrs = ['patient_uid']            
        values = (patient_uid,)
        self.db.insert("patient", values=values, attrs=attrs)
        patient_id = self.get_patient_id(patient_uid)
        return patient_id
       
    def add_study(self, patient_id: int, study_uid: str) -> Tuple[int, int]:
        """Add study UID into study table
        
        Args:
            patient_id: integer, the index for the patient
            study_uid: string, the study UID for this dicom file
        Return: tuple
            A tuple of (study_id, folder_index), which folder_index is for this study id
        """
        # Search database by given patient id and study uid
        cond = {'patient_id': patient_id}
        attrs = ['study_id', 'study_uid', 'folder_index']
        order = {'folder_index': 'down', }
        rs = self.db.select("study", attrs, cond, order)
        # Found some rows having same patient_id
        if len(rs) != 0:
            for row in rs:
                if study_uid == row[1]:
                    study_id = row[0]
                    folder_index = row[2]
                    return study_id, folder_index
            # If not found in these row, then set folder index to the largest folder_index + 1
            folder_index = rs[0][2] + 1
        else:
            folder_index = 1 
        # Insert new study
        attrs = ['study_uid', 'folder_index', 'patient_id']
        values = (study_uid, folder_index, patient_id)
        self.db.insert("study", values, attrs)
        study_id, folder_index = self.get_study_id(patient_id, study_uid)
        return study_id, folder_index
    
    def add_series(self, patient_id:int, study_id:int, series_uid:str) -> Tuple[int, int, bool]:
        """Add series UID into series table
        
        Args:
            patient_id: integer, the patient index
            study_id: integer, the study index
            series_uid: str, the series UID for this dicom
        Return: Tuple[int, int, bool]
            tuple(series_id, folder_index, is_new_series) for this series
        """
        cond = {'patient_id':patient_id, 'study_id':study_id}
        attrs = ['series_id', 'series_uid', 'folder_index']
        order = {'folder_index': 'down', }
        rs = self.db.select("series", attrs, cond, order)

        # Found some records having same patient_id and study_id
        if len(rs) != 0:
            for row in rs:
                if series_uid == row[1]:
                    series_id = row[0]
                    folder_index = row[2]
                    return series_id, folder_index, False
            # Not Found in these records, then use (the highest folder_index + 1)
            folder_index = rs[0][2] + 1
        else:
            folder_index = 1
        # Insert new series
        attrs = ['series_uid', 'folder_index', 'patient_id', 'study_id']
        values = (series_uid, folder_index, patient_id, study_id)
        self.db.insert("series", values, attrs)
        
        rs = self.db.select("series", 
                            attrs = ['series_id'], 
                            cond = {'patient_id': patient_id, 'study_id': study_id, 'series_uid': series_uid})
        series_id = rs[0][0]       
        return series_id, folder_index, True

    def add_folder_path(self, series_id:int, patient_id:int, study_folder_index:int, series_folder_index:int)-> None:
        cond = {'series_id':series_id, }
        rs = self.db.select("series_path", cond=cond)
        # This series_id is already in the series_path table 
        if len(rs) != 0:
            return

        attrs = ['series_id', 'patient_id', 'study_folder_index', 'series_folder_index']
        values = (series_id, patient_id, study_folder_index, series_folder_index)
        self.db.insert("series_path", values, attrs)

    def add_empty_series_status(self, series_id:int) -> None:
        cond = {'series_id':series_id, }
        rs = self.db.select("series_status", cond=cond)
        # This series_id is already in the seriesStatus 
        if len(rs) != 0:
            return
        # Insert new record into series_status table
        taiwan_now = get_local_time_in_taiwan()
        cur_time = taiwan_now.strftime("%Y/%m/%d")
        attrs = ['series_id', 'is_continual', 'is_npy', 'is_inference', 'is_relabel', 'is_train', 'created_date']
        values = (series_id, 0, 0, 0, 0, 0, cur_time)
        self.db.insert("series_status", values, attrs)

    def add_new_dicom_file(self, patient_uid:str, study_uid:str, series_uid:str) -> Tuple[int, int, int]:
        patient_id = self.add_patient(patient_uid)
        study_id, study_folder_index = self.add_study(patient_id, study_uid)
        series_id, series_folder_index, is_new_series = self.add_series(patient_id, study_id, series_uid)
        if is_new_series:
            self.add_folder_path(series_id, patient_id, study_folder_index, series_folder_index)
            self.add_empty_series_status(series_id)

        return patient_id, study_folder_index, series_folder_index

    def add_new_user(self, user_name: str) -> int:
        """Add new user into user table

        Args:
            user_name: the name of new user
        Returns: int
            the index of new user
        """
        attrs = ['user_name']
        values = (user_name,)
        self.db.insert('user', values=values, attrs=attrs)
        user_id = self.get_user_id(user_name)
        return user_id

    def get_user_id(self, user_name: str) -> Optional[int]:
        """Get user id in database for given user_name
        Returns: int
            user id in database, if not found, return None
        """
        cond = {'user_name':user_name}
        rs = self.db.select("user", cond=cond)
        # Not Found
        if len(rs) == 0:
            return None
        else:
            return int(rs[0][0])

    def get_not_continual(self) -> Dict[int, Tuple[int, int, int]]:
        """Get the folder information which 'is_continual' is 0  
       
        Search series_status table and find the series_id of the row which 'is_continual' is 0, then use series_id to search series_path table to get folder

        Returns: 
            dict, pair(series_id : tuple(patient_id, study_folder_index, series_folder_index))
        """
        cond = {'is_continual':0, }
        return self._get_folder_infos(cond)
    
    def get_can_do_inference(self) -> Dict[int, Tuple[int, int, int]]:
        """Get the information of series folder which is continual, already converted into npy file and not be inference
        """
        cond = {'is_npy':1, 'is_continual':1, 'is_inference': 0}
        return self._get_folder_infos(cond)

    def get_can_do_relabel(self) -> Dict[int, Tuple[int, int, int]]:
        """Get the information of series folder which is continual, already converted into npy file and already inferenced.
        
        Returns: dict, 
            A dict of pair(series_id : tuple(patient_id, study_folder_index, series_folder_index))
        """
        cond = {'is_npy':1, 'is_continual':1, 'is_inference': 1, 'is_relabel': 0}
        return self._get_folder_infos(cond)

    def get_can_do_npy(self) -> Dict[int, Tuple[int, int, int]]:
        """Get the information of series folder which is continual and not be converted into npy file.
        """
        cond = {'is_continual':1, 'is_npy':0}
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

    def get_patient_id(self, patient_uid: str) -> Union[int, None]:
        """Get patient id in database for given patient_uid
        Returns: int
            patient id in database

        """
        cond = {'patient_uid':patient_uid}
        rs = self.db.select("patient", cond=cond)
        # Not Found
        if len(rs) == 0:
            return None
        else:
            return int(rs[0][0])

    def get_older_datas(self, num_data: int) -> List[int]:
        attrs = ['series_id', 'created_date']
        statuses = self.db.select("series_status", attrs=attrs)
        series_ids = dict()
        for row in statuses:
            series_id, created_date = row
            created_date = datetime.strptime(created_date, "%Y/%m/%d")
            series_ids[series_id] = created_date
        # Sort by created_date
        sorted_series_ids = sorted(series_ids.keys(), key=lambda k: series_ids[k])
        
        return sorted_series_ids[:num_data]

    def get_user_id_mapping(self) -> Dict[int, str]:
        """Get user id mapping

        Returns: dict
            A dict of pair(user_id : user_name)
        """
        attrs = ['user_id', 'user_name']
        rs = self.db.select("user", attrs=attrs)
        user_id_mapping = dict()
        for row in rs:
            user_id, user_name = row
            user_id_mapping[user_id] = user_name
        return user_id_mapping

    def get_can_relabel_series_path_and_relabel_status(self) -> Dict[int, Tuple[Tuple[int, int, int], Tuple[bool, str]]]:
        """Get all series path and relabel status
        Returns: dict
            A dict of pair(series_path : is_relabel)
        """
        infos = dict()
        attrs = ['series_id', 'is_relabel', 'relabel_user_id']
        cond = {'is_continual':1, 'is_npy':1, 'is_inference':1}
        statuses = self.db.select("series_status", attrs=attrs, cond=cond)

        user_id_mapping = self.get_user_id_mapping()
        attrs =  ['patient_id','study_folder_index','series_folder_index']
        for row in statuses:
            series_id, is_relabel, relabel_user_id = row
            relabel_user_name = user_id_mapping.get(relabel_user_id, None)
            cond = {'series_id': series_id, }
            rs = self.db.select("series_path", attrs=attrs, cond=cond)
            infos[series_id] = (rs[0], (is_relabel, relabel_user_name)) # rs[0] is tuple(patient_id, study_folder_index, series_folder_index)
        return infos

    def remove_series(self, series_id: int) -> str:
        """
        Returns: str
            A path of removed series. If there are not any series in study which has series whose series 
            id is same as given series id, then return a path of this study.Similarly, if there are not 
            any stduies in patient which has study whose study id is same as stduy which has series whose 
            series id is same as given series id, then return a path of this patient.
        """
        # Remove series from table 'series_path'
        rs = self.db.select('series_path', cond={'series_id': series_id}, attrs=['patient_id', 'study_folder_index', 'series_folder_index'])
        patient_id, study_folder_index, series_folder_index = rs[0]
        self.db.delete('series_path', cond={'series_id': series_id})
        # Remove series from table 'series_stauts'
        self.db.delete('series_status', cond={'series_id': series_id})
        
        removed_path = gen_dicom_path(patient_id, study_folder_index, series_folder_index)
        
        # Remove series from table 'series' and check if there are any series having same study id as removed series.
        rs = self.db.select('series', cond={'series_id': series_id}, attrs=['study_id'])
        study_id = rs[0][0]
        self.db.delete('series', cond={'series_id': series_id})
        
        # Search if there are another series having same study id.
        rs = self.db.select('series', cond={'study_id': study_id}, attrs=['series_id']) 
        # Because there are some series having same study id, just remove the folder of series.
        if len(rs) != 0:
            return removed_path # e.g 'ID-000001\std-0001\Ser-001'
        
        # Remove whole study folder, e.g: make 'ID-000001\std-0001\Ser-001' to 'ID-000001\std-0001\'
        removed_path = os.path.dirname(removed_path) 
        # Remove study from table 'study'
        self.db.delete('study', cond={'study_id': study_id})
        # Search if there are another study having same patient id.
        rs = self.db.select('study', cond={'patient_id': patient_id}, attrs=['study_id'])
        # Because there are some study having same patient id, just remove the folder of study.
        if len(rs) != 0:
            return removed_path # e.g 'ID-000001\std-0001\'

        # Remove patient from table 'patient'
        self.db.delete('patient', cond={'patient_id': patient_id})
        # Remove whole patient folder, e.g: make 'ID-000001\std-0001\' to 'ID-000001\'
        removed_path = os.path.dirname(removed_path) 
        return removed_path # e.g 'ID-000001\'
    
    def update_is_npy(self, series_id: int, status:bool) -> None:
        """Update is_npy for given series_id for series_status table 

        Args:
            series_id: A index of the series
            status: the new status for column 'is_npy'
        """
        if not isinstance(status, bool):
            raise ValueError("Status should be bool.")
        self._update_series_status(series_id, is_npy=status)

    def update_is_relabel(self, series_id: int, status:bool) -> None:
        """Update is_relabel for given series_id for series_status table 

        Args:
            series_id: A index of the series
            status: the new status for column 'is_relabel'
        """
        if not isinstance(status, bool):
            raise ValueError("Status should be bool.")
        self._update_series_status(series_id, is_relabel=status)

    def update_is_continual(self, series_id: int, status:bool) -> None:
        """Update is_continual for given series_id for series_status table 

        Args:
            series_id: the index of the series.
            status: the new status for column 'is_continual'
        """
        if not isinstance(status, bool):
            raise ValueError("Status should be bool.")
        self._update_series_status(series_id, is_continual=status)
      
    def update_is_inference(self, series_id: int, status:bool) -> None:
        """Update is_inference for given series_id for series_status table 

        Args:
            series_id: A index of the series
            status: the new status for column 'is_inference'
        """
        if not isinstance(status, bool):
            raise ValueError("Status should be bool.")
        self._update_series_status(series_id, is_inference = status)

    def update_is_train(self, series_id: int, status:bool) -> None:
        """Update is_inference for given series_id for series_status table 

        Args:
            series_id: A index of the series
            status: the new status for column 'is_inference'
        """
        if not isinstance(status, bool):
            raise ValueError("Status should be bool.")
        self._update_series_status(series_id, is_train = status)

    def update_can_train(self, series_id: int) -> None:
        """
        Set is_continual, is_inference, is_npy, is_relabel to True
        """
        self._update_series_status(series_id, is_continual = True, is_inference = True, is_relabel = True, is_npy = True)

    def _update_series_status(self, series_id: int, 
                                    is_continual: Optional[bool]=None, 
                                    is_npy: Optional[bool]=None,
                                    is_inference: Optional[bool]=None,
                                    is_relabel: Optional[bool]=None,
                                    is_train: Optional[bool]=None,):

        cond = {'series_id':series_id, }
        attrs = []
        values = []
        # column for 'is_continual' 
        if is_continual != None:
            attrs.append('is_continual')
            values.append(int(is_continual))
        # column for 'is_npy' 
        if is_npy != None:
            attrs.append('is_npy')
            values.append(int(is_npy))
        # column for 'is_inference' 
        if is_inference != None:
            attrs.append('is_inference')
            values.append(int(is_inference))
        # column for 'is_relabel' 
        if is_relabel != None:
            attrs.append('is_relabel')
            values.append(int(is_relabel))
        # column for 'is_train' 
        if is_train != None:
            attrs.append('is_train')
            values.append(int(is_train))
        self.db.update("series_status", attrs, values=values, cond=cond)

    def update_relabel_user(self, series_id: int, user_name: str) -> None:
        """Update relabel_user_id for given series_id for series_status table 

        Args:
            series_id: A index of the series
            user_name: the name of user who relabel this series
        """
        user_id = self.get_user_id(user_name)
        if user_id == None:
            user_id = self.add_new_user(user_name)
        cond = {'series_id':series_id, }
        attrs = ['relabel_user_id']
        values = [user_id]
        self.db.update("series_status", attrs, values=values, cond=cond)
        
    def get_study_id(self, patient_id: int, study_uid: str) -> Union[None, Tuple[int, int]]:
        """
        Return: tuple
            A tuple of (study_id, folder_index), which folder_index is for this study id
        """
        rs = self.db.select("study", 
                            attrs = ['study_id', 'folder_index'], 
                            cond = {'patient_id': patient_id, 'study_uid': study_uid})
        if len(rs) == 0:
            return None
        else:
            return rs[0]
        
    def _get_folder_infos(self, cond:dict) -> Dict[int, Tuple[int, int, int]]:
        """Get folder information from seriesStatus table
        Args:
            cond: A dict contain condition for this search
        """
        folders = dict()
        attrs = ['series_id']
        statuses = self.db.select("series_status", attrs=attrs, cond=cond)

        attrs =  ['patient_id','study_folder_index','series_folder_index']
        for row in statuses:
            series_id = row[0]
            cond = {'series_id': series_id, }
            rs = self.db.select("series_path", attrs=attrs, cond=cond)
            folders[series_id] = rs[0]
        return folders

    def get_folder_info_by_series_id(self, series_id: int) -> Optional[Tuple[int, int, int]]:
        attrs =  ['patient_id','study_folder_index','series_folder_index']
        rs = self.db.select('series_path', attrs=attrs, cond={'series_id': series_id})
        if len(rs) == 0:
            return None

        return rs[0]
    
    def get_not_training(self) -> Dict[int, Tuple[int, int, int, datetime]]:
        """Get the information of series, which is relabeled by doctors and not be trained. 

        Returns: dict, pair(series_id : (patient_id, study_folder_index, series_folder_index, add_time))
        """
        folder_infos = dict()
        attrs = ['series_id', 'created_date']
        cond = {'is_relabel' : 1, 
                'is_train': 0}
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
    
    def get_already_training(self) -> Dict[int, Tuple[int, int, int, datetime]]:
        """Get the information of series, which is relabeled by doctors and be already trained. 

        Returns: dict, pair(series_id : (patient_id, study_folder_index, series_folder_index, add_time))
        """
        folder_infos = dict()
        attrs = ['series_id', 'created_date']
        cond = {'is_relabel' : 1, 
                'is_train': 1}
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

    def get_training_available(self) -> Dict[int, Tuple[int, int, int, datetime]]:
        """Get the information of series, which is relabeled by doctors and be already trained. 

        Returns: dict, pair(series_id : (patient_id, study_folder_index, series_folder_index, add_time))
        """
        folder_infos = dict()
        attrs = ['series_id', 'created_date']
        cond = {'is_relabel' : 1}
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

    def get_uid_path_mapping(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Returns: Dict[str, Tuple[int, int, int]]
            A Dict of ('patient_uid/study_uid/series_uid', [patient_id, study_folder_index, series_folder_index]) pair
        """
        series_paths = self.db.select('series_path', attrs = ['series_id', 'patient_id', 'study_folder_index', 'series_folder_index'])
        patient_info = self.db.select('patient', attrs = ['patient_id', 'patient_uid'])
        study_info = self.db.select('study', attrs = ['study_id', 'study_uid'])
        series_info = self.db.select('series', attrs = ['series_id', 'series_uid', 'patient_id', 'study_id'])
        
        patient_id_uid_mapping = {patient_id: patient_uid for patient_id, patient_uid in patient_info}
        study_id_uid_mapping = {study_id: study_uid for study_id, study_uid in study_info}
        series_id_path_mapping = {series_id: [patient_id, study_folder_index, series_folder_index] for series_id, patient_id, study_folder_index, series_folder_index in series_paths}
        
        series_uid_path_mapping = dict()
        for series_id, series_uid, patient_id, study_id in series_info:
            path = series_id_path_mapping[series_id]
            patient_uid = patient_id_uid_mapping[patient_id]
            study_uid = study_id_uid_mapping[study_id]
            uids_key = gen_uids_key(patient_uid, study_uid, series_uid)
            series_uid_path_mapping[uids_key] = path
        return series_uid_path_mapping
    
    def close(self):
        self.db.close()