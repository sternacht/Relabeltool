class DataType:
    integer = 'INTEGER'
    text = 'TEXT'
    float = 'REAL'
class Constraint:
    primary_key = 'PRIMARY KEY'
    unique = 'UNIQUE'
    auto_increment = 'AUTOINCREMENT'
    not_null = 'NOT NULL'

########################################## DICOM Database Setting ##########################################
PATIENT_ID_FORMAT = 'ID-{:06d}'
STUDY_ID_FORMAT = 'Std-{:04d}'
SERIES_ID_FORMAT = 'Ser-{:03d}'
INSTANCE_NUM_FORMAT = '{:04d}'

patient_foreign_key_setting = 'FOREIGN KEY(patient_id) REFERENCES patient(patient_id)'

patient_table = [('patient_id', DataType.integer, Constraint.primary_key, Constraint.auto_increment),
                ('patient_uid', DataType.text, Constraint.unique)]

study_table = [('study_id', DataType.integer, Constraint.primary_key, Constraint.auto_increment),
            ('study_uid', DataType.text, Constraint.unique),
            ('folder_index', DataType.integer, Constraint.not_null),
            ('patient_id', DataType.integer, Constraint.not_null),
            (patient_foreign_key_setting,)
            ]

series_table = [('series_id', DataType.integer, Constraint.primary_key, Constraint.auto_increment),
                ('series_uid', DataType.text, Constraint.unique),
                ('folder_index', DataType.integer, Constraint.not_null),
                ('patient_id', DataType.integer, Constraint.not_null),
                ('study_id', DataType.integer, Constraint.not_null),
                (patient_foreign_key_setting,),
                ('FOREIGN KEY(study_id) REFERENCES study(study_id)',)
                ]

series_status_table = [('series_id', DataType.integer, Constraint.primary_key),
                    ('is_continual', DataType.integer, Constraint.not_null),
                    ('is_npy', DataType.integer, Constraint.not_null),
                    ('is_inference', DataType.integer, Constraint.not_null),
                    ('is_relabel', DataType.integer, Constraint.not_null),
                    ('is_train', DataType.integer, Constraint.not_null),
                    ('created_date', DataType.text, Constraint.not_null),
                    ('relabel_user_id', DataType.integer),
                    ('FOREIGN KEY(series_id) REFERENCES series(series_id)',)
                   ]

series_path_table = [('series_id', DataType.integer, Constraint.primary_key),
                    ('patient_id', DataType.integer, Constraint.not_null),
                    ('study_folder_index', DataType.integer, Constraint.not_null),
                    ('series_folder_index', DataType.integer, Constraint.not_null),
                    ('FOREIGN KEY(series_id) REFERENCES series(series_id)',),
                    (patient_foreign_key_setting,)
                    ]

user_table = [('user_id', DataType.integer, Constraint.primary_key, Constraint.auto_increment),
              ('user_name', DataType.text, Constraint.unique)]

dicom_database_setting = {'patient': patient_table,
                        'study': study_table, 
                        'series': series_table, 
                        'series_status': series_status_table, 
                        'series_path': series_path_table,
                        'user': user_table}

########################################## FL Database Setting ##########################################
class FlTrainingStatus:
    finish = 'finish'
    fail = 'fail'
    in_progress = 'in_progress'

fl_traininig_table = [('id', DataType.integer, Constraint.primary_key, Constraint.auto_increment),
                        ('name', DataType.text, Constraint.not_null),
                        ('start_date', DataType.text, Constraint.not_null),
                        ('end_date', DataType.text),
                        ('status', DataType.text, Constraint.not_null),
                    ]

model_table_template = [('id', DataType.integer, Constraint.primary_key, Constraint.auto_increment),
                        ('fl_training_id', DataType.integer),
                        ('name', DataType.text, Constraint.not_null),
                        ('metrics', DataType.text, Constraint.not_null),
                        ('round', DataType.integer, Constraint.not_null),
                        ('experiment_name', DataType.text, Constraint.not_null),
                        ('FOREIGN KEY(fl_training_id) REFERENCES fl_traininig(id)',),
                        ('UNIQUE(fl_training_id, name)',),
                        ]

num_of_stage = 2
fl_database_setting = {'fl_training': fl_traininig_table}


########################################## Process Database Setting ##########################################
running_process_table = [('id', DataType.integer, Constraint.primary_key, Constraint.auto_increment),
                        ('work', DataType.text),
                        ('process_id', DataType.text),
                        ('check_time', DataType.integer)]

process_database_setting = {'running_process': running_process_table}