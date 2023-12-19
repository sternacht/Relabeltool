import os
import sqlite3
import logging
from typing import List, Tuple, Union, Dict, Any, Optional
from .sql_builder import SQLBuilder

logger = logging.getLogger(__name__)

class Database(object):
    def __init__(self, db_path: str, db_tables: dict) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path

        # Initialize tables
        self.init_table(db_tables)

    def connect(self):
        self.db = sqlite3.connect(self.db_path)
        self.cursor = self.db.cursor()

    def init_table(self, db_tables: dict) -> None:
        """Initialize Dicom Table
        """
        self.connect()
        # Get all of the name of table in database
        res = self.cursor.execute("SELECT name FROM sqlite_master")
        cur_table_names = [name[0] for name in res.fetchall()]
        # If table not be established, then create it
        self.table_names = []
        for name, table_setting in db_tables.items():
            self.table_names.append(name)
            if name not in cur_table_names:
                sql = SQLBuilder.create_table(name, table_setting)
                self._execute_sql(sql)
        self.close()

    def select(self, 
            table_name: str, 
            attrs: Union[str, List[str]] = '*', 
            cond: Optional[Union[str, Dict[str, Any]]] = None, 
            order: Optional[Union[str, Dict[str, str]]] = None) -> List[Tuple[Any]]:
        """Select several attributes from given table

        Args:
            table_name: the name of table for selecting
            attrs: 
                1. str: the name of attribute
                2. list: a list containing several name of attributes 
            cond: 
                1. str: a condition for searching
                2. dict: a dict containing multuple pairs(attribute, value), e.g: {'name':'ford', 'age':5}
            order: 
                1. str: a order for searching
                2. dict: a dict containing multuple pairs(attribute, value), value must be 'up' or 'down', e.g: {'name':'up', 'age':5}
        Returns:
            A list contains multiple attribute tuples
        """
        self._table_exist_check(table_name)
        sql = SQLBuilder.build_select_SQL(table_name, attrs, cond, order)
        rs = self._execute_sql(sql)
        
        return rs.fetchall()

    def insert(self, 
            table_name: str, 
            values: List[Any], 
            attrs: Optional[Union[str, List[str]]] = None) -> None:
        """Insert row or rows into table 

        Args:
            table_name: the name of table for inserting
            values: list(one row) or list of list(many rows) 
            attrs: list, a list of attributes
        Returns: None
        """
        self._table_exist_check(table_name)
        # Insert multiple rows
        if isinstance(values[0], list):
            if len(attrs) != 0:
                raise ValueError("When insert many row into database, the attribute cannot be specified")
            num_attr = len(values[0])
            sql = SQLBuilder.build_insert_many_SQL(table_name, num_attr)
            self._execute_insert_many(sql, values, need_commit=True)
        # Insert one row
        else:
            sql = SQLBuilder.build_insert_SQL(table_name, values, attrs)
            self._execute_sql(sql, need_commit=True)
    
    def update(self, 
            table_name: str, 
            attrs: Union[str, List[str]],
            values: Union[Any, List[Any]],
            cond: Union[str, Dict[str, Any]]) -> None:
        """Update row

        Args:
            table_name: the name of table for updating
            attrs: a list containing several names of attribute
            values: a list containing several values for given attributes
            cond:                 
                1. str: a condition for searching
                2. dict: a dict containing multuple pairs(attribute, value), e.g: {'name':'ford', 'age':5}
        Returns: None
        """
        self._table_exist_check(table_name)
        sql = SQLBuilder.build_update_SQL(table_name, attrs, values, cond)
        self._execute_sql(sql, need_commit=True)

    def delete(self, 
            table_name: str, 
            cond: Union[str, Dict[str, Any]]) -> None:
        """Delete row

        Args:
            table_name: the name of table for deleting
            cond:                 
                1. str: a condition for searching
                2. dict: a dict containing multuple pairs(attribute, value), e.g: {'name':'ford', 'age':5}
        Returns: None
        """
        self._table_exist_check(table_name)
        sql = SQLBuilder.build_delete_SQL(table_name, cond)
        self._execute_sql(sql, need_commit=True)

    def get_length_of_table(self, table_name: str) -> int:
        """Get number of row for table matching given table name
        Args:
            table_name: the name of table
        Returns: 
            length of table
        """
        self._table_exist_check(table_name)
        sql = SQLBuilder.build_length_of_table_SQL(table_name)
        rs = self._execute_sql(sql)
        length = rs.fetchone()[0]
        return length

    def _execute_sql(self, sql:str, need_commit=False) -> List[tuple]:
        logger.debug(f'Execute SQL "{sql}"')
        rs = self.cursor.execute(sql)
        if need_commit:
            self.db.commit()
        return rs

    def _table_exist_check(self, table_name:str) -> None:
        if table_name not in self.table_names:
            raise ValueError("Table {} not exist!".format(table_name))

    def _execute_insert_many(self, sql: str, data: List[str], need_commit:bool = False) -> None:
        """Insert multiple data(Always do Commit)
        """
        self.cursor.executemany(sql, data)
        if need_commit:
            self.db.commit()
 
    def close(self) -> None:
        """close connection of database safely
        """
        self.db.close()
        self.db = None
        self.cursor = None