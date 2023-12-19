from typing import Dict, List, Optional, Union, Any

class SQLBuilder(object):
    @staticmethod
    def build_insert_many_SQL(table_name:str, num_attr:int) -> str:
        """Build Insert many SQL
        Insert many SQL Example: 'INSERT INTO patient VALUES(?, ?, ?)'
        Args:
            num_attr: number of attribute for inserting
        """
        attrs = ",?" * num_attr
        attrs = attrs[1:]
        sql = "INSERT INTO {} VALUES({})".format(table_name, attrs)
        return sql
    
    @staticmethod
    def build_insert_SQL(table_name: str, 
                        values: List[Any],
                        attrs: Optional[Union[str, List[str]]] = None) -> str:
        """Build Insert SQL

        e.g 'INSERT INTO patient VALUES(0, 'NA1953620')'
        """
        attrs = SQLBuilder.gen_attrs(attrs)

        # Generate 'VALUES' part for SQL
        values_part = ''
        for i, value in enumerate(values):
            if i != 0:
                values_part += ','
            if isinstance(value, str):
                value = "'{}'".format(value)
            values_part += str(value)

        if attrs == '':
            sql = "INSERT INTO {} VALUES({})".format(table_name, values_part)
        else:
            sql = "INSERT INTO {} ({}) VALUES({})".format(table_name, attrs, values_part)
        return sql
    
    @staticmethod
    def build_select_SQL(table_name: str, 
                        attrs: Union[str, List[str]] = '*',
                        cond: Optional[Union[str, Dict[str, Any]]] = None,
                        order: Optional[Union[str, Dict[str, str]]] = None) -> str:
        """Build Select SQL

        Args:
            attrs: str(one attribute) or list(a list of attributes) 
            cond: A condition for selecting
        """
        attrs = SQLBuilder.gen_attrs(attrs, can_be_None=False)
        sql = "SELECT {} FROM {}".format(attrs, table_name)
        # If arg condition is dict of pair(key, value), then convert it into string
        sql += SQLBuilder.gen_cond(cond)
        sql +=  SQLBuilder.gen_order(order)

        return sql
  
    @staticmethod
    def build_delete_SQL(table_name: str, 
                        cond: Union[str, Dict[str, Any]]) -> str:

        cond = SQLBuilder.gen_cond(cond)
        if cond == '':
            raise ValueError("Condition should not be None!")
        sql = "DELETE FROM {} {}".format(table_name, cond)
        return sql
    
    @staticmethod
    def build_update_SQL(table_name: str, 
                        attrs: Union[str, List[str]], 
                        values: Union[Any, List[Any]],
                        cond: Union[str, Dict[str, Any]]) -> str:

        cond = SQLBuilder.gen_cond(cond)

        if not isinstance(attrs, list):
            attrs = [attrs]
        if not isinstance(values, list):
            values = [values]

        update_part = ""
        for i, (attr, value) in enumerate(zip(attrs, values)):
            if i != 0:
                update_part += ', '
            if isinstance(value, str):
                value = "'{}'".format(value)
            update_part += "{} = {}".format(attr, value)

        sql = "UPDATE {} SET {} {}".format(table_name, update_part, cond)
        return sql
    
    @staticmethod
    def build_length_of_table_SQL(table_name: str) -> str:
        return SQLBuilder.build_select_SQL(table_name, attrs="COUNT(*)")

    @staticmethod
    def create_table(table_name: str, attrs: List[str]) -> str:
        # Table Columns
        attr_part = ''
        for i, attr in enumerate(attrs):
            if i != 0:
                attr_part += ','
            attr_part += " ".join(attr)
        sql = "CREATE TABLE {}({})".format(table_name, attr_part)
        return sql

    @staticmethod
    def create_index_table(table_name: str, attrs: List[str]) -> str:
        attr_part = ""
        for i, attr in enumerate(attrs):
            if i != 0:
                attr_part += ','
            attr_part += " ".join(attr)
        sql = "CREATE INDEX index_{0} ON {0}({1})".format(table_name, attr_part)
        return sql

    @staticmethod
    def gen_cond(conds: Union[str, Dict[str, Any]], 
                oper="AND") -> str:
        """Generate condition
        """
        if conds == None:
            return ''
        elif isinstance(conds, str):
            return ' WHERE {}'.format(conds)

        # Generate condtion based on given conds dict
        oper = oper.upper()
        result = ""
        for i, (key, value) in enumerate(conds.items()):
            if i != 0:
                result += " {} ".format(oper)
            # If value is a string, add single quotation marks
            if isinstance(value, str):
                value = "'{}'".format(value)
            cond = "{} = {}".format(key, value)
            result += cond
        return ' WHERE {}'.format(result)
    
    @staticmethod
    def gen_order(orders: Optional[Union[str, Dict[str, str]]]) -> str:
        """Generate condition
        """
        if orders == None:
            return ''
        elif isinstance(orders, str):
            return ' ORDER BY {}'.format(orders)

        result = ''
        for i, (key, value) in enumerate(orders.items()):
            if i != 0:
                result += ', '
            value = value.lower()
            if value == 'up' or value == 'asc':
                value = 'ASC'
            else:
                value = 'DESC'
            order = "{} {}".format(key, value)
            result += order
        return ' ORDER BY {}'.format(result)
    
    @staticmethod
    def gen_attrs(attrs: Optional[Union[str, List[str]]], can_be_None=True) -> str:
        """Expand multiple attributes into string
        """
        if attrs == None:
            if can_be_None:
                return ''
            else:
                raise ValueError('Attributes can not be None!')
        elif isinstance(attrs, str):
            return attrs
        else:
            # Concat the attrbutes
            attrs = ', '.join(attrs)
            return attrs

