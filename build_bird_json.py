import json
import sqlite3
from tqdm import tqdm
from utils import format_table_schema, construct_db_key_map

def get_database_schema_tables(db_id):
    """Get database schema tables using the tables.json file"""
    with open('./ukg_data/bird_train/train_databases/patched_train_tables.json', 'r') as f:
        tables = json.load(f)
    for table in tables:
        if table['db_id'] == db_id:
            break
    assert table['db_id'] == db_id
    table_info = construct_db_key_map(table)
    schema = {}
    for table_name in table_info:
        schema[table_name] = []
        header_values = []
        for column_name in table_info[table_name]:
            if 'primary_key' in table_info[table_name][column_name]:
                header_values.append(f"{column_name} (primary key)")
            elif 'foreign_key' in table_info[table_name][column_name]:
                foreign_key = table_info[table_name][column_name]['foreign_key']
                header_values.append(f"{column_name} (foreign key to `{foreign_key['col']}` in `{foreign_key['table']}`)")
            else:
                header_values.append(column_name)
        schema[table_name].append("|".join(header_values))
        # select 5 rows of data from this table
        con = sqlite3.connect(f'ukg_data/bird_train/train_databases/{db_id}/{db_id}.sqlite')
        con.text_factory = lambda b: b.decode(errors = 'ignore')
        cur = con.cursor()
        cur.execute("SELECT * FROM \"{}\" LIMIT 5;".format(table_name))
        rows = cur.fetchall()
        for row in rows:
            schema[table_name].append("|".join([str(each) for each in row]))
    return schema

def get_ukg_linearized_schema(db_id):
    with open('./ukg_data/bird_train/train_databases/patched_train_tables.json', 'r') as f:
        tables = json.load(f)
    for table in tables:
        if table['db_id'] == db_id:
            break
    assert table['db_id'] == db_id
    linearized_schema = f"| {db_id}"
    table_info = construct_db_key_map(table)
    for table_name in table_info:
        linearized_schema += f" | {table_name} :"
        column_names = [column_name for column_name in table_info[table_name]]
        linearized_schema += f" {' , '.join(column_names)}"
    return linearized_schema

if __name__ == '__main__':

    # we should include the table schema, and format it like we did in the cosql dataset.
    with open('./ukg_data/bird_train/train.json') as f:
        data = json.load(f)

    with open('./ukg_data/bird_train/train_databases/patched_train_tables.json') as f:
        table_schemas = json.load(f)

    # constuct a dictionary of table schemas
    ts_map = {table['db_id']: table for table in table_schemas}

    for i in tqdm(range(len(data))):
        data[i]['table_schema'] = ts_map[data[i]['db_id']]
        #data[i]['struct_in'] = format_table_schema(get_database_schema_tables(data[i]['db_id']))
        data[i]['struct_in'] = get_ukg_linearized_schema(data[i]['db_id'])
        data[i]['text_in'] = data[i]['question']
        data[i]['seq_out'] = data[i]['SQL']
    # now we want to format it using the same format as the cosql dataset

    # save the resulting json
    with open('./ukg_data/bird_train.json', 'w') as f:
        # dump properly indented
        json.dump(data, f, indent=4)




