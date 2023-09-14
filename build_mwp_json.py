import json
import sqlite3
from tqdm import tqdm
from utils import format_table_schema, construct_db_key_map

def format_table(table, row_num):
    """Format a table into a linearized string
    col : a | b | c row 1 : 1 | 2 | 3 row 2 : 4 | 5 | 6, etc.
    """
    colnames = list(table.keys())
    rows = []
    col_str = "col : " + " | ".join(colnames)
    for i in range(row_num):
        row = []
        for colname in colnames:
            row.append(str(table[colname][i]))
        rows.append(f"row {i+1} : "+ " | ".join(row))
    return col_str + " " + " ".join(rows)

if __name__ == '__main__':

    # we should include the table schema, and format it like we did in the cosql dataset.
    with open('./UnifiedSKG/data/downloads/extracted/e6b31a31a315f4c6c5f7852bf3f90ec2cdcb76a426428a7890ca27ee2e120b6b/tabmwp/problems_train.json') as f:
        data = json.load(f)

    newdata = []
    for k, v in data.items():
        #data[i]['struct_in'] = format_table_schema(get_database_schema_tables(data[i]['db_id']))
        ex = {}
        ex['struct_in'] = format_table(v['table_for_pd'], v['row_num']-1)
        ex['table_for_pd'] = v['table_for_pd']
        ex['text_in'] = v['question']
        ex['solution'] = v['solution']
        ex['choices'] = v['choices']
        ex['grade'] = v['grade']
        ex['seq_out'] = v['answer']
        newdata.append(ex)
    # now we want to format it using the same format as the cosql dataset

    # save the resulting json
    with open('./ukg_data/tabmwp_train.json', 'w') as f:
        # dump properly indented
        json.dump(newdata, f, indent=4)




