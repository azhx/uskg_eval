import os

# remove the datasets 
dataset_names = ['totto', 'webqsp', 'msr_sqa', 'sql2text', 'spider', 'cosql', 'kvret', 'hybridqa', 'sparc', 'grailqa', 'compwebq', 'tab_fact', 'wikitq', 'wikisql_tapas', 'mmqa', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop']
path_template = "/home/alex/v3-score/UnifiedSKG/data/compwebq/*"
for dataset_name in dataset_names:
    path = path_template.replace("compwebq", dataset_name)
    os.system(f"rm -rf {path}")
    print(f"removed {path}")
