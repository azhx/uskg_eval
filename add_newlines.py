import json

def replace_row_number(string):
    import re
    pattern = r"row \d+ :"
    new_string = re.sub(pattern, lambda m: "\n" + m.group(0), string)
    return new_string

with open("llama_data_v7_non_upsampled.json", "rb") as f:
    data = json.load(f)

for i, ex in enumerate(data):
    if "struct_in" in ex and len(ex["struct_in"]) > 0:
        data[i]["struct_in"] = replace_row_number(ex["struct_in"])

with open("llama_data_v7_non_upsampled.json", "w") as f:
    json.dump(data, f, indent=4)
    