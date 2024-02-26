# build the jsonl files to call gpt4/ gpt3.5 on for all datasets

import json

# load the processed_data/v11_all_prompt_input_test.json
with open('processed_data/v11_all_inst_train.json', 'r') as f:
    all_train_data = json.load(f)
with open('processed_data/v11_all_inst_test.json', 'r') as f:
    all_test_data = json.load(f)

def get_example(dataset, source, n =1):
    # get an example with source from the dataset
    lst = []
    for example in dataset:
        if source in example['arg_path']:
            lst.append(example['formatted_input'] + example['seq_out'])
    # choose n random examples from lst

    import random
    random.shuffle(lst)
    return lst[:n]

one_shot_examples = {}
datasets = set([each['arg_path'] for each in all_train_data])
for each in datasets:
    ex = get_example(all_train_data, each, 1)[0]
    one_shot_examples[each] = ex[ex.find("### Input"):]

# write the one_shot_examples to a json file
# infotabs and finqa need to be fiddled with.
with open('one_shot_gpt_examples.json', 'w') as f:
    
    json.dump(one_shot_examples, f, indent=4)

gpt35_data = []
gpt4_data = []
gpt35_tiny = []
gpt4_tiny = []
example_dps = {}
for i, ex in enumerate(all_test_data):
    finput = ex['formatted_input'].split("### Instruction:\n")[1]
    inst = finput[:finput.find("### Input")]
    suffix = finput[finput.find("### Input"):]
    msgs = [
            {
                "role": "user",
                "content": f"{inst}### Example:\n{one_shot_examples[ex['arg_path']]}\n\n{suffix}"
            }
    ]
    gpt35_data.append({
        "model": "gpt-3.5-turbo",
        "messages": msgs
    })
    gpt4_data.append({
        "model": "gpt-4",
        "messages": msgs
    })
    if ex['arg_path'] not in example_dps:
        example_dps[ex['arg_path']] = msgs[0]['content']
        gpt35_tiny.append({
            "model": "gpt-3.5-turbo",
            "messages": msgs,
            "max_tokens": 300,
        })
        gpt4_tiny.append({
            "model": "gpt-4",
            "messages": msgs,
            "max_tokens": 300,
        })

# printout of example_dps
for k, v in example_dps.items():
    print(k)
    print(v)

# write out gpt35 and gpt4 data to jsonl files
with open('processed_data/gpt35_test.jsonl', 'w') as f:
    for each in gpt35_data:
        json.dump(each, f)
        f.write('\n')

with open('processed_data/gpt35_tiny.jsonl', 'w') as f:
    for each in gpt35_tiny:
        json.dump(each, f)
        f.write('\n')

with open('processed_data/gpt4_test.jsonl', 'w') as f:
    for each in gpt4_data:
        json.dump(each, f)
        f.write('\n')

with open('processed_data/gpt4_tiny.jsonl', 'w') as f:
    for each in gpt4_tiny:
        json.dump(each, f)
        f.write('\n')

"""
python examples/api_request_parallel_processor.py \
  --requests_filepath processed_data/gpt4_tiny.jsonl \
  --save_filepath output/gpt4_tiny/responses.jsonl \
  --request_url https://yeqiu-gpt4-3.xyhelper.cn/v1/chat/completions \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 6250000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20

python api_request_parallel_processor.py \
  --requests_filepath processed_data/gpt35_tiny.jsonl \
  --save_filepath output/gpt35_tiny/responses.jsonl \
  --request_url http://121.127.44.54:8102/v1/chat/completions \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 6250000 \
  --max_attempts 5 \
  --logging_level 20


curl --location 'http://121.127.44.54:8102/v1/chat/completions' \
--header 'Authorization: Bearer 113951af-e601-4c0c-9250-7a2a97a6cba6' \
--header 'Content-Type: application/json' \
--data '{
    "model": "gpt-3.5-turbo",
    "user": 1,
    "messages": [
        {
            "role": "user",
            "content": "1+1=？"
        }
    ],
    "stream":false
}'

curl --location 'http://121.127.44.53:8102/v1/chat/completions' \
--header 'Authorization: Bearer 8743c0de-6614-456a-a1ac-2f0d5069823d' \
--header 'Content-Type: application/json' \
--data '{
    "model": "gpt-4",
    "messages": [
        {
            "role": "user",
            "content": "1+1=？"
        }
    ],
    "stream":false
}'

python api_request_parallel_processor.py \
  --requests_filepath processed_data/gpt4_tiny.jsonl \
--save_filepath output/gpt4_tiny/responses.jsonl \
  --request_url https://yeqiu-gpt4-3.xyhelper.cn/v1/chat/completions \
  --auth "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJncmVnQG1lbGluZGFzLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctbkgxWkhtV0U3RGVLZjBUSDZmOU5VU2k2IiwidXNlcl9pZCI6InVzZXItZWRCcFNYWWRFWEFqNHFGR3pyTWdLUVFKIn0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJhdXRoMHw2Mzk2NDkzNjg4NzY0OGQzYjk5OTZmMTUiLCJhdWQiOlsiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS92MSIsImh0dHBzOi8vb3BlbmFpLm9wZW5haS5hdXRoMGFwcC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNjk4NTc1MDIwLCJleHAiOjE2OTk0MzkwMjAsImF6cCI6IlRkSkljYmUxNldvVEh0Tjk1bnl5d2g1RTR5T282SXRHIiwic2NvcGUiOiJvcGVuaWQgZW1haWwgcHJvZmlsZSBtb2RlbC5yZWFkIG1vZGVsLnJlcXVlc3Qgb3JnYW5pemF0aW9uLnJlYWQgb3JnYW5pemF0aW9uLndyaXRlIG9mZmxpbmVfYWNjZXNzIn0.JeRVBNeR2kpJEoB2DsSodJqNp80a8tWel6c7OutzHlh3v2-HLGImAqhNdPSsU4t6GhpnpHmoVF-l-IBOehuJrvMNZ3nD2wi5rJhb6NPzIOGv3TsOoff7j08qfPHQY4ezDPWIdgNe-T_oniZ-SsbdVpOWrCtIo3HO_3ZzXaxE4LHfFhTezIyj6V9ADEcwsSg_u4-Z66BUcAKDGQGUwpYbfC9wwcnwabtyztg8XPr0abfsbNrYJh09ICSAKtdlwmXT3yMtO1iUEkylCIH_xSWknJOdFOqNvBWdxrg15uW_BYYwpFGgL_cnKiHUbvwzIwqJliZTCvrbZFShi09yhWurLQ" \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 3000000 \
  --max_attempts 5 \
  --logging_level 20

python api_request_parallel_processor.py \
  --requests_filepath processed_data/gpt4_test.jsonl \
  --save_filepath output/gpt4_test/responses.jsonl \
  --request_url https://yeqiu-gpt4-3.xyhelper.cn/v1/chat/completions \
  --auth "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJncmVnQG1lbGluZGFzLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctbkgxWkhtV0U3RGVLZjBUSDZmOU5VU2k2IiwidXNlcl9pZCI6InVzZXItZWRCcFNYWWRFWEFqNHFGR3pyTWdLUVFKIn0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJhdXRoMHw2Mzk2NDkzNjg4NzY0OGQzYjk5OTZmMTUiLCJhdWQiOlsiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS92MSIsImh0dHBzOi8vb3BlbmFpLm9wZW5haS5hdXRoMGFwcC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNjk4NTc1MDIwLCJleHAiOjE2OTk0MzkwMjAsImF6cCI6IlRkSkljYmUxNldvVEh0Tjk1bnl5d2g1RTR5T282SXRHIiwic2NvcGUiOiJvcGVuaWQgZW1haWwgcHJvZmlsZSBtb2RlbC5yZWFkIG1vZGVsLnJlcXVlc3Qgb3JnYW5pemF0aW9uLnJlYWQgb3JnYW5pemF0aW9uLndyaXRlIG9mZmxpbmVfYWNjZXNzIn0.JeRVBNeR2kpJEoB2DsSodJqNp80a8tWel6c7OutzHlh3v2-HLGImAqhNdPSsU4t6GhpnpHmoVF-l-IBOehuJrvMNZ3nD2wi5rJhb6NPzIOGv3TsOoff7j08qfPHQY4ezDPWIdgNe-T_oniZ-SsbdVpOWrCtIo3HO_3ZzXaxE4LHfFhTezIyj6V9ADEcwsSg_u4-Z66BUcAKDGQGUwpYbfC9wwcnwabtyztg8XPr0abfsbNrYJh09ICSAKtdlwmXT3yMtO1iUEkylCIH_xSWknJOdFOqNvBWdxrg15uW_BYYwpFGgL_cnKiHUbvwzIwqJliZTCvrbZFShi09yhWurLQ" \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 300000 \
  --max_attempts 5 \
  --logging_level 20

  python api_request_parallel_processor.py \
  --requests_filepath processed_data/gpt4_test.jsonl \
  --save_filepath output/gpt4_test/responses.jsonl \
  --request_url http://121.127.44.53:8102/v1/chat/completions \
  --auth "Bearer 8743c0de-6614-456a-a1ac-2f0d5069823d" \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 300000 \
  --max_attempts 5 \
  --logging_level 20

python api_request_parallel_processor.py \
  --requests_filepath processed_data/gpt4_tiny.jsonl \
  --save_filepath output/gpt4_tiny/responses.jsonl \
  --request_url http://121.127.44.53:8102/v1/chat/completions \
  --auth "Bearer 8743c0de-6614-456a-a1ac-2f0d5069823d" \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 300000 \
  --max_attempts 5 \
  --logging_level 20

curl --location 'https://yeqiu-gpt4-3.xyhelper.cn/v1/chat/completions' \
--header 'Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJncmVnQG1lbGluZGFzLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctbkgxWkhtV0U3RGVLZjBUSDZmOU5VU2k2IiwidXNlcl9pZCI6InVzZXItZWRCcFNYWWRFWEFqNHFGR3pyTWdLUVFKIn0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJhdXRoMHw2Mzk2NDkzNjg4NzY0OGQzYjk5OTZmMTUiLCJhdWQiOlsiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS92MSIsImh0dHBzOi8vb3BlbmFpLm9wZW5haS5hdXRoMGFwcC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNjk4NTc1MDIwLCJleHAiOjE2OTk0MzkwMjAsImF6cCI6IlRkSkljYmUxNldvVEh0Tjk1bnl5d2g1RTR5T282SXRHIiwic2NvcGUiOiJvcGVuaWQgZW1haWwgcHJvZmlsZSBtb2RlbC5yZWFkIG1vZGVsLnJlcXVlc3Qgb3JnYW5pemF0aW9uLnJlYWQgb3JnYW5pemF0aW9uLndyaXRlIG9mZmxpbmVfYWNjZXNzIn0.JeRVBNeR2kpJEoB2DsSodJqNp80a8tWel6c7OutzHlh3v2-HLGImAqhNdPSsU4t6GhpnpHmoVF-l-IBOehuJrvMNZ3nD2wi5rJhb6NPzIOGv3TsOoff7j08qfPHQY4ezDPWIdgNe-T_oniZ-SsbdVpOWrCtIo3HO_3ZzXaxE4LHfFhTezIyj6V9ADEcwsSg_u4-Z66BUcAKDGQGUwpYbfC9wwcnwabtyztg8XPr0abfsbNrYJh09ICSAKtdlwmXT3yMtO1iUEkylCIH_xSWknJOdFOqNvBWdxrg15uW_BYYwpFGgL_cnKiHUbvwzIwqJliZTCvrbZFShi09yhWurLQ' \
--header 'Content-Type: application/json' \
--header 'limit: false' \
--data '{
    "model": "gpt-4",
    "messages": [
        {
            "role": "user",
            "content": "西红柿炒钢丝球怎么做？"
        }
    ],
    "stream": false
}'
"""

