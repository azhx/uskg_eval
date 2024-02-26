# curl --location --request POST 'http://45.61.188.16:3000/v1/chat/completions' \
# --header 'Authorization: Bearer sk-2HDAhuFqIlRrc0yi2756229dF4F4447eA3FbF87cD70a8b37' \
# --header 'User-Agent: Apifox/1.0.0 (https://www.apifox.cn)' \
# --header 'Content-Type: application/json' \
# --header 'Accept: */*' \
# --header 'Host: 45.61.188.16:3000' \
# --header 'Connection: keep-alive' \
# --data-raw '{
#     "model": "gpt-3.5-turbo",
#     "messages": [
#       {
#         "role": "system",
#         "content": "You are a helpful assistant."
#       },
#       {
#         "role": "user",
#         "content": "Who won the world series in 2020?"
#       },
#       {
#         "role": "assistant",
#         "content": "The Los Angeles Dodgers won the World Series in 2020."
#       },
#       {
#         "role": "user",
#         "content": "Where was it played?"
#       }
#     ]
#   }'
# python api_request_parallel_processor.py \
#   --requests_filepath gpt_data/gpt35_tiny.jsonl \
#   --save_filepath outputs/gpt35_tiny/gpt35_tiny.jsonl \
#   --request_url http://45.61.188.16:3000/v1/chat/completions \
#   --auth "Bearer sk-2HDAhuFqIlRrc0yi2756229dF4F4447eA3FbF87cD70a8b37" \
#   --max_requests_per_minute 128 \
#   --max_tokens_per_minute 300000 \
#   --max_attempts 5 \
#   --logging_level 20


SIZE=test

python api_request_parallel_processor.py \
  --requests_filepath "gpt_baseline/gpt_data/gpt35_${SIZE}.jsonl" \
  --save_filepath "/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/output/gpt35_${SIZE}/gpt35_${SIZE}.jsonl" \
  --request_url http://45.61.188.16:3000/v1/chat/completions \
  --auth "Bearer sk-2HDAhuFqIlRrc0yi2756229dF4F4447eA3FbF87cD70a8b37" \
  --max_requests_per_minute 1000 \
  --max_tokens_per_minute 3000000 \
  --max_attempts 5 \
  --logging_level 20




# curl --location 'http://167.71.105.205:80/v1/chat/completions' \
#     --header 'Authorization: Bearer Fy9tGqVd11QjwI99r44Ic587S34ZF7SU0VO1xnT1mcD3tH3M' \
#     --header 'Content-Type: application/json' \
#     --data '{
#         "model": "gpt-4-1106-preview",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "你的知识库截止时间？"
#             }
#         ],
#         "stream": false
#     }'

