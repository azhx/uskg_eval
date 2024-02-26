python api_request_parallel_processor.py \
###
 # @Author: ygjin11 1633504509@qq.com
 # @Date: 2024-01-04 02:21:54
 # @LastEditors: ygjin11 1633504509@qq.com
 # @LastEditTime: 2024-02-12 19:18:24
 # @FilePath: /uskg_eval/gpt_baseline/run_gpt_4.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
  --requests_filepath gpt_data/gpt4_tiny.jsonl \
  --save_filepath gpt_response/gpt4_tiny.jsonl \
  --request_url https://hkust.azure-api.net/v1/chat/completions \
  --auth "Bearer 00e982c212ec44a0863c9eacb54b2165" \
  --max_requests_per_minute 128 \
  --max_tokens_per_minute 300000 \
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