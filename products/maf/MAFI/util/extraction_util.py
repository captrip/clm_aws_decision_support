import json
import re
def llm_output_parser(llm_response,outputClass_object):
    think_tag = re.search(r"<think>(.*?)</think>",llm_response,re.DOTALL)
    thought = think_tag.group().strip() if think_tag else None
    actual_json_response_match = re.search(r"\{.*\}",llm_response,re.DOTALL)
    return thought,json.loads(actual_json_response_match.group())