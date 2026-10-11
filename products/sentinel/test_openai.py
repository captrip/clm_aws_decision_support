# Save as test.py and run: python test.py

import os
os.environ.setdefault("AWS_PROFILE", "258574424891")

from openai import OpenAI
from aws_bedrock_token_generator import provide_token

client = OpenAI(
    api_key=provide_token(region="us-east-1"),
    base_url="https://bedrock-mantle.us-east-1.api.aws/v1",
    project="default",
)

response = client.chat.completions.create(
    model = "openai.gpt-5.6-luna",
    # model="openai.gpt-oss-120b",
    messages=[{"role": "user", "content": "What is Amazon Bedrock?"}],
)
print(response.choices[0].message.content)