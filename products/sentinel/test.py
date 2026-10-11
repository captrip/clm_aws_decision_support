from anthropic import AnthropicBedrockMantle

client = AnthropicBedrockMantle(
    aws_region="us-east-1",
    aws_profile="258574424891",
    default_headers={"anthropic-workspace-id": "default"},
)

with client.messages.stream(
    model="anthropic.claude-haiku-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Summarize this document..."}],
) as stream:

    for text in stream.text_stream:
        print(text, end="", flush=True)