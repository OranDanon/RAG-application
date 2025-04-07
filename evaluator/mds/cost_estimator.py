import tiktoken
import os

folder_path = ""
encoding = tiktoken.encoding_for_model("text-embedding-3-small")
total_tokens = 0

for filename in os.listdir():
    if filename.endswith(".md"):
        with open(os.path.join(filename), "r", encoding="utf-8") as f:
            text = f.read()
            tokens = encoding.encode(text)
            total_tokens += len(tokens)

print(f"Total tokens: {total_tokens}")
print(f"Estimated cost: ${(total_tokens / 1000) * 0.00002:.6f}")
