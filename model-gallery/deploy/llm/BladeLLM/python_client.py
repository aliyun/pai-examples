import json
import requests

eas_url = "<EAS API Endpoint>"
eas_key = "<EAS API KEY>"

url = f"{eas_url}/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": eas_key,
}


def main():
    stream = True
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，请介绍一下你自己。"},
    ]
    req = {
        "messages": messages,
        "stream": stream,
        "temperature": 0.0,
        "top_p": 0.5,
        "top_k": 10,
        "max_tokens": 300,
    }
    response = requests.post(
        url,
        json=req,
        headers=headers,
        stream=stream,
    )

    if stream:
        for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False):
            msg = chunk.decode("utf-8")
            if msg.startswith("data"):
                info = msg[6:]
                if info == "[DONE]":
                    break
                else:
                    resp = json.loads(info)
                    print(resp["choices"][0]["delta"]["content"], end="", flush=True)
    else:
        resp = json.loads(response.text)
        print(resp["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
