# source: https://github.com/THUDM/ChatGLM-6B/blob/main/api.py

import os

from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel, AutoConfig
import uvicorn, json, datetime
import torch


model = None
tokenizer = None

# 默认的模型保存路径
chatglm_model_path = "/eas/workspace/model/"
# ptuning checkpoints保存路径
ptuning_checkpoint = "/ml/ptuning_checkpoints/"
pre_seq_len = 128
app = FastAPI()


def load_model():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        chatglm_model_path, trust_remote_code=True
    )

    if os.path.exists(ptuning_checkpoint):
        # P-tuning v2
        print(f"Loading model/ptuning_checkpoint weight...")
        config = AutoConfig.from_pretrained(chatglm_model_path, trust_remote_code=True)
        config.pre_seq_len = pre_seq_len
        config.prefix_projection = False

        model = AutoModel.from_pretrained(
            chatglm_model_path, config=config, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            chatglm_model_path, trust_remote_code=True
        )
        prefix_state_dict = torch.load(
            os.path.join(ptuning_checkpoint, "pytorch_model.bin")
        )
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder.") :]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

        model = model.half().cuda()
        model.transformer.prefix_encoder.float().cuda()
        model.eval()
    else:
        print(f"Loading model weight...")
        model = AutoModel.from_pretrained(chatglm_model_path, trust_remote_code=True)
        model.half().cuda()
        model.eval()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get("prompt")
    history = json_post_list.get("history")
    max_length = json_post_list.get("max_length")
    top_p = json_post_list.get("top_p")
    temperature = json_post_list.get("temperature")
    response, history = model.chat(
        tokenizer,
        prompt,
        history=history,
        max_length=max_length if max_length else 2048,
        top_p=top_p if top_p else 0.7,
        temperature=temperature if temperature else 0.95,
    )
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {"response": response, "history": history, "status": 200, "time": time}
    log = (
        "["
        + time
        + "] "
        + '", prompt:"'
        + prompt
        + '", response:"'
        + repr(response)
        + '"'
    )
    print(log)
    return answer


if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
