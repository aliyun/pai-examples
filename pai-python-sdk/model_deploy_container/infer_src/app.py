import json
from flask import Flask, request
import os
import torch
import numpy as np

app = Flask(__name__)
model = None
# 默认的模型文件路径
MODEL_PATH = "/eas/workspace/model/"


def load_model():
    """加载模型"""
    global model
    model = torch.jit.load(os.path.join(MODEL_PATH, "toy_model.pt"))
    model.eval()


@app.route("/", methods=["POST"])
def predict():
    data = np.asarray(json.loads(request.data)).astype(np.float32)
    output_tensor = model(torch.from_numpy(data))
    pred_res = output_tensor.detach().cpu().numpy()
    return json.dumps(pred_res.tolist())


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=int(os.environ.get("LISTENING_PORT", 8000)))
