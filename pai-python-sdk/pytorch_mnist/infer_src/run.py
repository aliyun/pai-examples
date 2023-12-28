import json
from flask import Flask, request
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import numpy as np
import io

app = Flask(__name__)
# 用户指定模型，默认会被加载到当前路径下。
MODEL_PATH = "/eas/workspace/model/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(
    os.path.join(MODEL_PATH, "mnist_cnn.pt"), map_location=device
).to(device)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


@app.route("/", methods=["POST"])
def predict():
    # 预处理图片数据
    im = Image.open(io.BytesIO(request.data))
    input_tensor = transform(im).to(device)
    input_tensor.unsqueeze_(0)
    # 使用模型进行推理
    output_tensor = model(input_tensor)
    pred_res = output_tensor.detach().cpu().numpy()[0]

    return json.dumps(pred_res.tolist())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("LISTENING_PORT", 8000)))
