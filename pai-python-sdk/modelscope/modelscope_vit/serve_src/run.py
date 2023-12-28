
import os
import io
import json

import uvicorn
from fastapi import FastAPI, Response, Request
import numpy as np

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image

# 用户指定模型，默认会被加载到当前路径下。 
MODEL_PATH = "/eas/workspace/model/"

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return json.JSONEncoder.default(self, obj)

app = FastAPI()

@app.post("/")
async def predict(request: Request):
    global p
    content = await request.body()
    img = Image.open(io.BytesIO(content))
    res = p(img)
    return Response(content=json.dumps(res, cls=NumpyEncoder), media_type="application/json")


if __name__ == '__main__':
    p = pipeline(
        Tasks.image_classification,
        model=MODEL_PATH,
    )
    uvicorn.run(app, host='0.0.0.0', port=8000)
