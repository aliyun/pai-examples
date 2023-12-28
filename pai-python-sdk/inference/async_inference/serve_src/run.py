import asyncio
from random import random

from fastapi import FastAPI, Request
import uvicorn, json, datetime

# 默认模型加载路径
model_path = "/eas/workspace/model/"

app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    print("Make mock prediction starting ...")
    # Mock prediction
    await asyncio.sleep(15)
    print("Prediction finished.")
    return [random() for _ in range(10)]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
