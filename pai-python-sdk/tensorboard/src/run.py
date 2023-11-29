
import os

import torch
from torch.utils.tensorboard import SummaryWriter


# 通过环境变量获取TensorBoard输出路径，默认为 /ml/output/tensorboard/
tb_log_dir = os.environ.get("PAI_OUTPUT_TENSORBOARD")
print(f"TensorBoard log dir: {tb_log_dir}")
writer = SummaryWriter(log_dir=tb_log_dir)

def train_model(iter):


    x = torch.arange(-5, 5, 0.1).view(-1, 1)
    y = -5 * x + 0.1 * torch.randn(x.size())

    model = torch.nn.Linear(1, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    train_model(100)
    writer.flush()


