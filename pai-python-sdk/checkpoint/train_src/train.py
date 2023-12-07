# Additional information
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


EPOCH = 5
CHECKPOINT_NAME = "checkpoint.pt"
LOSS = 0.4

# Define a custom mock dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(10)  # Generating random input tensor
        y = torch.randint(0, 2, (1,)).item()  # Generating random target label (0 or 1)
        return x, y


# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


net = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)
start_epoch = 0


def load_checkpoint():
    """Load checkpoint if exists."""
    global net, optimizer, start_epoch, LOSS
    checkpoint_dir = os.environ.get("PAI_OUTPUT_CHECKPOINTS")
    if not checkpoint_dir:
        return
    checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_NAME)
    if not os.path.exists(checkpoint_path):
        return
    data = torch.load(checkpoint_path)

    net.load_state_dict(data["model_state_dict"])
    optimizer.load_state_dict(data["optimizer_state_dict"])
    start_epoch = data["epoch"]


def save_checkpoint(epoch):
    global net, optimizer, start_epoch, LOSS
    checkpoint_dir = os.environ.get("PAI_OUTPUT_CHECKPOINTS")
    if not checkpoint_dir:
        return
    checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_NAME)
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    return args


def train():
    args = parse_args()
    load_checkpoint()
    batch_size = 4
    dataloader = DataLoader(RandomDataset(), batch_size=batch_size, shuffle=True)
    num_epochs = args.epochs
    print(num_epochs)
    for epoch in range(start_epoch, num_epochs):
        net.train()
        for i, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training progress
            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}"
                )

        # Save checkpoint
        save_checkpoint(epoch=epoch)
    # save the model
    torch.save(
        net.state_dict(),
        os.path.join(os.environ.get("PAI_OUTPUT_MODEL", "."), "model.pt"),
    )


if __name__ == "__main__":
    train()
