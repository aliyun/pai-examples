import argparse
import logging
import os

import pandas as pd
from xgboost import XGBClassifier

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

TRAINING_BASE_DIR = "/ml/"
TRAINING_OUTPUT_MODEL_DIR = os.path.join(TRAINING_BASE_DIR, "output/model/")


def load_dataset(channel_name):
    path = os.path.join(TRAINING_BASE_DIR, "input/data", channel_name)
    if not os.path.exists(path):
        return None, None

    # use first file in the channel dir.
    file_name = next(
        iter([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]),
        None,
    )
    if not file_name:
        logging.warning(f"Not found input file in channel path: {path}")
        return None, None

    file_path = os.path.join(path, file_name)
    df = pd.read_csv(
        filepath_or_buffer=file_path,
        sep=",",
    )

    train_y = df["target"]
    train_x = df.drop(["target"], axis=1)
    return train_x, train_y


def main():
    parser = argparse.ArgumentParser(description="XGBoost train arguments")
    # 用户指定的任务参数
    parser.add_argument(
        "--n_estimators", type=int, default=500, help="The number of base model."
    )
    parser.add_argument(
        "--objective", type=str, help="Objective function used by XGBoost"
    )

    parser.add_argument(
        "--max_depth", type=int, default=3, help="The maximum depth of the tree."
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.2,
        help="Step size shrinkage used in update to prevents overfitting.",
    )
    parser.add_argument(
        "--eval_metric",
        type=str,
        default=None,
        help="Evaluation metrics for validation data",
    )

    args, _ = parser.parse_known_args()

    # 加载数据集
    train_x, train_y = load_dataset("train")
    print("Train dataset: train_shape={}".format(train_x.shape))
    test_x, test_y = load_dataset("test")
    if test_x is None or test_y is None:
        print("Test dataset not found")
        eval_set = [(train_x, train_y)]
    else:
        eval_set = [(train_x, train_y), (test_x, test_y)]

    clf = XGBClassifier(
        max_depth=args.max_depth,
        eta=args.eta,
        n_estimators=args.n_estimators,
        objective=args.objective,
    )
    clf.fit(train_x, train_y, eval_set=eval_set, eval_metric=args.eval_metric)

    model_path = os.environ.get("PAI_OUTPUT_MODEL")
    os.makedirs(model_path, exist_ok=True)
    clf.save_model(os.path.join(model_path, "model.json"))
    print(f"Save model succeed: model_path={model_path}/model.json")


if __name__ == "__main__":
    main()
