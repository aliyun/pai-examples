
import os
import re
import logging
import shutil


from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer


# 从环境变量中获取超参（由PAI的训练服务注入）
BATCH_SIZE = int(os.environ.get("PAI_HPS_BATCH_SIZE", 16))
LEARNING_RATE = float(os.environ.get("PAI_HPS_INITIAL_LEARNING_RATE", 1e-3))
NUM_EPOCHS = int(os.environ.get("PAI_HPS_EPOCHS", 1))
NUM_CLASSES = int(os.environ.get("PAI_HPS_NUM_CLASSES", 14))
MODEL_ID_OR_PATH = os.environ.get("PAI_INPUT_MODEL", "damo/cv_vit-base_image-classification_ImageNet-labels")

OUTPUT_MODEL_DIR = os.environ.get("PAI_OUTPUT_MODEL", "./model/")
WORK_DIR = os.environ.get("PAI_OUTPUT_CHECKPOINTS", "./checkpoints/")


# 将产出的模型保存到模型输出目录(OUTPUT_MODEL_DIR)
def save_model():
    best_ckpt_pattern = re.compile(
        pattern=r"^best_accuracy_top-1_epoch_\d+.pth$"
    )
    print("Saving best checkpoint as pytorch_model.pt")
    print("List work dir: ", os.listdir(WORK_DIR))

    f_name = next((f for f in os.listdir(WORK_DIR) if best_ckpt_pattern.match(f)), None)
    if f_name:
        # 使用最佳checkpoints作为输出模型
        print("Found best checkpoint: ", f_name)
        shutil.copyfile(
            src=os.path.join(WORK_DIR, f_name),
            dst=os.path.join(OUTPUT_MODEL_DIR, "pytorch_model.pt"),
        )
        os.remove(os.path.join(WORK_DIR, f_name))
    else:
        # 如果没有，则使用最后一个epoch的checkpoints作为输出模型
        print("Not found best checkpoint.")
        last_ckpt_file = "epoch_{}.pth".format(NUM_EPOCHS)
        if os.path.isfile(os.path.join(WORK_DIR, last_ckpt_file)):
            shutil.copyfile(
                src=os.path.join(WORK_DIR, last_ckpt_file),
                dst=os.path.join(OUTPUT_MODEL_DIR, "pytorch_model.pt"),
            )
    # 模型配置信息
    shutil.copyfile(
        src=os.path.join(WORK_DIR, "configuration.json"),
        dst=os.path.join(OUTPUT_MODEL_DIR, "configuration.json"),
    )


# 修改配置文件
def cfg_modify_fn(cfg):
    cfg.train.dataloader.batch_size_per_gpu = BATCH_SIZE # batch大小
    cfg.train.dataloader.workers_per_gpu = 8     # 每个gpu的worker数目
    cfg.train.max_epochs = 1                     # 最大训练epoch数
    cfg.model.mm_model.head.num_classes = NUM_CLASSES                       # 分类数
    cfg.model.mm_model.train_cfg.augments[0].num_classes = NUM_CLASSES      # 分类数
    cfg.model.mm_model.train_cfg.augments[1].num_classes = NUM_CLASSES      # 分类数
    cfg.train.optimizer.lr = LEARNING_RATE                # 学习率
    cfg.train.lr_config.warmup_iters = 1         # 预热次数

    # Note: OSS挂载到输出路径中，不支持软链接.
    cfg.train.checkpoint_config.create_symlink = False
    return cfg


ms_train_dataset = MsDataset.load(
            'flowers14', namespace='tany0699',
            subset_name='default', split='train') # 加载训练集

ms_val_dataset = MsDataset.load(
            'flowers14', namespace='tany0699',
            subset_name='default', split='validation') # 加载验证集


# 构建训练器
kwargs = dict(
    model=MODEL_ID_OR_PATH,                 # 模型id
    work_dir=WORK_DIR,
    train_dataset=ms_train_dataset, # 训练集  
    eval_dataset=ms_val_dataset,    # 验证集
    cfg_modify_fn=cfg_modify_fn     # 用于修改训练配置文件的回调函数
    )
trainer = build_trainer(name=Trainers.image_classification, default_args=kwargs)

# 进行训练
trainer.train()

# 进行评估
result = trainer.evaluate()
print('Evaluation Result:', result)

# 保存模型
save_model()
