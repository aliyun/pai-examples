# 概览

- [使用PAI预置算法微调预训练模型](./pretrained_model/pretrained_model.ipynb)

PAI 提供了一系列的公共可用的开源预训练模型，包含大语言模型，AIGC 等各个领域，并为模型配置了相应的微调训练算法。当前示例中，将演示用户如何获取这些模型，使用这些模型完成模型的微调训练。

- [使用TensorBoard可视化训练过程](./tensorboard/tensorboard.ipynb)

PAI 提供了托管的 TensorBoard 服务，支持用户在 PAI 上创建一个TensorBoard实例，可视化训练过程，帮助用户更好的理解模型训练过程。在当前 Notebook 中，我们将介绍如何使用在训练作业中写出 TensorBoard，使用 TensorBoard 可视化训练过程。

- [提交PyTorch分布式作业](./pytorch_ddp/pytorch_ddp.ipynb)

PAI 支持用户提交分布式 PyTorch 训练作业，PAI训练服务会通过环境变量的方式注入作业的分布式环境信息，当前示例将介绍如何使用PAI Python SDK，以[PyTorch DDP(DistributedDataParallel)](https://pytorch.org/docs/stable/notes/ddp.html)模式提交分布式PyTorch训练作业。

- [在训练作业中使用checkpoint](./checkpoint/checkpoint.ipynb)

训练作业中通常需要通过checkpoint，保存模型的训练状态，以便在训练中断后，可以从中断的地方继续训练。在当前示例中，我们将介绍如何在训练作业中加载或是保存 checkpoint ，从而支持训练作业中断恢复。
