# 【算法挑战赛】基于无人机的人体行为识别——国赛仓库

- **百度网盘链接：** https://pan.baidu.com/s/1EVJnTb9U2xShelBH7BboxQ?pwd=inuc

  ------

- **简介：**

  - 本次比赛方案以top仓库与TE-GCN仓库为代码基础构建方案，我们选择使用了20个模型、4种模态及4种数据增强方法进行模型的训练与推理，而后使用三阶段多集成方案对不同模型的预测结果进行集成。我们主要的创新工作可总结为：
    - （1）以使用多种不同数据增强的方式模拟使用大量的训练数据；
    - （2）探究模型内外集成组合的方案效果，确定了模型内部集成与多模型集成的集成方案；
    - （3）创新了自适应模型集成权重选择算法，更加合理与高效地进行模型集成；
    - （4）创新清晰集与模糊集分类设想，提高了模型的预测准确率。

<div align="center">
  <img src="/pic/架构.png" alt="架构图">
</div>

  ------

- **仓库结构：**

  - model：比赛使用的代码仓库
    - ICMEW2024-Track10
      - Model_infrence目录下为Mix_Former和Mix_GCN的代码，包括相关组合模型的训练日志等
    - TE-GCN
      - TE-GCN模型的代码，包括相关组合模型的训练日志等
  - emsemble：比赛使用模型集成代码仓库
    - stage0：模型内部初步集成工具代码
    - stage1：所有模型全部集成工具代码
    - stage2：集成后处理代码
  - 参赛文档.pdf：说明文档

    ------

- **百度网盘内容结构说明：**
  - data：训练数据和测试数据的npz文件，其中训练数据的测试集为验证集，为真实标签，testB数据为人工生成的相同标签

  - pkl-val：61个组合模型在验证集上的推理结果文件

  - pkl-test：61个组合模型在测试集上的推理结果文件

  - train-pt：61个组合模型的最佳权重

    ------

- **运行方式：** 见参赛文档【运行说明】部分

