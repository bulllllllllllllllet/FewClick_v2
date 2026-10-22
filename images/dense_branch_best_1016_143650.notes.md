# 当前保留稠密解码分支图

- 工具：内置 image_gen；生成提示词见同名 `.prompt.txt`。初稿经公式、尺寸和辅助输入连线校正后保存最终图；未修改模型。
- 源码：`benchmarks/v5/phase_6_mask_decoder/dense/dense_prompt_decoder.py` 的 `DensePromptDecoder(high_resolution=True, width=48)`。
- 对应开发候选：`matched96k_1014_221710/dense128_epoch_031.pth`，3k图块×32轮、冻结上游编码器；不是后续native/middle/full256实验版本。
- 输入省略batch维度，按C×H×W：deep256×64×64、shallow32×128×128、task256、semantic12×64×64、原J5概率512×512及正负提示热图。
- layer1原256通道以固定分组均值压缩成32通道，在分支外完成；分支内再可学习投影32→24。
- 深层256→48；task经LayerNorm/Linear256→96拆成gamma/beta，调制公式为 `F' = F * (1 + 0.1*tanh(gamma)) + beta`。
- 上下文拼接：48调制通道+12语义概率+1原概率+1不确定性+2点击热图=64通道；两层3×3卷积64→48→48。
- 上采样128×128后与24通道浅层拼接，再3×3卷积72→48、1×1卷积48→1；将残差插值至512×512。
- 输出为 `sigmoid(logit(clamp(p0,1e-6,1-1e-6)) + delta)`，不是概率直接相加。残差输出层零初始化保证初始复现原预测。
- 远处query没有本地点时，正负点图均为0，任务向量来自独立源support；不把源局部坐标复制到query。
- 128指内部浅层融合网格，不是最终mask尺寸；当前图不表示整张WSI端到端流程。
