# 设计约定

清单定义、实际输入字节和逐样本结果分别可核验。续跑检查定义摘要和输入内容，避免悄悄复用过期结果。

## 可复现性与边界

随机过程使用局部 NumPy 生成器与显式种子；不改变全局随机状态。
合成样例验证实现行为，不代表真实语音模型的识别率、听感或训练成果。
不同指标的静音、空输入和量化策略见对应 `docs/api/` 页面。

## 上游参考

- [NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html)：数值变换接口。
- [NumPy quantile](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)：显式 linear 分位数定义。
- [JiWER](https://jitsi.github.io/jiwer/)：转录错误率与空参考的约定。
- [AudioCraft AudioGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/AUDIOGEN.md)：真实音频生成系统的背景。
- [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)：音频语言模型接口背景。

这些项目是参考资料，各自的模型、数据、许可和性能属于相应上游。
本项目不宣称复现了它们的训练规模或基准结果。
