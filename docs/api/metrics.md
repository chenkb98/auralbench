# metrics


## `si_sdr`

去均值后对参考信号投影，计算尺度不变 SDR，并将结果限制在有限 dB 范围。完全静音配对返回上限；仅一侧静音返回下限。

## `log_spectral_distance`

以固定谱幅下限计算 dB 频谱差的 RMS。它是简单信号诊断，不称为 FAD、PESQ 或 STOI，也不代替听感评测。
