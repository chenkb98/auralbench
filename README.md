# auralbench

![CI](https://github.com/chenkb98/auralbench/actions/workflows/ci.yml/badge.svg)

Auditable evaluation for generated audio and speech models, with manifests, metrics, statistics, and reports.

Chen Kaibin · 华南理工大学在读。项目关注音频生成和语音大模型研究中的可复现工具。

本仓库的 2025–2026 提交时间线为重建的演示数据，不代表实际开发日期。

## 快速开始

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

激活虚拟环境后运行：

```sh
auralbench demo --output outputs/demo
```

## 主要功能

清单定义、实际输入字节和逐样本结果分别可核验。续跑检查定义摘要和输入内容，避免悄悄复用过期结果。

| 内容 | 入口 |
| --- | --- |
| 使用与参数 | [使用文档](docs/usage.md) |
| 完整模块 API | [API 索引](docs/api-reference.md) |
| 数据流与边界 | [架构](docs/architecture.md) |
| 数值定义与参考 | [设计说明](docs/design-notes.md) |
| 可运行合成示例 | [examples](examples/README.md) |

## 验证

```sh
make build
make test
make format-check
make lint typecheck
```

CI 覆盖 Python 3.11 和 3.12。开发工具版本由 requirements-dev.txt 固定。
测试包含空输入、无效值、往返转换、完整字段模式和公共接口契约。

## 范围

默认流程不需要 GPU、付费服务、模型权重或外部数据。音频示例与训练示例是合成基线，
不能用来宣称真实语音准确率或感知质量。PCM16 会量化并饱和削波；线性重采样没有抗混叠滤波。

MIT 许可。参与方式见 [CONTRIBUTING](CONTRIBUTING.md)。
