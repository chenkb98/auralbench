# text


## `normalize_unicode`

显式选择 Unicode 规范化形式，默认 NFKC 会合并兼容字符，例如全角拉丁字母。评测时应固定此设置，避免隐式改变文本标准。

## `casefold`

使用 Unicode casefold 处理大小写，适合跨语言文本比较。它可能改变字符数量，因此 CER 的规范化策略必须记录。

## `collapse_whitespace`

将 Unicode 空白连续段统一为一个普通空格，并移除两端空白。保留词边界，不将原本分开的单词连接起来。

## `strip_punctuation`

用空格替换 Unicode 标点类字符，保留词间隔。数学符号与 emoji 不属于标点类，默认保留。

## `normalize_text`

规范化顺序固定为 NFKC、可选大小写合并、可选标点替换和空白折叠。默认保留标点，避免不经说明地降低错误率。

## `word_tokens`

按空白分词，不伪装为中文分词器。无空格中文会作为单个词 token，中文转录通常更适合同时报告 CER。

## `character_tokens`

按规范化后的 Unicode 码点计算字符序列，默认去除空格。码点与用户感知的字形簇不同，组合 emoji 的 CER 应据此解读。

## `edit_alignment`

动态规划计算最小编辑距离与逐 token 对齐，确定性处理平局。空参考的非空预测全部计为插入；长转录需先分段，以限制内存与计算量。

## `word_error_rate`

WER 为编辑总数除以参考词数；空参考以分母 1 处理插入，行为与 JiWER 4 的约定一致。插入较多时 WER 可以大于 1。
