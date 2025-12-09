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
