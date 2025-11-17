# text


## `normalize_unicode`

显式选择 Unicode 规范化形式，默认 NFKC 会合并兼容字符，例如全角拉丁字母。评测时应固定此设置，避免隐式改变文本标准。

## `casefold`

使用 Unicode casefold 处理大小写，适合跨语言文本比较。它可能改变字符数量，因此 CER 的规范化策略必须记录。

## `collapse_whitespace`

将 Unicode 空白连续段统一为一个普通空格，并移除两端空白。保留词边界，不将原本分开的单词连接起来。
