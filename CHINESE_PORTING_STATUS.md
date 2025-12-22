# 中文依存解析实现摘要

本次已在现有代码基础上接入中文解析（jieba 分词）并最小化改动范围，核心调整如下：

- **parser.py**
  - 引入 `jieba` 分词；在 `parse(language="Chinese")` 分支对输入句子先分词再按空格重组。
  - 增加中文专用区域常量：BA/BEI/ASPECT/CLASSIFIER/NUM/DE，并定义 `CHINESE_AREAS`/`CHINESE_RECURRENT_AREAS`/`CHINESE_EXPLICIT_AREAS`。
  - 新增中文词表 `CHINESE_LEXEME_DICT`（含基础名词、及物/不及物动词、把/被、体标记、在、的等），及对应规则模板（PRE/POST RULES）。
  - 定义 `CHINESE_READOUT_RULES`，在纤维读出时保留中文特有结构（把/被/体标记等）。
  - 新增 `ChineseParserBrain`，注册中文区域、设定可塑性（LEX 双向强、语法区内弱循环、跨区互连），并复用现有解析/读出逻辑。

- **brain.py**
  - 无需修改：现有 `add_area`/`update_plasticities` 能支持新增区域；解析器类内已完成中文区域与可塑性配置。

- **使用方式**
  - 依赖：`pip install jieba`（已完成安装记录）。
  - 调用：`parse(sentence="我把苹果放桌子上", language="Chinese", ...)`；若词不在中文词表需补充 `CHINESE_LEXEME_DICT` 条目。

如需扩展词表或依存标签映射，可在 `CHINESE_LEXEME_DICT` 和 `CHINESE_READOUT_RULES` 中按需添加。
