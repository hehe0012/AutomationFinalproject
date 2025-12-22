**Parse Sentence 测试指南**

本项目包含基于“assemblies”模型的句子解析与模拟代码。主要通过 [parse_sentence.py](parse_sentence.py) 进行快速测试与演示，它会打印句子的依存边（SUBJ/OBJ 等）。

**环境依赖**
- **Python**: 3.x（建议 3.9+）
- **Packages**: numpy, scipy, matplotlib, pptree, jieba（中文分词）

安装示例：

```bash
pip install numpy scipy matplotlib pptree jieba
```

**快速开始**
- 英文示例：

```bash
python parse_sentence.py --language English --sentence "dogs chase cats" --rounds 20 --quiet
```

- 中文示例：

```bash
python parse_sentence.py --language Chinese --sentence "我把苹果放桌子上" --rounds 20
```

运行后将打印解析出的依存关系边。`--quiet` 可关闭详细的模拟日志，仅输出解析结果。

**内置测试（推荐）**
- 通过 `--dump_tests` 可运行内置的多条中文测试句并打印各自的依存边：

```bash
python parse_sentence.py --language Chinese --dump_tests
```

这会依次输出内置样例的句子与依存关系，便于快速验证解析逻辑与配置是否生效。

**常用参数**
- `--sentence`: 待解析的句子，默认中文示例。
- `--language`: 语言分支，`English | Chinese | Russian`。
- `--rounds`: 每个 token 的投影轮数（默认 20）。
- `--p`: 连接概率（默认 0.1）。
- `--lex_k`: LEX assemblies 的规模 k（默认 20）。
- `--quiet`: 关闭详细日志，只输出依存边。

中文分词相关（仅 `Chinese`）：
- `--jieba_mode`: `cut | search | full`，默认 `cut`。
- `--jieba_hmm`: `true | false`，默认 `true`。
- `--userdict`: 自定义词典路径（等价于环境变量 `JIEBA_USERDICT`）。
- `--jieba_suggest`: 逗号分隔的词频建议项（传递给 `jieba.suggest_freq`）。
- `--jieba_addwords`: 逗号分隔的 `word[:freq[:tag]]` 规格（传递给 `jieba.add_word`）。
- `--readout_minimal`: 启用最小化中文读出（仅 SUBJ/OBJ）。

**中文分词（jieba）提示**
- 中文解析在 [parser.py](parser.py) 内部使用 `jieba` 分词。
- 如需改善分词：
	- 提供自定义词典文件并通过 `--userdict` 或环境变量 `JIEBA_USERDICT` 指定；
	- 使用 `--jieba_addwords` 增加特定短语；
	- 使用 `--jieba_suggest` 调整可能被合并的动宾结构（如 `踢,球`）。

示例用户词典条目（每行一个，`word freq tag` 可选）：

```
桌子上 100000 nz
苹果 100000 n
```

**期望输出**
- 运行命令后，程序会打印解析得到的依存边列表（可能包含主语、宾语等标签）。若开启详细日志，将同时显示模拟与读出过程信息。

**常见问题**
- Windows 上可使用 `py` 或 `python`；若多版本并存，请确认 PATH 指向的解释器版本。
- 未安装依赖时会报错，请先执行上方安装命令。

更多细节与高级用法请直接阅读 [parse_sentence.py](parse_sentence.py) 与 [parser.py](parser.py)。
