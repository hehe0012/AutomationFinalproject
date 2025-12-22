# 中文依存解析迁移（需改动 brain.py / parser.py）

目标：在不破坏现有英文/俄文解析的前提下，支持中文句子，分词使用 jieba，读出依存关系。

## 总体流程
1. 句子经 `jieba.cut` 分词后，用空格拼接传给 `parse`（保持现有按空格切分的接口）。
2. 在 `parser.py` 增加 `ChineseParserBrain`、中文词表与规则、中文 `READOUT_RULES`，在 `parse()` 分支调用。
3. 在 `brain.py` 确保新增区域、可塑性配置、显式/非显式区初始化兼容新增中文区域。

## parser.py 需要的主要改动
- **依赖**：在文件顶部引入 `jieba`（仅 parser 层需要）。
- **新增常量/区域**：
  - 新区域：`BA`（把），`BEI`（被），`ASPECT`（了/过/着），`CLASSIFIER`，`NUM`，`DE`（的），如需从句可复用/新增 `REL_CLAUSE`。
  - 更新 `AREAS`、`EXPLICIT_AREAS`（通常只 `LEX`）、`RECURRENT_AREAS`（加入新语法区）。
- **中文词表**：
  - 定义 `CHINESE_LEXEME_DICT`，为每个词给出 `index`、`PRE_RULES`、`POST_RULES`。
  - 规则模板示例：
    - 名词：解禁 `LEX→SUBJ/OBJ/PREP_P`，数量/量词路径 `CLASSIFIER/NUM→NOUN`。
    - 动词：解禁 `VERB→SUBJ/OBJ/ADVERB/PREP_P`，预留 `VERB→ASPECT`。
    - 把：开启 `LEX→BA`、`BA→OBJ`，抑制与主动宾语冲突的纤维。
    - 被：开启 `LEX→BEI`、`BEI→SUBJ`（逻辑施事）和 `VERB` 链接，抑制主动路径。
    - 体标记（了/过/着）：激活 `ASPECT`，POST 中可抑制继续堆叠。
    - “的”：通过 `DE` 连接修饰语与中心词，可调用与 `DEP_CLAUSE` 类似的处理。
- **中文读出规则**：`CHINESE_READOUT_RULES` 将区域映射到读出链路；同时定义依存标签映射（如 `SUBJ→nsubj`, `OBJ→obj`, `BA→ba`, `BEI→bei`, `ASPECT→aspect`, `DE→attr/rel`, `PREP_P→obl`）。
- **新增 ChineseParserBrain**（仿 English/Russian）：
  - 在 `__init__` 注册中文区域、调用 `update_plasticities` 增加可塑性（LEX 双向强，语法区内弱循环，跨区按需）。
  - 调整 `LEX_SIZE/LEX_k` 以覆盖中文词表；其他区域的 `n/k` 依据稀疏度设定。
- **parse 分支**：
  - 在 `parse()` 中增加 `language == "Chinese"` 分支，构造 `ChineseParserBrain`，选择中文词表、区域、读出规则。
  - 在进入主循环前：`tokens = list(jieba.cut(sentence))`; `sentence = " ".join(tokens)`，保持后续逻辑不变。
- **读出逻辑**：复用现有 `FIBER_READOUT`，确保 `getActivatedFibers` 允许中文区域被保留；读出时将 `to_area` 映射为依存标签输出。

## brain.py 需要的主要改动
- **区域初始化**：对新增中文区域（BA/BEI/ASPECT/CLASSIFIER/NUM/DE/可选 REL_CLAUSE）支持 `add_area`/`add_explicit_area` 的默认路径，不需特殊逻辑但需在 `update_plasticities` 调用前定义。
- **可塑性配置**：
  - 扩展调用 `update_plasticities` 时的 `area_update_map`，为新区域添加：
    - LEX 与各语法区的双向高 beta（便于快速形成装配体）。
    - 语法区内自循环/互连较低 beta。
    - 特殊路径（如 BA→OBJ、BEI→SUBJ、ASPECT→VERB）可单独指定更强/更弱 beta。
- **字典兼容性**：`activateWord/getWord` 已基于 `lexeme_dict` 与 `k` 工作，可直接用中文词串，无需改编码。确保 `lexeme_dict` 全覆盖 jieba 分词后的 token，或在上层处理 OOV。

## 配置与参数建议
- `p`、`beta`：可沿用英文默认起点，若中文句更长/功能词多，可适当增大 `project_rounds`（如 25–40）。
- `n/k`：LEX 显式区 `n = vocab_size * k`；语法区 `n` 维持 1e4–1e5，`k` 50–200 视稀疏度调节。

## 验证用例（分词后）
- 基础 SVO：“我 爱 你”。
- 把字：“我 把 苹果 放 桌子 上”。
- 被动：“苹果 被 我 吃 了”。
- 体标记：“他 去 过 北京”。
- “的” 修饰/定语从句：“我 看到 的 人”，“我 说 的 话”。
- 连动/兼语：“请 你 吃饭”。

## 最终交付
- 不改现有代码；本文件为修改蓝图。后续实现时按上述步骤修改 `parser.py` 与 `brain.py`，并增加中文词表/规则与读出标签映射。
