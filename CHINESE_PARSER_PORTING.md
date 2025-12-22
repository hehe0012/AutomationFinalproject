# 中文解析器迁移清单（使用 jieba）

## 范围
- 目标：让装配体解析器支持中文的依存式读出。
- 约束：分词使用 jieba。

## 1) 输入预处理
- 依赖：在环境中安装并导入 `jieba`（`pip install jieba`）。
- 在调用 `parse()` 前先分词：`tokens = list(jieba.cut(sentence))`，再用空格连接以符合现有接口。
- OOV 策略：未知词需要映射到 `LEXEME_DICT` 中的占位词，或被过滤/回退处理。

## 2) 词表与装配体
- 构建中文 `LEXEME_DICT`：每个词分配唯一 `index`，并定义对应的 `PRE_RULES`/`POST_RULES`。
- 尺寸调优：若词表较大，需要增大 `LEX_SIZE`/`LEX_k` 以降低装配体碰撞。
- 保持 LEX 为显式区，确保 `n = vocab_size * k`。

## 3) 区域（脑区）设计
- 保留核心：`LEX`, `SUBJ`, `OBJ`, `VERB`, `ADJ`, `ADVERB`, `PREP`, `PREP_P`。
- 可能新增的中文特有区域：`BA`（把），`BEI`（被），`ASPECT`（了/过/着），`CLASSIFIER`，`NUM`，`DE`（的），如需可加 `REL_CLAUSE`（关系/从句）。
- 确定 `EXPLICIT_AREAS`（通常仅 `LEX`）与 `RECURRENT_AREAS`（句法/功能区包含新增区域）。

## 4) 规则模板（PRE_RULES / POST_RULES）
- 名词：解禁 `SUBJ/OBJ/PREP_P` 方向；如用到 `DET/ADJ` 亦需解禁；用完再关闭。
- 动词：打开 `VERB→SUBJ/OBJ/ADVERB`、`VERB→PREP_P`，为体标记预留 `VERB→ASPECT` 钩子。
- 体标记（了/过/着）：激活 `ASPECT`，可在 POST 中抑制继续堆叠。
- 把：打开 `LEX→BA`、`BA→OBJ` 通路，调整抑制以让后续名词短语绑定为把字受事。
- 被：打开被动通路 `LEX→BEI`，`BEI→SUBJ`（逻辑施事）并连接 `VERB`，抑制与主动路径冲突的纤维。
- 的：经 `DE` 将修饰成分连到中心词；支持递归/属性链接（可复用 `DEP_CLAUSE`）。
- 数量/量词：打开 `NUM/CLASSIFIER→NOUN` 通路，抑制重复附着。
- POST_RULES：关闭上述纤维/区域，恢复抑制，避免串扰。

## 5) 可塑性与连接
- 调整 `default_beta`、`LEX_beta`、`recurrent_beta`、`interarea_beta` 以适配新增区域和更长句。
- 在 `update_plasticities` 中为新增区域添加可塑性：LEX 双向偏强；同区内弱循环；跨区按角色调节。

## 6) 读出（依存）
- 更新 `READOUT_RULES`，包含中文特有区域，确保纤维读出保留 BA/BEI/ASPECT/DE 等边。
- 将区域映射到依存标签：如 `SUBJ→nsubj`，`OBJ→obj`，`BA→ba`，`BEI→bei`，`ASPECT→aspect`，`DE→attr/rel`，`PREP_P→obl`。

## 7) 递归/从句
- 处理“的/说/因为/如果”等触发的从句或关系从句：复用 `DEP_CLAUSE` 或增设 `REL_CLAUSE`。触发时冻结外层、解冻从句区，完成若干轮投射后再合并。

## 8) 参数与收敛
- 为新增区域调整 `n`/`k`：句法区可维持 1e4–1e5 神经元，`k` 约 50–200 视稀疏度而定。
- 中文句更长且功能词多，可能需要适当增大 `project_rounds`。

## 9) 测试清单
- 基础句型：SVO、把字句、被动句、连动/兼语、体标记（了/过/着）、“的”修饰/定语从句、介宾前置/后置、数量短语。
- 对每类检查：a) 读出边是否符合预期角色；b) `getWord` 能否恢复分词 token；c) 支持集是否收敛且无失控增长。

## 10) 集成步骤（未来编码时）
- 将 `jieba` 加入依赖；在 `parse` 前完成分词并以空格拼回传入。
- 实现 `ChineseParserBrain`（仿 English/Russian），注册新增区域和可塑性。
- 提供中文 `LEXEME_DICT` 与规则模板，在 `parse(..., language="Chinese")` 分支调用。
- 扩展读出映射，输出含中文特有标签的依存边。

## 11) 操作注意
- 代码标识尽量保持 ASCII，词表字符串可用中文。
- 先用小词表、短句调参（beta、轮数），稳定后再扩大。# Chinese Parser Porting Checklist (using jieba)

## Scope
- Goal: enable the assembly parser to handle Chinese dependency-like readout.
- Constraint: plan only, no code changes yet. Use jieba for segmentation.

## 1) Input prep
- Add `jieba` dependency and import; `pip install jieba` in env.
- Pre-segment sentence before calling `parse`: `tokens = list(jieba.cut(sentence))`; join with spaces for existing parser interface.
- Decide OOV handling: unknown tokens must map to a placeholder entry in `LEXEME_DICT` or be filtered.

## 2) Lexicon and assemblies
- Build Chinese `LEXEME_DICT`: each word gets unique `index` and tailored `PRE_RULES`/`POST_RULES`.
- Size tuning: increase `LEX_SIZE`/`LEX_k` if vocab is large to reduce assembly collisions.
- Keep LEX explicit; ensure `n = vocab_size * k`.

## 3) Areas (brain regions)
- Core to keep: `LEX`, `SUBJ`, `OBJ`, `VERB`, `ADJ`, `ADVERB`, `PREP`, `PREP_P`.
- Likely add Chinese-specific areas: `BA` (把), `BEI` (被), `ASPECT` (了/过/着), `CLASSIFIER`, `NUM`, `DE` (的), maybe `REL_CLAUSE` if needed.
- Decide `EXPLICIT_AREAS` (usually `LEX`) and `RECURRENT_AREAS` (syntax/functional areas including new ones).

## 4) Rule templates (PRE_RULES / POST_RULES)
- Noun: disinhibit links to `SUBJ/OBJ/PREP_P`; manage `DET/ADJ` if used; close after use.
- Verb: open `VERB→SUBJ/OBJ/ADVERB`, `VERB→PREP_P`; add `VERB→ASPECT` hooks.
- Aspect particles (了/过/着): enable `ASPECT`; optionally inhibit further aspect stacking.
- 把: open `LEX→BA`, `BA→OBJ` gating; adjust inhibitions so following NP binds as BA-object.
- 被: open passive path `LEX→BEI`, `BEI→SUBJ` (logical agent) and `VERB` links; inhibit conflicting active links.
- 的: connect modifier to head via `DE`; enable recursive/attribute links (could reuse `DEP_CLAUSE`).
- Classifier/number: open `NUM/CLASSIFIER→NOUN` path; inhibit duplicate attachments.
- Close with POST_RULES to restore inhibition and prevent bleed-over.

## 5) Plasticity and connectivity
- Revisit betas: `default_beta`, `LEX_beta`, `recurrent_beta`, `interarea_beta` to handle added areas and higher token counts.
- Add plasticity entries for new areas in `update_plasticities` (LEX strong both ways; intra-area recurrent weaker; cross-area tuned per role).

## 6) Readout (dependencies)
- Update `READOUT_RULES` to include new areas; ensure fiber readout keeps Chinese-specific edges (BA/BEI/ASPECT/DE/etc.).
- Map areas to dependency labels: e.g., `SUBJ→nsubj`, `OBJ→obj`, `BA→ba`, `BEI→bei`, `ASPECT→aspect`, `DE→attr/rel`, `PREP_P→obl`.

## 7) Recursion / clauses
- For relative/embedded clauses, reuse `DEP_CLAUSE` or add `REL_CLAUSE`: on trigger (的/说/因为...), freeze outer areas, unfix clause area, run inner projection rounds, then merge.

## 8) Parameters and convergence
- Adjust `n`/`k` for new areas: syntax areas keep ~1e4–1e5 neurons, `k` ~50–200 depending on sparsity.
- Possibly increase `project_rounds` for longer Chinese sentences and functional particles.

## 9) Testing checklist
- Minimal cases: SVO, 把字句, 被动句, 连动/兼语, 体标记 (了/过/着), 的 修饰/定语从句, 介词短语前置/后置, 数量短语。
- For each case verify: (a) readout edges align with expected roles, (b) `getWord` recovers tokens, (c) assemblies converge without runaway growth.

## 10) Integration steps (when coding)
- Add `jieba` to requirements; pre-segment before `parse` and feed space-joined tokens.
- Implement `ChineseParserBrain` similar to English/Russian; register new areas and plasticities.
- Provide Chinese `LEXEME_DICT` and rule templates; wire `parse(..., language="Chinese")` branch.
- Extend readout mapping to emit dependency labels for Chinese-specific areas.

## 11) Operational notes
- Keep ASCII; avoid non-ASCII identifiers in code if possible, but allow Chinese tokens in lexicon strings.
- Start with a small vocab and sentences to tune betas and rounds, then scale.
