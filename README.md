Quick README

This repository contains code for simulating operations in the assembly model of brain computation.

Usage

```bash
python parse_sentence.py --language English --sentence "dogs chase cats" --rounds 20 --quiet
python parse_sentence.py --language Chinese --sentence "我把苹果放桌子上" --rounds 20
```

Chinese segmentation (jieba)

- Chinese parsing uses `jieba` for tokenization inside `parser.parse()`.
- To improve segmentation, provide a custom user dictionary:
	- Set env `JIEBA_USERDICT` to the dictionary path, or
	- Place `jieba_userdict.txt` in the project root.
- Jieba userdict format: one entry per line, optionally `word freq tag`.

Example userdict lines:

```
桌子上 100000 nz
苹果 100000 n
```
