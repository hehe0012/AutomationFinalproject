"""
Run the assembly parser on a sentence and print dependency edges.

Usage examples:
    # Chinese (jieba inside parser)
    python parse_sentence.py --language Chinese --sentence "我把苹果放桌子上" --rounds 20

    # English
    python parse_sentence.py --language English --sentence "dogs chase cats" --rounds 20 --quiet

Dependencies: numpy, scipy, matplotlib, pptree, jieba (for Chinese segmentation).
Install extras with: pip install numpy scipy matplotlib pptree jieba
"""
import argparse
import jieba
import jieba.posseg as pseg
import unittest
import os
from parser import parse, ReadoutMethod, chinese_segment_with_filter, parse_dependencies

# Encourage stable token boundaries for the custom phrases used in the examples.
CUSTOM_WORDS = [
    "红温",
    "无可奈何地",
    "并非",
    "人类",
    "愚蠢的",
    "聪明的",
    "硬邦邦的",
    "愤怒地",
    "一颗",
    "善良",
    "温柔",
    "大度",
    "愚蠢",
]

"""
TEST_CASES entries:
(
    name: str,
    sentence: str,
    expected_tokens: List[str],
    expected_pos: List[str]  # jieba.posseg flags aligned with expected_tokens
)
"""
TEST_CASES = [
        ("pattern1_minimal", "我红温了"),
        ("pattern1_with_adverb", "我无可奈何地红温了"),
        ("pattern2_copula", "我并非人类"),
        ("pattern2_transitive", "我踢球"),
        ("pattern3_adjective_predicate", "你真善良"),
        ("pattern4_adj_subject_copula", "愚蠢的我并非人类"),
        ("pattern4_adj_subject_transitive", "愚蠢的我踢球"),
        ("pattern5_adj_object_copula", "我并非愚蠢的人类"),
        ("pattern5_adj_object_transitive", "我踢硬邦邦的球"),
        ("pattern6_both_adj_copula", "聪明的我并非愚蠢的人类"),
        ("pattern6_both_adj_transitive", "愚蠢的我踢硬邦邦的球"),
        ("pattern7_full_stack", "我愤怒地踢一颗硬邦邦的球"),
]


@unittest.skipUnless(jieba, "jieba is required for Chinese example tokenization tests")
class ChineseExampleTokenizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure custom words are recognized as single tokens to match the grammar patterns.
        for word in CUSTOM_WORDS:
            jieba.add_word(word)

        # Force split for verbs + objects that jieba may merge (e.g., "踢球").
        jieba.suggest_freq(("踢", "球"), True)

    def test_examples_tokenize_as_expected(self):
        for name, sentence, expected, expected_pos in TEST_CASES:
            with self.subTest(case=name):
                tokens = list(jieba.cut(sentence, HMM=True))
                self.assertEqual(tokens, expected)
                pos_tags = [w.flag for w in pseg.lcut(sentence, HMM=True)]
                self.assertEqual(pos_tags, expected_pos)


def main():
    parser = argparse.ArgumentParser(description="Parse a sentence with the assembly parser (supports English/Chinese/Russian).")
    parser.add_argument("--sentence", type=str, default="我把苹果放桌子上", help="Sentence to parse.")
    parser.add_argument(
        "--language",
        type=str,
        default="Chinese",
        choices=["English", "Chinese", "Russian"],
        help="Language branch to use.",
    )
    parser.add_argument("--rounds", type=int, default=20, help="Projection rounds per token.")
    parser.add_argument("--p", type=float, default=0.1, help="Connection probability p.")
    parser.add_argument("--lex_k", type=int, default=20, help="Size k for LEX assemblies.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose simulation logging.")

    # Jieba tuning options (Chinese only)
    parser.add_argument("--jieba_mode", type=str, default=os.environ.get("JIEBA_MODE", "cut"), choices=["cut", "search", "full"], help="Jieba tokenization mode for Chinese.")
    parser.add_argument("--jieba_hmm", type=str, default=os.environ.get("JIEBA_HMM", "true"), choices=["true", "false"], help="Enable HMM in Jieba (true/false).")
    parser.add_argument("--userdict", type=str, default=os.environ.get("JIEBA_USERDICT", None), help="Path to a Jieba user dictionary.")
    parser.add_argument("--jieba_suggest", type=str, default=os.environ.get("JIEBA_SUGGEST", ""), help="Comma-separated terms to pass to jieba.suggest_freq.")
    parser.add_argument("--jieba_addwords", type=str, default=os.environ.get("JIEBA_ADDWORDS", ""), help="Comma-separated specs 'word[:freq[:tag]]' to pass to jieba.add_word.")
    parser.add_argument("--readout_minimal", action="store_true", help="Use minimal Chinese readout (only SUBJ/OBJ edges).")
    parser.add_argument("--dump_tests", action="store_true", help="Print tokenization and POS for all TEST_CASES and exit.")
    args = parser.parse_args()

    verbose = not args.quiet

    # Propagate Jieba settings via environment for parser.parse
    if args.userdict:
        os.environ["JIEBA_USERDICT"] = args.userdict
    if args.jieba_mode:
        os.environ["JIEBA_MODE"] = args.jieba_mode
    if args.jieba_hmm:
        os.environ["JIEBA_HMM"] = args.jieba_hmm
    if args.jieba_suggest is not None:
        os.environ["JIEBA_SUGGEST"] = args.jieba_suggest
    if args.jieba_addwords is not None:
        os.environ["JIEBA_ADDWORDS"] = args.jieba_addwords
    if args.readout_minimal:
        os.environ["READOUT_MINIMAL"] = "true"

    if args.dump_tests:
        for word in CUSTOM_WORDS:
            jieba.add_word(word)
        jieba.suggest_freq(("踢", "球"), True)
        for name, sentence in TEST_CASES:

            deps = parse_dependencies(
                sentence=sentence,
                language="Chinese",
                p=0.1,
                LEX_k=20,
                project_rounds=10,
                verbose=False,
                debug=False,
                readout_method=ReadoutMethod.FIBER_READOUT,
                minimal=False,
            )
            deps.sort(key=lambda d: d[2])
            print(f"Case {name}: {sentence}")
            print("Got dependencies: ")
            print(deps)
            print("")
        return
    # English: use parse() to match original behavior; others use parse_dependencies().
    if args.language.lower() == "english":
        parse(
            sentence=args.sentence,
            language=args.language,
            p=args.p,
            LEX_k=args.lex_k,
            project_rounds=args.rounds,
            verbose=verbose,
            debug=False,
            readout_method=ReadoutMethod.FIBER_READOUT,
        )
    else:
        ans = parse_dependencies(
            sentence=args.sentence,
            language=args.language,
            p=args.p,
            LEX_k=args.lex_k,
            project_rounds=args.rounds,
            verbose=verbose,
            debug=False,
            readout_method=ReadoutMethod.FIBER_READOUT,
            minimal=False,
        )
        print("Dependencies (head, dependent, relation):")
        for dep in ans:
            print(dep)


if __name__ == "__main__":
    main()
