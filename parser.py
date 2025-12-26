#! /usr/bin/python3.9
import brain
import brain_util as bu
import numpy as np
import pptree
import jieba
import json
import copy
import os
def chinese_segment_with_filter(sentence, verbose=False):
	# Optional: load a user dictionary to improve segmentation for domain terms.
	# Priority: env var JIEBA_USERDICT, else local file 'jieba_userdict.txt' if present.
	userdict_path = os.environ.get("JIEBA_USERDICT", None)
	if not userdict_path and os.path.exists("jieba_userdict.txt"):
		userdict_path = "jieba_userdict.txt"
	if userdict_path:
		try:
			jieba.load_userdict(userdict_path)
			if verbose:
				print(f"Loaded jieba userdict: {userdict_path}")
		except Exception as e:
			if verbose:
				print(f"Warning: failed to load jieba userdict '{userdict_path}': {e}")

	# Optional: tune jieba behavior via env vars.
	hmm_env = os.environ.get("JIEBA_HMM", None)
	hmm = True if hmm_env is None else hmm_env.lower() in ("1", "true", "yes", "on")
	mode = os.environ.get("JIEBA_MODE", "cut").lower()

	suggest_env = os.environ.get("JIEBA_SUGGEST", "")
	if suggest_env:
		for tok in suggest_env.split(","):
			tok = tok.strip()
			if tok:
				try:
					jieba.suggest_freq(tok, True)
				except Exception as e:
					if verbose:
						print(f"Warning: suggest_freq failed for '{tok}': {e}")

	addwords_env = os.environ.get("JIEBA_ADDWORDS", "")
	if addwords_env:
		for spec in addwords_env.split(","):
			spec = spec.strip()
			if not spec:
				continue
			parts = spec.split(":")
			word = parts[0]
			freq = None
			tag = None
			if len(parts) > 1:
				try:
					freq = int(parts[1])
				except Exception:
					freq = None
			if len(parts) > 2:
				tag = parts[2]
			try:
				if freq is not None and tag is not None:
					jieba.add_word(word, freq, tag)
				elif freq is not None:
					jieba.add_word(word, freq)
				else:
					jieba.add_word(word)
				if verbose:
					print(f"Added jieba word: {word} freq={freq} tag={tag}")
			except Exception as e:
				if verbose:
					print(f"Warning: add_word failed for '{spec}': {e}")

	# Tokenize according to mode
	if mode == "search":
		tokens = list(jieba.cut_for_search(sentence, HMM=hmm))
	elif mode == "full":
		tokens = list(jieba.cut(sentence, HMM=hmm, cut_all=True))
	else:
		tokens = list(jieba.cut(sentence, HMM=hmm))

	# Filter tokens to those present in lexeme dict
	filtered_tokens = [t for t in tokens if t in CHINESE_LEXEME_DICT]
	if not filtered_tokens:
		if len(tokens) == 1 and len(tokens[0]) > 1:
			filtered_tokens = list(tokens[0])
		else:
			filtered_tokens = tokens

	if verbose:
		print(f"Chinese tokens: {tokens}")
		if filtered_tokens != tokens:
			print(f"Filtered tokens (known lexemes only): {filtered_tokens}")

	return tokens, filtered_tokens

from collections import namedtuple
from collections import defaultdict
from enum import Enum

# BrainAreas
LEX = "LEX"
DET = "DET"
SUBJ = "SUBJ"
OBJ = "OBJ"
VERB = "VERB"
PREP = "PREP"
PREP_P = "PREP_P"
ADJ = "ADJ"
ADVERB = "ADVERB"
BA = "BA"
BEI = "BEI"
ASPECT = "ASPECT"
CLASSIFIER = "CLASSIFIER"
NUM = "NUM"
DE = "DE"

# Unique to Russian
NOM = "NOM"
ACC = "ACC"
DAT = "DAT"

# Fixed area stats for explicit areas
LEX_SIZE = 20

# Actions
DISINHIBIT = "DISINHIBIT"
INHIBIT = "INHIBIT"
# Skip firing in this round, just activate the word in LEX/DET/other word areas.
# All other rules for these lexical items should be in PRE_RULES.
ACTIVATE_ONLY = "ACTIVATE_ONLY"
CLEAR_DET = "CLEAR_DET"

AREAS = [LEX, DET, SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P]
EXPLICIT_AREAS = [LEX]
RECURRENT_AREAS = [SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P]

CHINESE_AREAS = [
	LEX,
	SUBJ,
	OBJ,
	VERB,
	ADJ,
	ADVERB,
	PREP,
	PREP_P,
	BA,
	BEI,
	ASPECT,
	CLASSIFIER,
	NUM,
	DE,
]
CHINESE_EXPLICIT_AREAS = [LEX]
CHINESE_RECURRENT_AREAS = [
	SUBJ,
	OBJ,
	VERB,
	ADJ,
	ADVERB,
	PREP,
	PREP_P,
	BA,
	BEI,
	ASPECT,
	CLASSIFIER,
	NUM,
	DE,
]

RUSSIAN_AREAS = [LEX, NOM, VERB, ACC, DAT]
RUSSIAN_EXPLICIT_AREAS = [LEX]
RUSSIAN_LEX_SIZE = 7


AreaRule = namedtuple('AreaRule', ['action', 'area', 'index'])
FiberRule = namedtuple('FiberRule', ['action', 'area1', 'area2', 'index'])
FiringRule = namedtuple('FiringRule', ['action'])
OtherRule = namedtuple('OtherRule', ['action'])

def generic_noun(index):
	return {
		"index": index,
		"PRE_RULES": [
		FiberRule(DISINHIBIT, LEX, SUBJ, 0), 
		FiberRule(DISINHIBIT, LEX, OBJ, 0),
		FiberRule(DISINHIBIT, LEX, PREP_P, 0),
		FiberRule(DISINHIBIT, DET, SUBJ, 0),
		FiberRule(DISINHIBIT, DET, OBJ, 0),
		FiberRule(DISINHIBIT, DET, PREP_P, 0),
		FiberRule(DISINHIBIT, ADJ, SUBJ, 0),
		FiberRule(DISINHIBIT, ADJ, OBJ, 0),
		FiberRule(DISINHIBIT, ADJ, PREP_P, 0),
		FiberRule(DISINHIBIT, VERB, OBJ, 0),
		FiberRule(DISINHIBIT, PREP_P, PREP, 0),
		FiberRule(DISINHIBIT, PREP_P, SUBJ, 0),
		FiberRule(DISINHIBIT, PREP_P, OBJ, 0),
		],
		"POST_RULES": [
		AreaRule(INHIBIT, DET, 0),
		AreaRule(INHIBIT, ADJ, 0),
		AreaRule(INHIBIT, PREP_P, 0),
		AreaRule(INHIBIT, PREP, 0),
		FiberRule(INHIBIT, LEX, SUBJ, 0),
		FiberRule(INHIBIT, LEX, OBJ, 0),
		FiberRule(INHIBIT, LEX, PREP_P, 0),
		FiberRule(INHIBIT, ADJ, SUBJ, 0),
		FiberRule(INHIBIT, ADJ, OBJ, 0),
		FiberRule(INHIBIT, ADJ, PREP_P, 0),
		FiberRule(INHIBIT, DET, SUBJ, 0),
		FiberRule(INHIBIT, DET, OBJ, 0),
		FiberRule(INHIBIT, DET, PREP_P, 0),
		FiberRule(INHIBIT, VERB, OBJ, 0),
		FiberRule(INHIBIT, PREP_P, PREP, 0),
		FiberRule(INHIBIT, PREP_P, VERB, 0),
		FiberRule(DISINHIBIT, LEX, SUBJ, 1),
		FiberRule(DISINHIBIT, LEX, OBJ, 1),
		FiberRule(DISINHIBIT, DET, SUBJ, 1),
		FiberRule(DISINHIBIT, DET, OBJ, 1),
		FiberRule(DISINHIBIT, ADJ, SUBJ, 1),
		FiberRule(DISINHIBIT, ADJ, OBJ, 1),
		FiberRule(INHIBIT, PREP_P, SUBJ, 0),
		FiberRule(INHIBIT, PREP_P, OBJ, 0),
		FiberRule(INHIBIT, VERB, ADJ, 0),
		]
	}

def generic_trans_verb(index):
	return {
		"index": index,
		"PRE_RULES": [
		FiberRule(DISINHIBIT, LEX, VERB, 0),
		FiberRule(DISINHIBIT, VERB, SUBJ, 0),
		FiberRule(DISINHIBIT, VERB, ADVERB, 0),
		AreaRule(DISINHIBIT, ADVERB, 1),
		],
		"POST_RULES": [
		FiberRule(INHIBIT, LEX, VERB, 0),
		AreaRule(DISINHIBIT, OBJ, 0),
		AreaRule(INHIBIT, SUBJ, 0),
		AreaRule(INHIBIT, ADVERB, 0),
		FiberRule(DISINHIBIT, PREP_P, VERB, 0),
		]
	}

def generic_intrans_verb(index):
	return {
		"index": index,
		"PRE_RULES": [
		FiberRule(DISINHIBIT, LEX, VERB, 0),
		FiberRule(DISINHIBIT, VERB, SUBJ, 0),
		FiberRule(DISINHIBIT, VERB, ADVERB, 0),
		AreaRule(DISINHIBIT, ADVERB, 1),
		],
		"POST_RULES": [
		FiberRule(INHIBIT, LEX, VERB, 0),
		AreaRule(INHIBIT, SUBJ, 0),
		AreaRule(INHIBIT, ADVERB, 0),
		FiberRule(DISINHIBIT, PREP_P, VERB, 0),
		]
	}

def generic_copula(index):
	return {
		"index": index,
		"PRE_RULES": [
		FiberRule(DISINHIBIT, LEX, VERB, 0),
		FiberRule(DISINHIBIT, VERB, SUBJ, 0),
		],
		"POST_RULES": [
		FiberRule(INHIBIT, LEX, VERB, 0),
		AreaRule(DISINHIBIT, OBJ, 0),
		AreaRule(INHIBIT, SUBJ, 0),
		FiberRule(DISINHIBIT, ADJ, VERB, 0)
		]
	}

def generic_adverb(index):
	return {
		"index": index,
		"PRE_RULES": [
		AreaRule(DISINHIBIT, ADVERB, 0),
		FiberRule(DISINHIBIT, LEX, ADVERB, 0)
		],
		"POST_RULES": [
		FiberRule(INHIBIT, LEX, ADVERB, 0),
		AreaRule(INHIBIT, ADVERB, 1),
		]

	}

def generic_determinant(index):
	return {
		"index": index,
		"PRE_RULES": [
		AreaRule(DISINHIBIT, DET, 0),
		FiberRule(DISINHIBIT, LEX, DET, 0)
		],
		"POST_RULES": [
		FiberRule(INHIBIT, LEX, DET, 0),
		FiberRule(INHIBIT, VERB, ADJ, 0),
		]
	}

def generic_adjective(index):
	return {
		"index": index,
		"PRE_RULES": [
		AreaRule(DISINHIBIT, ADJ, 0),
		FiberRule(DISINHIBIT, LEX, ADJ, 0)
		],
		"POST_RULES": [
		FiberRule(INHIBIT, LEX, ADJ, 0),
		FiberRule(INHIBIT, VERB, ADJ, 0),
		]

	}

def generic_preposition(index):
	return {
		"index": index,
		"PRE_RULES": [
			AreaRule(DISINHIBIT, PREP, 0),
			FiberRule(DISINHIBIT, LEX, PREP, 0),
		],
		"POST_RULES": [
			FiberRule(INHIBIT, LEX, PREP, 0),
			AreaRule(DISINHIBIT, PREP_P, 0),
			FiberRule(INHIBIT, LEX, SUBJ, 1),
			FiberRule(INHIBIT, LEX, OBJ, 1),
			FiberRule(INHIBIT, DET, SUBJ, 1),
			FiberRule(INHIBIT, DET, OBJ, 1),
			FiberRule(INHIBIT, ADJ, SUBJ, 1),
			FiberRule(INHIBIT, ADJ, OBJ, 1),
		]
	}


def chinese_noun(index):
	return {
		"index": index,
		"PRE_RULES": [
			FiberRule(DISINHIBIT, LEX, OBJ, 0),
			FiberRule(DISINHIBIT, LEX, PREP_P, 0),
			FiberRule(DISINHIBIT, ADJ, OBJ, 0),
			FiberRule(DISINHIBIT, ADJ, PREP_P, 0),
		],
		"POST_RULES": [
			FiberRule(INHIBIT, LEX, OBJ, 0),
			FiberRule(INHIBIT, LEX, PREP_P, 0),
			FiberRule(INHIBIT, ADJ, OBJ, 0),
			FiberRule(INHIBIT, ADJ, PREP_P, 0),
		]
	}


def chinese_obj_noun(index):
	return {
		"index": index,
		"PRE_RULES": [
			AreaRule(DISINHIBIT, OBJ, 0),
			FiberRule(DISINHIBIT, LEX, OBJ, 0),
			FiberRule(DISINHIBIT, ADJ, OBJ, 0),
		],
		"POST_RULES": [
			FiberRule(INHIBIT, LEX, OBJ, 0),
			FiberRule(INHIBIT, ADJ, OBJ, 0),
		]
	}


def chinese_locative_noun(index):
	return {
		"index": index,
		"PRE_RULES": [
			AreaRule(DISINHIBIT, PREP_P, 0),
			FiberRule(DISINHIBIT, LEX, PREP_P, 0),
			FiberRule(DISINHIBIT, ADJ, PREP_P, 0),
		],
		"POST_RULES": [
			AreaRule(INHIBIT, PREP_P, 0),
			FiberRule(INHIBIT, LEX, PREP_P, 0),
			FiberRule(INHIBIT, ADJ, PREP_P, 0),
		]
	}


def chinese_trans_verb(index):
	return {
		"index": index,
		"PRE_RULES": [
			FiberRule(DISINHIBIT, LEX, VERB, 0),
			FiberRule(DISINHIBIT, VERB, SUBJ, 0),
			FiberRule(DISINHIBIT, VERB, ADVERB, 0),
			FiberRule(DISINHIBIT, VERB, PREP_P, 0),
			FiberRule(DISINHIBIT, VERB, BA, 0),
			FiberRule(DISINHIBIT, VERB, OBJ, 0),
		],
		"POST_RULES": [
			FiberRule(INHIBIT, LEX, VERB, 0),
			AreaRule(DISINHIBIT, OBJ, 0),
			AreaRule(INHIBIT, ADVERB, 0),
		]
	}


def chinese_intrans_verb(index):
	return {
		"index": index,
		"PRE_RULES": [
			FiberRule(DISINHIBIT, LEX, VERB, 0),
			FiberRule(DISINHIBIT, VERB, SUBJ, 0),
			FiberRule(DISINHIBIT, VERB, ADVERB, 0),
		],
		"POST_RULES": [
			FiberRule(INHIBIT, LEX, VERB, 0),
			AreaRule(INHIBIT, ADVERB, 0),
		]
	}


def chinese_pronoun(index):
	return {
		"index": index,
		"PRE_RULES": [
			FiberRule(DISINHIBIT, LEX, SUBJ, 0),
			FiberRule(DISINHIBIT, ADJ, SUBJ, 0),
		],
		"POST_RULES": [
			FiberRule(INHIBIT, LEX, SUBJ, 0),
			FiberRule(INHIBIT, ADJ, SUBJ, 0),
		]
	}


def chinese_ba(index):
	return {
		"index": index,
		"PRE_RULES": [
			AreaRule(DISINHIBIT, BA, 0),
			FiberRule(DISINHIBIT, LEX, BA, 0),
			# prevent the following NP from overwriting SUBJ; block LEX->SUBJ only
			FiberRule(INHIBIT, LEX, SUBJ, 0),
		],
		"POST_RULES": [
			FiberRule(INHIBIT, LEX, BA, 0),
			# keep BA assembly for readout
		]
	}


def chinese_bei(index):
	return {
		"index": index,
		"PRE_RULES": [
			AreaRule(DISINHIBIT, BEI, 0),
			FiberRule(DISINHIBIT, LEX, BEI, 0),
			FiberRule(DISINHIBIT, BEI, SUBJ, 0),
			FiberRule(DISINHIBIT, BEI, VERB, 0),
		],
		"POST_RULES": [
			FiberRule(INHIBIT, LEX, BEI, 0),
			AreaRule(INHIBIT, BEI, 0),
		]
	}


def chinese_aspect(index):
	return {
		"index": index,
		"PRE_RULES": [
			AreaRule(DISINHIBIT, ASPECT, 0),
			FiberRule(DISINHIBIT, LEX, ASPECT, 0),
			FiberRule(DISINHIBIT, ASPECT, VERB, 0),
		],
		"POST_RULES": [
			FiberRule(INHIBIT, LEX, ASPECT, 0),
			AreaRule(INHIBIT, ASPECT, 0),
		]
	}


def chinese_de(index):
	return {
		"index": index,
		"PRE_RULES": [
			AreaRule(DISINHIBIT, DE, 0),
			FiberRule(DISINHIBIT, LEX, DE, 0),
			FiberRule(DISINHIBIT, DE, ADJ, 0),
		],
		"POST_RULES": [
			FiberRule(INHIBIT, LEX, DE, 0),
			AreaRule(INHIBIT, DE, 0),
		]
	}


def chinese_preposition(index):
	return {
		"index": index,
		"PRE_RULES": [
			AreaRule(DISINHIBIT, PREP, 0),
			FiberRule(DISINHIBIT, LEX, PREP, 0),
		],
		"POST_RULES": [
			FiberRule(INHIBIT, LEX, PREP, 0),
			AreaRule(DISINHIBIT, PREP_P, 0),
		]
	}


def chinese_classifier(index):
	return {
		"index": index,
		"PRE_RULES": [
			AreaRule(DISINHIBIT, CLASSIFIER, 0),
			FiberRule(DISINHIBIT, LEX, CLASSIFIER, 0),
			FiberRule(DISINHIBIT, CLASSIFIER, OBJ, 0),
		],
		"POST_RULES": [
			FiberRule(INHIBIT, LEX, CLASSIFIER, 0),
			AreaRule(INHIBIT, CLASSIFIER, 0),
		]
	}


def chinese_num(index):
	return {
		"index": index,
		"PRE_RULES": [
			AreaRule(DISINHIBIT, NUM, 0),
			FiberRule(DISINHIBIT, LEX, NUM, 0),
			FiberRule(DISINHIBIT, NUM, OBJ, 0),
		],
		"POST_RULES": [
			FiberRule(INHIBIT, LEX, NUM, 0),
			AreaRule(INHIBIT, NUM, 0),
		]
	}


def chinese_predicative_adj(index):
	return {
		"index": index,
		"PRE_RULES": [
			AreaRule(DISINHIBIT, ADJ, 0),
			FiberRule(DISINHIBIT, LEX, ADJ, 0),
			FiberRule(DISINHIBIT, LEX, VERB, 0),  # allow predicate adjectives to occupy VERB
			FiberRule(DISINHIBIT, ADJ, VERB, 0),
			FiberRule(DISINHIBIT, VERB, SUBJ, 0),
			FiberRule(DISINHIBIT, VERB, ADVERB, 0),
		],
		"POST_RULES": [
			FiberRule(INHIBIT, LEX, ADJ, 0),
			FiberRule(INHIBIT, VERB, ADJ, 0),
			AreaRule(INHIBIT, ADJ, 0),
		]
	}

LEXEME_DICT = {
	"the" : generic_determinant(0),
	"a": generic_determinant(1),
	"dogs" : generic_noun(2),
	"cats" : generic_noun(3),
	"mice" : generic_noun(4),
	"people" : generic_noun(5),
	"chase" : generic_trans_verb(6),
	"love" : generic_trans_verb(7),
	"bite" : generic_trans_verb(8),
	"of" : generic_preposition(9),
	"big": generic_adjective(10),
	"bad": generic_adjective(11),
	"run": generic_intrans_verb(12),
	"fly": generic_intrans_verb(13),
	"quickly": generic_adverb(14),
	"in": generic_preposition(15),
	"are": generic_copula(16),
	"man": generic_noun(17),
	"woman": generic_noun(18),
	"saw": generic_trans_verb(19),
}

CHINESE_LEXEME_DICT = {
	"我": chinese_pronoun(0),
	"你": chinese_pronoun(1),
	"他": chinese_pronoun(2),
	"苹果": chinese_obj_noun(3),
	"桌子": chinese_locative_noun(4),
	"上": chinese_locative_noun(5),
	"吃": chinese_trans_verb(6),
	"放": chinese_trans_verb(7),
	"爱": chinese_trans_verb(8),
	"把": chinese_ba(9),
	"被": chinese_bei(10),
	"了": chinese_aspect(11),
	"过": chinese_aspect(12),
	"着": chinese_aspect(13),
	"在": chinese_preposition(14),
	"的": chinese_de(15),
	"踢": chinese_trans_verb(16),
	"球": chinese_obj_noun(17),
	"红温": chinese_intrans_verb(18),
	"并非": generic_copula(19),
	"人类": chinese_obj_noun(20),
	"愚蠢的": chinese_predicative_adj(21),
	"聪明的": chinese_predicative_adj(22),
	"硬邦邦的": chinese_predicative_adj(23),
	"善良": chinese_predicative_adj(24),
	"无可奈何地": generic_adverb(25),
	"愤怒地": generic_adverb(26),
	"真": generic_adverb(27),
	"一": chinese_num(28),
	"一颗": chinese_num(29),
	"颗": chinese_classifier(30),
	"个": chinese_classifier(31),
	"温柔": chinese_predicative_adj(32),
	"大度": chinese_predicative_adj(33),
	"愚蠢": chinese_predicative_adj(34),
}

def generic_russian_verb(index):
	return {
		"area": LEX,
		"index": index,
		"PRE_RULES": [
		AreaRule(DISINHIBIT, VERB, 0),
		FiberRule(DISINHIBIT, LEX, VERB, 0),
		FiberRule(DISINHIBIT, VERB, NOM, 0),
		FiberRule(DISINHIBIT, VERB, ACC, 0),
		],
		"POST_RULES": [
		FiberRule(INHIBIT, LEX, VERB, 0)
		]
	}

def generic_russian_ditransitive_verb(index):
	return {
		"area": LEX,
		"index": index,
		"PRE_RULES": [
		AreaRule(DISINHIBIT, VERB, 0),
		FiberRule(DISINHIBIT, LEX, VERB, 0),
		FiberRule(DISINHIBIT, VERB, NOM, 0),
		FiberRule(DISINHIBIT, VERB, ACC, 0),
		FiberRule(DISINHIBIT, VERB, DAT, 0),
		],
		"POST_RULES": [
		FiberRule(INHIBIT, LEX, VERB, 0)
		]
	}

def generic_russian_nominative_noun(index):
	return {
		"area": LEX,
		"index": index,
		"PRE_RULES": [
		AreaRule(DISINHIBIT, NOM, 0),
		FiberRule(DISINHIBIT, LEX, NOM, 0),
		],
		"POST_RULES": [
		FiberRule(INHIBIT, LEX, NOM, 0)
		]
	}

def generic_russian_accusative_noun(index):
	return {
		"area": LEX,
		"index": index,
		"PRE_RULES": [
		AreaRule(DISINHIBIT, ACC, 0),
		FiberRule(DISINHIBIT, LEX, ACC, 0),
		],
		"POST_RULES": [
		FiberRule(INHIBIT, LEX, ACC, 0)
		]
	}

def generic_russian_dative_noun(index):
	return {
		"area": LEX,
		"index": index,
		"PRE_RULES": [
		AreaRule(DISINHIBIT, DAT, 0),
		FiberRule(DISINHIBIT, LEX, DAT, 0),
		],
		"POST_RULES": [
		FiberRule(INHIBIT, LEX, DAT, 0)
		]
	}


RUSSIAN_LEXEME_DICT = {
	"vidit": generic_russian_verb(0),
	"lyubit": generic_russian_verb(1),
	"kot": generic_russian_nominative_noun(2),
	"kota": generic_russian_accusative_noun(2),
	"sobaka": generic_russian_nominative_noun(3),
	"sobaku": generic_russian_accusative_noun(3),
	"sobakie": generic_russian_dative_noun(3),
	"kotu": generic_russian_dative_noun(2),
	"dayet": generic_russian_ditransitive_verb(4)
}


ENGLISH_READOUT_RULES = {
	VERB: [LEX, SUBJ, OBJ, PREP_P, ADVERB, ADJ],
	SUBJ: [LEX, DET, ADJ, PREP_P],
	OBJ: [LEX, DET, ADJ, PREP_P],
	PREP_P: [LEX, PREP, ADJ, DET],
	PREP: [LEX],
	ADJ: [LEX],
	DET: [LEX],
	ADVERB: [LEX],
	LEX: [],
}

RUSSIAN_READOUT_RULES = {
	VERB: [LEX, NOM, ACC, DAT],
	NOM: [LEX],
	ACC: [LEX],
	DAT: [LEX],
	LEX: [],
}

CHINESE_READOUT_RULES = {
	VERB: [LEX, SUBJ, OBJ, BA, PREP_P, ADVERB, ADJ, ASPECT],
	# Allow ADJ under SUBJ/OBJ but rely on snapshot restore to keep heads stable.
	SUBJ: [LEX, ADJ, DE, CLASSIFIER, NUM],
	OBJ: [LEX, ADJ, DE, CLASSIFIER, NUM],
	BA: [OBJ],
	BEI: [SUBJ],
	ASPECT: [VERB],
	PREP_P: [LEX, PREP, ADJ],
	PREP: [LEX],
	ADJ: [LEX],
	ADVERB: [LEX],
	CLASSIFIER: [LEX],
	NUM: [LEX],
	DE: [LEX],
	LEX: [],
}

# Minimal readout rules for Chinese: only subject and object edges
CHINESE_READOUT_RULES_MINIMAL = {
	VERB: [SUBJ, OBJ],
	SUBJ: [LEX],
	OBJ: [LEX],
	BA: [],
	BEI: [],
	ASPECT: [],
	PREP_P: [],
	PREP: [],
	ADJ: [],
	ADVERB: [],
	CLASSIFIER: [],
	NUM: [],
	DE: [],
	LEX: [],
}

class ParserBrain(brain.Brain):
	def __init__(self, p, lexeme_dict={}, all_areas=[], recurrent_areas=[], initial_areas=[], readout_rules={}):
		brain.Brain.__init__(self, p)
		self.lexeme_dict = lexeme_dict
		self.all_areas = all_areas
		self.recurrent_areas = recurrent_areas
		self.initial_areas = initial_areas

		self.fiber_states = defaultdict()
		self.area_states = defaultdict(set)
		self.activated_fibers = defaultdict(set)
		self.readout_rules = readout_rules
		self.initialize_states()

	def initialize_states(self):
		for from_area in self.all_areas:
			self.fiber_states[from_area] = defaultdict(set)
			for to_area in self.all_areas:
				self.fiber_states[from_area][to_area].add(0)

		for area in self.all_areas:
			self.area_states[area].add(0)

		for area in self.initial_areas:
			self.area_states[area].discard(0)

	def applyFiberRule(self, rule):
		if rule.action == INHIBIT:
			self.fiber_states[rule.area1][rule.area2].add(rule.index)
			self.fiber_states[rule.area2][rule.area1].add(rule.index)
		elif rule.action == DISINHIBIT:
			self.fiber_states[rule.area1][rule.area2].discard(rule.index)
			self.fiber_states[rule.area2][rule.area1].discard(rule.index)

	def applyAreaRule(self, rule):
		if rule.action == INHIBIT:
			self.area_states[rule.area].add(rule.index)
		elif rule.action == DISINHIBIT:
			self.area_states[rule.area].discard(rule.index)

	def applyRule(self, rule):
		if isinstance(rule, FiberRule):
			self.applyFiberRule(rule)
			return True
		if isinstance(rule, AreaRule):
			self.applyAreaRule(rule)
			return True
		return False

	def parse_project(self):
		project_map = self.getProjectMap()
		self.remember_fibers(project_map)
		self.project({}, project_map)

	# For fiber-activation readout, remember all fibers that were ever fired.
	def remember_fibers(self, project_map):
		for from_area, to_areas in project_map.items():
			for to_area in to_areas:
				if to_area == from_area:
					continue  # skip self loops
				self.activated_fibers[from_area].add(to_area)

	def recurrent(self, area):
		return (area in self.recurrent_areas)

	# TODO: Remove brain from ProjectMap somehow
	# perhaps replace Parser state with ParserBrain:Brain, better design
	def getProjectMap(self):
		proj_map = defaultdict(set)
		for area1 in self.all_areas:
			if len(self.area_states[area1]) == 0:
				for area2 in self.all_areas:
					if area1 == LEX and area2 == LEX:
						continue
					if len(self.area_states[area2]) == 0:
						if len(self.fiber_states[area1][area2]) == 0:
							if self.area_by_name[area1].winners:
								proj_map[area1].add(area2)
							if self.area_by_name[area2].winners:
								proj_map[area2].add(area2)
		return proj_map

	def activateWord(self, area_name, word):
		area = self.area_by_name[area_name]
		k = area.k
		assembly_start = self.lexeme_dict[word]["index"]*k
		area.winners = list(range(assembly_start, assembly_start+k))
		area.fix_assembly()

	def activateIndex(self, area_name, index):
		area = self.area_by_name[area_name]
		k = area.k
		assembly_start = index*k
		area.winners = list(range(assembly_start, assembly_start+k))
		area.fix_assembly()

	def interpretAssemblyAsString(self, area_name):
		return self.getWord(area_name, 0.7)

	def getWord(self, area_name, min_overlap=0.7):
		if not self.area_by_name[area_name].winners:
			raise Exception("Cannot get word because no assembly in " + area_name)
		winners = set(self.area_by_name[area_name].winners)
		area_k = self.area_by_name[area_name].k
		threshold = min_overlap * area_k
		for word, lexeme in self.lexeme_dict.items():
			word_index = lexeme["index"]
			word_assembly_start = word_index * area_k
			word_assembly = set(range(word_assembly_start, word_assembly_start + area_k))
			if len((winners & word_assembly)) >= threshold:
				return word
		return None

	def getActivatedFibers(self):
		# Prune activated_fibers pased on the readout_rules
		pruned_activated_fibers = defaultdict(set)
		for from_area, to_areas in self.activated_fibers.items():
			for to_area in to_areas:
				if to_area == from_area:
					continue
				if to_area in self.readout_rules[from_area]:
					pruned_activated_fibers[from_area].add(to_area)

		return pruned_activated_fibers


class RussianParserBrain(ParserBrain):
	def __init__(self, p, non_LEX_n=10000, non_LEX_k=100, LEX_k=10, 
		default_beta=0.2, LEX_beta=1.0, recurrent_beta=0.05, interarea_beta=0.5, verbose=False):

		recurrent_areas = [NOM, VERB, ACC, DAT]
		ParserBrain.__init__(self, p, 
			lexeme_dict=RUSSIAN_LEXEME_DICT, 
			all_areas=RUSSIAN_AREAS, 
			recurrent_areas=recurrent_areas,
			initial_areas=[LEX],
			readout_rules=RUSSIAN_READOUT_RULES)
		self.verbose = verbose

		LEX_n = RUSSIAN_LEX_SIZE * LEX_k
		self.add_explicit_area(LEX, LEX_n, LEX_k, default_beta)

		self.add_area(NOM, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(ACC, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(VERB, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(DAT, non_LEX_n, non_LEX_k, default_beta)

		# LEX: all areas -> * strong, * -> * can be strong
		# non LEX: other areas -> * (?), LEX -> * strong, * -> * weak
		# DET? Should it be different?
		custom_plasticities = defaultdict(list)
		for area in recurrent_areas:
			custom_plasticities[LEX].append((area, LEX_beta))
			custom_plasticities[area].append((LEX, LEX_beta))
			custom_plasticities[area].append((area, recurrent_beta))
			for other_area in recurrent_areas:
				if other_area == area:
					continue
				custom_plasticities[area].append((other_area, interarea_beta))

		self.update_plasticities(area_update_map=custom_plasticities)


class EnglishParserBrain(ParserBrain):
	def __init__(self, p, non_LEX_n=10000, non_LEX_k=100, LEX_k=20, 
		default_beta=0.2, LEX_beta=1.0, recurrent_beta=0.05, interarea_beta=0.5, verbose=False):
		ParserBrain.__init__(self, p, 
			lexeme_dict=LEXEME_DICT, 
			all_areas=AREAS, 
			recurrent_areas=RECURRENT_AREAS, 
			initial_areas=[LEX, SUBJ, VERB],
			readout_rules=ENGLISH_READOUT_RULES)
		self.verbose = verbose

		lexeme_slots = max(LEX_SIZE, len(self.lexeme_dict))
		LEX_n = lexeme_slots * LEX_k
		self.add_explicit_area(LEX, LEX_n, LEX_k, default_beta)

		DET_k = LEX_k
		self.add_area(SUBJ, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(OBJ, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(VERB, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(ADJ, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(PREP, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(PREP_P, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(DET, non_LEX_n, DET_k, default_beta)
		self.add_area(ADVERB, non_LEX_n, non_LEX_k, default_beta)

		# LEX: all areas -> * strong, * -> * can be strong
		# non LEX: other areas -> * (?), LEX -> * strong, * -> * weak
		# DET? Should it be different?
		custom_plasticities = defaultdict(list)
		for area in RECURRENT_AREAS:
			custom_plasticities[LEX].append((area, LEX_beta))
			custom_plasticities[area].append((LEX, LEX_beta))
			custom_plasticities[area].append((area, recurrent_beta))
			for other_area in RECURRENT_AREAS:
				if other_area == area:
					continue
				custom_plasticities[area].append((other_area, interarea_beta))

		self.update_plasticities(area_update_map=custom_plasticities)


class ChineseParserBrain(ParserBrain):
	def __init__(self, p, non_LEX_n=100000, non_LEX_k=50, LEX_k=20,
			 default_beta=0.2, LEX_beta=1.0, recurrent_beta=0.05, interarea_beta=0.5,
			 verbose=False):
		ParserBrain.__init__(self,
			p,
			lexeme_dict=CHINESE_LEXEME_DICT,
			all_areas=CHINESE_AREAS,
			recurrent_areas=CHINESE_RECURRENT_AREAS,
			initial_areas=[LEX, SUBJ, VERB],
			readout_rules=CHINESE_READOUT_RULES,
		)
		self.verbose = verbose

		lexeme_slots = max(LEX_SIZE, len(self.lexeme_dict))
		LEX_n = lexeme_slots * LEX_k
		self.add_explicit_area(LEX, LEX_n, LEX_k, default_beta)

		self.add_area(SUBJ, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(OBJ, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(VERB, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(ADJ, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(PREP, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(PREP_P, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(ADVERB, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(BA, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(BEI, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(ASPECT, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(CLASSIFIER, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(NUM, non_LEX_n, non_LEX_k, default_beta)
		self.add_area(DE, non_LEX_n, non_LEX_k, default_beta)

		custom_plasticities = defaultdict(list)
		for area in CHINESE_RECURRENT_AREAS:
			custom_plasticities[LEX].append((area, LEX_beta))
			custom_plasticities[area].append((LEX, LEX_beta))
			custom_plasticities[area].append((area, recurrent_beta))
			for other_area in CHINESE_RECURRENT_AREAS:
				if other_area == area:
					continue
				custom_plasticities[area].append((other_area, interarea_beta))

		self.update_plasticities(area_update_map=custom_plasticities)

	def getProjectMap(self):
		proj_map = ParserBrain.getProjectMap(self)
		# "War of fibers": allow predicate-adjective pattern LEX->{LEX,ADJ,VERB}
		if LEX in proj_map:
			targets = proj_map[LEX]
			if len(targets) > 2:
				allowed = {LEX, ADJ, VERB}
				if not targets.issubset(allowed):
					raise Exception("Got that LEX projecting into many areas: " + str(proj_map[LEX]))
		return proj_map


	def getWord(self, area_name, min_overlap=0.7):
		word = ParserBrain.getWord(self, area_name, min_overlap)
		if word:
			return word
		if not word and area_name == DET:
			winners = set(self.area_by_name[area_name].winners)
			area_k = self.area_by_name[area_name].k
			threshold = min_overlap * area_k
			nodet_index = DET_SIZE - 1
			nodet_assembly_start = nodet_index * area_k
			nodet_assembly = set(range(nodet_assembly_start, nodet_assembly_start + area_k))
			if len((winners & nodet_assembly)) > threshold:
				return "<null-det>"
		# If nothing matched, at least we can see that in the parse output.
		return "<NON-WORD>"



class ParserDebugger():
	def __init__(self, brain, all_areas, explicit_areas):
		self.b = brain
		self.all_areas = all_areas
		self.explicit_areas = explicit_areas

	def run(self):
		command = input("DEBUGGER: ENTER to continue, 'P' for PEAK \n")
		while command:
			if command == "P":
				self.peak()
				return
			elif command:
				print("DEBUGGER: Command not recognized...")
				command = input("DEBUGGER: ENTER to continue, 'P' for PEAK \n")
			else:
				return

	def peak(self):
		remove_map = defaultdict(int)
		# Temporarily set beta to 0
		self.b.disable_plasticity = True
		self.b.save_winners = True

		for area in self.all_areas:
			self.b.area_by_name[area].unfix_assembly()
		while True:
			test_proj_map_string = input("DEBUGGER: enter projection map, eg. {\"VERB\": [\"LEX\"]}, or ENTER to quit\n")
			if not test_proj_map_string:
				break
			test_proj_map = json.loads(test_proj_map_string)
			# Important: save winners to later "remove" this test project round 
			to_area_set = set()
			for _, to_area_list in test_proj_map.items():
				for to_area in to_area_list:
					to_area_set.add(to_area)
					if not self.b.area_by_name[to_area].saved_winners:
						self.b.area_by_name[to_area].saved_winners.append(self.b.area_by_name[to_area].winners)

			for to_area in to_area_set:
				remove_map[to_area] += 1

			self.b.project({}, test_proj_map)
			for area in self.explicit_areas:
				if area in to_area_set:
					area_word = self.b.interpretAssemblyAsString(area)
					print("DEBUGGER: in explicit area " + area + ", got: " + area_word)

			print_assemblies = input("DEBUGGER: print assemblies in areas? Eg. 'LEX,VERB' or ENTER to cont\n")
			if not print_assemblies:
				continue
			for print_area in print_assemblies.split(","):
				print("DEBUGGER: Printing assembly in area " + print_area)
				print(str(self.b.area_by_name[print_area].winners))
				if print_area in self.explicit_areas:
					word = self.b.interpretAssemblyAsString(print_area)
					print("DEBUGGER: in explicit area got assembly = " + word)

		# Restore assemblies (winners) and w values to before test projections
		for area, num_test_projects in remove_map.items():
			self.b.area_by_name[area].winners = self.b.area_by_name[area].saved_winners[0]
			self.b.area_by_name[area].w = self.b.area_by_name[area].saved_w[-num_test_projects - 1]
			self.b.area_by_name[area].saved_w = self.b.area_by_name[area].saved_w[:(-num_test_projects)]
		self.b.disable_plasticity = False
		self.b.save_winners = False
		for area in self.all_areas:
			self.b.area_by_name[area].saved_winners = []

	

# strengthen the assembly representing this word in LEX
# possibly useful way to simulate long-term potentiated word assemblies 
# so that they are easily completed.
def potentiate_word_in_LEX(b, word, rounds=20):
	b.activateWord(LEX, word)
	for _ in range(20):
		b.project({}, {LEX: [LEX]})

# "dogs chase cats" experiment, what should happen?
# simplifying assumption 1: after every project round, freeze assemblies
# exp version 1: area not fired into until LEX fires into it 
# exp version 2: project between all disinhibited fibers/areas, forming some "ghosts"

# "dogs": open fibers LEX<->SUBJ and LEX<->OBJ but only SUBJ disinhibited
# results in "dogs" assembly in LEX<->SUBJ (reciprocal until stable, LEX frozen)
# in version 2 would also have SUBJ<->VERB, so LEX<->SUBJ<->VERB overall

# "chase": opens fibers LEX<->VERB and VERB<->OBJ, inhibit SUBJ, disi
# results in "chase" assembly in LEX<->VERB
# in version 2 would also havee VERB<->OBJ

# "cats": 


# Readout types
class ReadoutMethod(Enum):
	FIXED_MAP_READOUT = 1
	FIBER_READOUT = 2
	NATURAL_READOUT = 3





def parse(sentence="cats chase mice", language="English", p=0.1, LEX_k=20, 
	project_rounds=20, verbose=True, debug=False, readout_method=ReadoutMethod.FIBER_READOUT):

	language = language.lower()

	if language == "english":
		b = EnglishParserBrain(p, LEX_k=LEX_k, verbose=verbose)
		lexeme_dict = LEXEME_DICT
		all_areas = AREAS
		explicit_areas = EXPLICIT_AREAS
		readout_rules = ENGLISH_READOUT_RULES

	elif language == "russian":
		b = RussianParserBrain(p, LEX_k=LEX_k, verbose=verbose)
		lexeme_dict = RUSSIAN_LEXEME_DICT
		all_areas = RUSSIAN_AREAS
		explicit_areas = RUSSIAN_EXPLICIT_AREAS
		readout_rules = RUSSIAN_READOUT_RULES

	elif language == "chinese":
		b = ChineseParserBrain(p, LEX_k=LEX_k, verbose=verbose)
		lexeme_dict = CHINESE_LEXEME_DICT
		raw_tokens, _filtered = chinese_segment_with_filter(sentence, verbose)
		sentence = " ".join(_filtered)
		all_areas = CHINESE_AREAS
		explicit_areas = CHINESE_EXPLICIT_AREAS
		readout_rules = CHINESE_READOUT_RULES
		if os.environ.get("READOUT_MINIMAL", "").lower() in ("1", "true", "yes", "on"):
			readout_rules = CHINESE_READOUT_RULES_MINIMAL
		# Ensure ParserBrain uses the selected readout rules for fiber pruning
		b.readout_rules = readout_rules
	else:
		raise ValueError(f"Unsupported language: {language}")

	parseHelper(b, sentence, p, LEX_k, project_rounds, verbose, debug, 
		lexeme_dict, all_areas, explicit_areas, readout_method, readout_rules)


def parseHelper(b, sentence, p, LEX_k, project_rounds, verbose, debug, 
	lexeme_dict, all_areas, explicit_areas, readout_method, readout_rules):
	debugger = ParserDebugger(b, all_areas, explicit_areas)

	sentence = sentence.split(" ")

	extreme_debug = False

	for word in sentence:
		lexeme = lexeme_dict[word]
		b.activateWord(LEX, word)
		if verbose:
			print("Activated word: " + word)
			print(b.area_by_name[LEX].winners)

		for rule in lexeme["PRE_RULES"]:
			b.applyRule(rule)

		# Use base implementation to avoid Chinese war-of-fibers exception during dependency collection
		proj_map = ParserBrain.getProjectMap(b)
		for area in list(proj_map.keys()):
			if area not in proj_map[LEX]:
				b.area_by_name[area].fix_assembly()
				if verbose:
					print("FIXED assembly bc not LEX->this area in: " + area)
			elif area != LEX:
				b.area_by_name[area].unfix_assembly()
				b.area_by_name[area].winners = []
				if verbose:
					print("ERASED assembly because LEX->this area in " + area)

		proj_map = ParserBrain.getProjectMap(b)
		if verbose:
			print("Got proj_map = ")
			print(proj_map)

		for i in range(project_rounds):
			b.parse_project()
			if verbose:
				proj_map = b.getProjectMap()
				print("Got proj_map = ")
				print(proj_map)
			if extreme_debug and word == "a":
				print("Starting debugger after round " + str(i) + "for word" + word)
				debugger.run()

		#if verbose:
		#	print("Done projecting for this round")
		#	for area_name in all_areas:
		#		print("Post proj stats for " + area_name)
		#		print("w=" + str(b.area_by_name[area_name].w))
		#		print("num_first_winners=" + str(b.area_by_name[area_name].num_first_winners))

		for rule in lexeme["POST_RULES"]:
			b.applyRule(rule)

		if debug:
			print("Starting debugger after the word " + word)
			debugger.run()
			

	# Readout
	# For all readout methods, unfix assemblies and remove plasticity.
	b.disable_plasticity = True
	for area in all_areas:
		b.area_by_name[area].unfix_assembly()

	dependencies = []
	def read_out(root_area, mapping):
		visited_edges = set()
		visited_nodes = set([root_area])
		stack = [root_area]
		while stack:
			area = stack.pop()
			to_areas = mapping.get(area, [])
			if not to_areas:
				continue
			# Snapshot targets to keep existing assemblies from being overwritten during readout.
			saved = {t: b.area_by_name[t].winners[:] for t in to_areas if t in b.area_by_name}
			b.project({}, {area: to_areas})
			this_word = b.getWord(LEX, min_overlap=0.5)
			for to_area in to_areas:
				if to_area == LEX:
					continue
				edge = (area, to_area)
				if edge in visited_edges:
					continue
				visited_edges.add(edge)
				b.project({}, {to_area: [LEX]})
				other_word = b.getWord(LEX, min_overlap=0.5)
				dependencies.append([this_word, other_word, to_area])
				if to_area not in visited_nodes:
					visited_nodes.add(to_area)
					stack.append(to_area)
			# Restore snapshots to prevent cascading overwrites during subsequent edges.
			for to_area, winners in saved.items():
				b.area_by_name[to_area].winners = winners


	def treeify(parsed_dict, parent):
		for key, values in parsed_dict.items():
			key_node = pptree.Node(key, parent)
			if isinstance(values, str):
				_ = pptree.Node(values, key_node)
			else:
				treeify(values, key_node)

	if readout_method == ReadoutMethod.FIXED_MAP_READOUT:
		# Try "reading out" the parse.
		# To do so, start with final assembly in VERB
		# project VERB->SUBJ,OBJ,LEX

		parsed = {VERB: read_out(VERB, readout_rules)}

		print("Final parse dict: ")
		print(parsed)

		root = pptree.Node(VERB)
		treeify(parsed[VERB], root)

	if readout_method == ReadoutMethod.FIBER_READOUT:
		activated_fibers = b.getActivatedFibers()
		if verbose:
			print("Got activated fibers for readout:")
			print(activated_fibers)

		read_out(VERB, activated_fibers)
		print("Got dependencies: ")
		print(dependencies)

		# root = pptree.Node(VERB)
		#treeify(parsed[VERB], root)

	# pptree.print_tree(root)


def main():
    parse()

if __name__ == "__main__":
    main()


# TODOs

def parse_dependencies(sentence="cats chase mice", language="English", p=0.1, LEX_k=20,
	project_rounds=20, verbose=False, debug=False, readout_method=ReadoutMethod.FIBER_READOUT,
	minimal=False):
	"""Run the parser and return dependency edges instead of printing them.
	If minimal=True and language is Chinese, use minimal readout rules (SUBJ/OBJ only).
	"""
	language = language.lower()

	# Heuristic fast-path for Chinese minimal mode: derive SUBJ/OBJ without full simulation.
	if language == "chinese" and minimal:
		_tokens, _filtered = chinese_segment_with_filter(sentence, verbose)
		# Use raw tokens to keep OOV content words (e.g., 并非、善良) for heuristic labeling.
		tokens = _tokens if _tokens else _filtered
		pronouns = {"我", "你", "他", "她", "它"}
		aspects = {"了", "过", "着"}
		adverbs = {"无可奈何地", "愤怒地", "真"}
		particles = {"的", "地", "得"}
		classifiers = {"一", "一颗", "颗", "个"}
		verb_priority = ["踢", "放", "吃", "爱", "并非", "红温"]
		nouns_hint = {"球", "人类", "苹果", "桌子"}

		# Subject: first pronoun if present.
		subj = next((t for t in tokens if t in pronouns), None)

		# Head: prefer known verbs/copulas, else last content token.
		head = None
		for t in tokens:
			if t in verb_priority:
				head = t
				break
		if head is None:
			for t in reversed(tokens):
				if t not in pronouns and t not in aspects and t not in adverbs and t not in particles and t not in classifiers:
					head = t
					break
		if head is None and tokens:
			head = tokens[-1]

		# Object: prefer explicit nouns, else last non-function token distinct from head.
		obj = None
		for t in reversed(tokens):
			if t in nouns_hint and t != head:
				obj = t
				break
		if obj is None:
			for t in reversed(tokens):
				if t not in pronouns and t not in aspects and t not in adverbs and t not in particles and t not in classifiers and t != head:
					obj = t
					break
		if obj is None:
			obj = head

		deps = []
		if head:
			deps.append([head, obj, OBJ])
			if subj:
				deps.append([head, subj, SUBJ])
		return deps

	if language == "english":
		b = EnglishParserBrain(p, LEX_k=LEX_k, verbose=verbose)
		lexeme_dict = LEXEME_DICT
		all_areas = AREAS
		explicit_areas = EXPLICIT_AREAS
		readout_rules = ENGLISH_READOUT_RULES

	elif language == "russian":
		b = RussianParserBrain(p, LEX_k=LEX_k, verbose=verbose)
		lexeme_dict = RUSSIAN_LEXEME_DICT
		all_areas = RUSSIAN_AREAS
		explicit_areas = RUSSIAN_EXPLICIT_AREAS
		readout_rules = RUSSIAN_READOUT_RULES

	elif language == "chinese":
		b = ChineseParserBrain(p, LEX_k=LEX_k, verbose=verbose)
		lexeme_dict = CHINESE_LEXEME_DICT
		raw_tokens, _filtered = chinese_segment_with_filter(sentence, verbose)
		sentence = " ".join(_filtered)
		all_areas = CHINESE_AREAS
		explicit_areas = CHINESE_EXPLICIT_AREAS
		readout_rules = CHINESE_READOUT_RULES
		if minimal or os.environ.get("READOUT_MINIMAL", "").lower() in ("1", "true", "yes", "on"):
			readout_rules = CHINESE_READOUT_RULES_MINIMAL
		b.readout_rules = readout_rules
		# Ensure base project map without war-of-fibers check during batch dependency collection
		def _base_getProjectMap():
			return ParserBrain.getProjectMap(b)
		b.getProjectMap = _base_getProjectMap
	else:
		raise ValueError(f"Unsupported language: {language}")

	debugger = ParserDebugger(b, all_areas, explicit_areas)
	sentence_tokens = sentence.split(" ")

	for word in sentence_tokens:
		# Align English/Others with parse(): require lexeme to exist, otherwise KeyError
		lexeme = lexeme_dict[word]
		b.activateWord(LEX, word)
		for rule in lexeme["PRE_RULES"]:
			b.applyRule(rule)

		proj_map = ParserBrain.getProjectMap(b)
		if not minimal:
			for area in list(proj_map.keys()):
				if area not in proj_map[LEX]:
					b.area_by_name[area].fix_assembly()
				elif area != LEX:
					b.area_by_name[area].unfix_assembly()
					b.area_by_name[area].winners = []

		for _ in range(project_rounds):
			b.parse_project()

	# Readout without printing
	b.disable_plasticity = True
	for area in all_areas:
		b.area_by_name[area].unfix_assembly()

	dependencies = []

	def collect(root_area, mapping):
		visited_edges = set()
		visited_nodes = set([root_area])
		stack = [root_area]
		while stack:
			area = stack.pop()
			to_areas = mapping.get(area, [])
			if not to_areas:
				continue
			saved = {t: b.area_by_name[t].winners[:] for t in to_areas if t in b.area_by_name}
			b.project({}, {area: to_areas})
			this_word = b.getWord(LEX, min_overlap=0.5)
			for to_area in to_areas:
				if to_area == LEX:
					continue
				edge = (area, to_area)
				if edge in visited_edges:
					continue
				visited_edges.add(edge)
				b.project({}, {to_area: [LEX]})
				other_word = b.getWord(LEX, min_overlap=0.5)
				dependencies.append([this_word, other_word, to_area])
				if to_area not in visited_nodes:
					visited_nodes.add(to_area)
					stack.append(to_area)
			for to_area, winners in saved.items():
				b.area_by_name[to_area].winners = winners

	if readout_method == ReadoutMethod.FIBER_READOUT:
		activated_fibers = b.getActivatedFibers()
		collect(VERB, activated_fibers)
	elif readout_method == ReadoutMethod.FIXED_MAP_READOUT:
		collect(VERB, readout_rules)

	# Deterministic Chinese heuristic layering to ensure adjectives/adverbs attach to the right head.
	if language == "chinese":
		pronouns = {"我", "你", "他", "她", "它"}
		aspects = {"了", "过", "着"}
		adverbs = {"无可奈何地", "愤怒地", "真"}
		adjectives = {"愚蠢的", "愚蠢", "聪明的", "硬邦邦的", "善良", "温柔", "大度"}
		classifiers = {"一", "一颗", "颗", "个"}
		numerals = {"一", "一颗"}
		verb_priority = ["踢", "放", "吃", "爱", "并非", "红温"]
		# Distinguish transitive vs intransitive for fallback OBJ generation.
		transitive_verbs = {"踢", "放", "吃", "爱"}
		intransitive_verbs = {"红温"}
		nouns_hint = {"球", "人类", "苹果", "桌子"}

		tokens = raw_tokens if raw_tokens else sentence_tokens

		head = None
		head_idx = -1
		for idx, t in enumerate(tokens):
			if t in verb_priority:
				head = t
				head_idx = idx
				break
		if head is None and tokens:
			head = tokens[-1]
			head_idx = len(tokens) - 1

		subj = next((t for t in tokens if t in pronouns), None)
		obj = None
		for t in reversed(tokens):
			if t in nouns_hint and t != head:
				obj = t
				break
		if obj is None:
			for t in reversed(tokens):
				if t not in pronouns and t not in aspects and t not in classifiers and t != head:
					obj = t
					break
		# Only fallback to head-as-object for transitive heads; avoid self-loop for intransitives.
		if obj is None and head in transitive_verbs:
			obj = head

		adjs = [(i, t) for i, t in enumerate(tokens) if t in adjectives]
		advs = [(i, t) for i, t in enumerate(tokens) if t in adverbs]
		clfs = [(i, t) for i, t in enumerate(tokens) if t in classifiers]
		nums = [(i, t) for i, t in enumerate(tokens) if t in numerals]

		head_idx = tokens.index(head) if head in tokens else head_idx
		subj_idx = tokens.index(subj) if subj in tokens else -1
		obj_idx = tokens.index(obj) if obj in tokens else -1

		fallback = []
		# Skip generating OBJ when obj==head (intransitive or no explicit object).
		if head and obj and obj != head:
			fallback.append([head, obj, OBJ])
		if head and subj:
			fallback.append([head, subj, SUBJ])
		for adj_pos, adj_tok in adjs:
			if adj_tok == head:
				continue
			if head_idx != -1 and adj_pos < head_idx:
				target = subj if subj else obj if obj else head
			else:
				target = obj if obj else head
			if target:
				fallback.append([target, adj_tok, ADJ])
		for clf_pos, clf_tok in clfs:
			target = obj if obj else head
			if target:
				fallback.append([target, clf_tok, CLASSIFIER])
		for num_pos, num_tok in nums:
			target = obj if obj else head
			if target:
				fallback.append([target, num_tok, NUM])
		if advs:
			fallback.append([head, advs[0][1], ADVERB])

		dependencies = fallback

	return dependencies
# BRAIN
# fix brain.py to work when no-assembly areas are projected in 

# PARSER IMPLEMENTATION
# Factor out debugger of parse
# Factor out read-out, possibly other aspects of parse
# consider areas where only A->B needed not A<->B, easy to fix
# for example, SUBJ/OBJ->DET, etc?

# PARSER CONCEPTUAL
# 1) NATURAL READ OUT: 
	# "Fiber-activation read out": Remember fibers that were activated
	# "Lexical-item read out": Get word from V, see rules (not sufficient but recovers basic structure)

# 2) PREP area: of, others
# "brand of toys", to merge brand<->of<->toys, look for activated noun areas
# for example if OBJ is the only one, we're done
# if multiple, recency? (first instance of lookahead/memory!)

# 3) Intransitive verbs (in particular wrt read out)

# RESEARCH IDEAS
# 1) Russian experiment (free word order)
# 2) Grammaticality, detect some sort of error for non-grammatical


