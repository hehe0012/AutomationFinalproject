#! /usr/bin/python3.9
import brain
import numpy as np
import pptree
import json
import copy
from collections import namedtuple, defaultdict
from enum import Enum
import jieba
import collections

jieba.load_userdict("user_dict2.txt")

# 中文脑区
# 基础脑区
LEX = "LEX"  # 词汇区
DET = "DET"  # 限定词区（中文量词/冠词）
SUBJ = "SUBJ"  # 主语区
OBJ = "OBJ"  # 宾语区
VERB = "VERB"  # 动词区
ADJ = "ADJ"  # 形容词区
ADVERB = "ADVERB"  # 副词区
PREP = "PREP"  # 介词区
PREP_P = "PREP_P"  # 介词短语区
QUANT = "QUANT"  # 量词区（新增）

# 中文核心脑区列表
AREAS = [LEX, DET, SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P, QUANT] # 新增量词区
EXPLICIT_AREAS = [LEX]  # 显式模拟脑区
RECURRENT_AREAS = [SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P, QUANT]  # 递归投影脑区，新增量词区

# 动作
DISINHIBIT = "DISINHIBIT"  # 去抑制
INHIBIT = "INHIBIT"  # 抑制
ACTIVATE_ONLY = "ACTIVATE_ONLY"  # 仅激活
CLEAR_DET = "CLEAR_DET"  # 清空限定词区

# 规则数据结构
AreaRule = namedtuple('AreaRule', ['action', 'area', 'index'])
FiberRule = namedtuple('FiberRule', ['action', 'area1', 'area2', 'index'])
FiringRule = namedtuple('FiringRule', ['action'])
OtherRule = namedtuple('OtherRule', ['action'])


def generic_noun(index):
    """中文名词规则：可作主语/宾语，可被形容词、量词修饰"""
    return {
        "index": index,
        "PRE_RULES": [
            # 允许词汇区向主/宾语区、介词短语区投影
            FiberRule(DISINHIBIT, LEX, SUBJ, 0),
            FiberRule(DISINHIBIT, LEX, OBJ, 0),
            FiberRule(DISINHIBIT, LEX, PREP_P, 0),
            # 允许限定词、形容词、量词向主/宾语区投影
            FiberRule(DISINHIBIT, DET, SUBJ, 0),
            FiberRule(DISINHIBIT, DET, OBJ, 0),
            FiberRule(DISINHIBIT, DET, PREP_P, 0),
            FiberRule(DISINHIBIT, ADJ, SUBJ, 0),
            FiberRule(DISINHIBIT, ADJ, OBJ, 0),
            FiberRule(DISINHIBIT, ADJ, PREP_P, 0),
            FiberRule(DISINHIBIT, QUANT, SUBJ, 0),
            FiberRule(DISINHIBIT, QUANT, OBJ, 0),
            # 动词向宾语区投影（及物动词）
            FiberRule(DISINHIBIT, VERB, OBJ, 0),
            # 介词短语内部投影
            FiberRule(DISINHIBIT, PREP_P, PREP, 0),
            FiberRule(DISINHIBIT, PREP_P, SUBJ, 0),
		    FiberRule(DISINHIBIT, PREP_P, OBJ, 0),
        ],
        "POST_RULES": [
            # 抑制已使用的修饰语通道
            AreaRule(INHIBIT, DET, 0),
            AreaRule(INHIBIT, ADJ, 0),
            AreaRule(INHIBIT, QUANT, 0),
            AreaRule(INHIBIT, PREP_P, 0),
            AreaRule(INHIBIT, PREP, 0),
            # 抑制词汇区向主/宾语区的重复投影
            FiberRule(INHIBIT, LEX, SUBJ, 0),
            FiberRule(INHIBIT, LEX, OBJ, 0),
            FiberRule(INHIBIT, LEX, PREP_P, 0),
            # 抑制修饰语向主/宾语区的重复投影
            FiberRule(INHIBIT, ADJ, SUBJ, 0),
            FiberRule(INHIBIT, ADJ, OBJ, 0),
            FiberRule(INHIBIT, ADJ, PREP_P, 0),
            FiberRule(INHIBIT, DET, SUBJ, 0),
            FiberRule(INHIBIT, DET, OBJ, 0),
            FiberRule(INHIBIT, DET, PREP_P, 0),
            FiberRule(INHIBIT, QUANT, SUBJ, 0),
            FiberRule(INHIBIT, QUANT, OBJ, 0),
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
    """中文及物动词规则：需带宾语，可被副词修饰"""
    return {
        "index": index,
        "PRE_RULES": [
            # 词汇区向动词区投影
            FiberRule(DISINHIBIT, LEX, VERB, 0),
            # 动词向主/宾语区投影
            FiberRule(DISINHIBIT, VERB, SUBJ, 0),
            FiberRule(DISINHIBIT, VERB, OBJ, 0),
            # 副词修饰动词
            FiberRule(DISINHIBIT, VERB, ADVERB, 0),
            AreaRule(DISINHIBIT, ADVERB, 1),
        ],
        "POST_RULES": [
            # 抑制词汇区向动词区的重复投影
            FiberRule(INHIBIT, LEX, VERB, 0),
            # 激活宾语区，抑制主语区（避免重复）
            AreaRule(DISINHIBIT, OBJ, 0),
            AreaRule(INHIBIT, SUBJ, 0),
            AreaRule(INHIBIT, ADVERB, 0),
            # 介词短语可修饰动词
            FiberRule(DISINHIBIT, PREP_P, VERB, 0),
        ]
    }

def generic_intrans_verb(index):
    """中文不及物动词规则：不带宾语，可被副词修饰"""
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
    """中文系动词规则（是、为、显得）：连接主语和表语"""
    return {
        "index": index,
        "PRE_RULES": [
            FiberRule(DISINHIBIT, LEX, VERB, 0),
            FiberRule(DISINHIBIT, VERB, SUBJ, 0),
            FiberRule(DISINHIBIT, VERB, ADJ, 0),  # 系动词+形容词（表语）
            FiberRule(DISINHIBIT, ADJ, VERB, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, VERB, 0),
            AreaRule(DISINHIBIT, OBJ, 0),  # 表语可视为宾语
            AreaRule(INHIBIT, SUBJ, 0),
            FiberRule(DISINHIBIT, ADJ, VERB, 0),
            # FiberRule(INHIBIT, VERB, ADJ, 0)
        ]
    }

def generic_adverb(index):
    """中文副词规则（快速地、非常）：修饰动词/形容词"""
    return {
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, ADVERB, 0),
            FiberRule(DISINHIBIT, LEX, ADVERB, 0),
            # 副词向动词/形容词区投影
            FiberRule(DISINHIBIT, ADVERB, VERB, 0),
            FiberRule(DISINHIBIT, ADVERB, ADJ, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, ADVERB, 0),
            AreaRule(INHIBIT, ADVERB, 1),
            FiberRule(INHIBIT, ADVERB, VERB, 0),
            FiberRule(INHIBIT, ADVERB, ADJ, 0),
        ]
    }

def generic_adjective(index):
    """中文形容词规则（美丽的、愚蠢的）：修饰名词，可作表语"""
    return {
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, ADJ, 0),
            FiberRule(DISINHIBIT, LEX, ADJ, 0),
            # 形容词向主/宾语区投影（修饰名词）
            # FiberRule(DISINHIBIT, ADJ, SUBJ, 0),
            # FiberRule(DISINHIBIT, ADJ, OBJ, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, ADJ, 0),
            FiberRule(INHIBIT, ADJ, SUBJ, 0),
            # FiberRule(INHIBIT, ADJ, OBJ, 0),
            # FiberRule(INHIBIT, VERB, ADJ, 0),
        ]
    }

def generic_determinant(index):
    """中文限定词规则（这、那、所有）：修饰名词"""
    return {
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, DET, 0),
            FiberRule(DISINHIBIT, LEX, DET, 0),
            # 限定词向主/宾语区投影
            FiberRule(DISINHIBIT, DET, SUBJ, 0),
            FiberRule(DISINHIBIT, DET, OBJ, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, DET, 0),
            FiberRule(INHIBIT, DET, SUBJ, 0),
            FiberRule(INHIBIT, DET, OBJ, 0),
        ]
    }

def generic_quantifier(index):
    """中文量词/数量词规则（个、颗、本）"""
    return {
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, QUANT, 0),
            FiberRule(DISINHIBIT, LEX, QUANT, 0),
            # 量词向主/宾语区投影（修饰名词）
            FiberRule(DISINHIBIT, QUANT, SUBJ, 0),
            FiberRule(DISINHIBIT, QUANT, OBJ, 0),
            # 限定词向量词区投影（这+个）
            FiberRule(DISINHIBIT, DET, QUANT, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, QUANT, 0),
            FiberRule(INHIBIT, QUANT, SUBJ, 0),
            FiberRule(INHIBIT, QUANT, OBJ, 0),
            FiberRule(INHIBIT, DET, QUANT, 0),
        ]
    }

def generic_preposition(index):
    """中文介词规则（在、关于、从）：构成介词短语"""
    return {
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, PREP, 0),
            FiberRule(DISINHIBIT, LEX, PREP, 0),
            # 介词向介词短语区投影
            FiberRule(DISINHIBIT, PREP, PREP_P, 0),
            # 介词短语向主/宾语/动词区投影（作修饰语）
            FiberRule(DISINHIBIT, PREP_P, SUBJ, 0),
            FiberRule(DISINHIBIT, PREP_P, OBJ, 0),
            FiberRule(DISINHIBIT, PREP_P, VERB, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, PREP, 0),
            AreaRule(DISINHIBIT, PREP_P, 0),
            FiberRule(INHIBIT, PREP, PREP_P, 0),
        ]
    }

# 中文词汇映射表
# 索引分配：限定词，量词，名词，动词，形容词，副词，介词，系动词
LEXEME_DICT = {
    # 限定词
    "这": generic_determinant(0),
    "那": generic_determinant(1),
    "所有": generic_determinant(2),
    # 量词
    "个": generic_quantifier(3),
    "颗": generic_quantifier(4),
    "一颗": generic_quantifier(5),
    "只": generic_quantifier(6),
    # 名词
    "我": generic_noun(7),
    "你": generic_noun(8),
    "他": generic_noun(9),
    "人类": generic_noun(10),
    "球": generic_noun(11),
    "书": generic_noun(12),
    "猫": generic_noun(13),
    "狗": generic_noun(14),
    # 及物动词
    "踢": generic_trans_verb(15),
    "看": generic_trans_verb(16),
    "喜欢": generic_trans_verb(17),
    "读": generic_trans_verb(18),
    # 不及物动词
    "红温": generic_intrans_verb(19),
    "红温了": generic_intrans_verb(20),
    "跑步": generic_intrans_verb(21),
    "睡觉": generic_intrans_verb(22),
    # 系动词
    "是": generic_copula(23),
    "变得": generic_copula(24),
    "并非": generic_copula(25),
    # 形容词
    "善良": generic_adjective(26),
    #"善良的": generic_adjective(27),
    #"愚蠢": generic_adjective(28),
    "愚蠢的": generic_adjective(27),
    #"聪明": generic_adjective(30),
    "聪明的": generic_adjective(28),
    #"硬邦邦": generic_adjective(32),
    "硬邦邦的": generic_adjective(29),
    "温柔": generic_adjective(30),
    #"温柔的": generic_adjective(35),
    "大度": generic_adjective(31),
    #"大度的": generic_adjective(37),
    # 副词
    "无可奈何地": generic_adverb(38),
    "愤怒地": generic_adverb(39),
    "非常": generic_adverb(40),
    "快速地": generic_adverb(41),
    "真": generic_adverb(42),
    # 介词
    "在": generic_preposition(43),
    "关于": generic_preposition(44),
    "从": generic_preposition(45),
}

# 中文读出规则
# 定义各脑区的依赖关系读取规则
CHINESE_READOUT_RULES = {
    #VERB: [LEX, SUBJ, OBJ, ADVERB, PREP_P, ADJ],  # 动词依赖：主、宾、副、介词短语、形容词
    VERB: [LEX, SUBJ, OBJ, ADVERB, PREP_P],        # 动词依赖：主、宾、副、介词短语
    SUBJ: [LEX, DET, ADJ, QUANT, PREP_P],          # 主语依赖：限定词、形容词、量词、介词短语
    OBJ: [LEX, DET, ADJ, QUANT, PREP_P],           # 宾语依赖：同上
    PREP_P: [LEX, PREP, ADJ, DET, QUANT],          # 介词短语依赖：介词、修饰语
    PREP: [LEX],                                   # 介词仅依赖词汇区
    ADJ: [LEX],                                    # 形容词仅依赖词汇区
    ADVERB: [LEX],                                 # 副词仅依赖词汇区
    DET: [LEX],                                    # 限定词仅依赖词汇区
    QUANT: [LEX, DET],                             # 量词依赖：限定词
    LEX: [],                                       # 词汇区无依赖
}

# 中文解析器大脑类
class ChineseParserBrain(brain.Brain):
    def __init__(self, p, non_LEX_n=10000, non_LEX_k=100, LEX_k=20,
                 default_beta=0.2, LEX_beta=1.0, recurrent_beta=0.05, interarea_beta=0.5, verbose=False):
        """初始化中文解析器大脑"""
        # 初始化父类Brain
        brain.Brain.__init__(self, p, save_size=True, save_winners=False, seed=0)
        self.lexeme_dict = LEXEME_DICT
        self.all_areas = AREAS
        self.recurrent_areas = RECURRENT_AREAS
        self.initial_areas = [LEX, SUBJ, VERB]  # 初始激活脑区
        self.readout_rules = CHINESE_READOUT_RULES
        self.verbose = verbose
        
        # 初始化脑区状态（抑制因子集合）
        self.fiber_states = defaultdict(lambda: defaultdict(set))
        self.area_states = defaultdict(set)
        self.activated_fibers = defaultdict(set)
        self.initialize_states()
        
        # 添加显式词汇区
        LEX_n = len(LEXEME_DICT) * LEX_k
        self.add_explicit_area(LEX, LEX_n, LEX_k, default_beta)
        
        # 添加其他脑区
        self.add_area(SUBJ, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(OBJ, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(VERB, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(ADJ, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(ADVERB, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(PREP, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(PREP_P, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(DET, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(QUANT, non_LEX_n, non_LEX_k, default_beta)
        
        # 自定义可塑性参数
        custom_plasticities = defaultdict(list)
        for area in RECURRENT_AREAS:
            # 词汇区与其他脑区的连接强化
            custom_plasticities[LEX].append((area, LEX_beta))
            custom_plasticities[area].append((LEX, LEX_beta))
            # 脑区自连接强化
            custom_plasticities[area].append((area, recurrent_beta))
            # 脑区间连接强化
            for other_area in RECURRENT_AREAS:
                if other_area != area:
                    custom_plasticities[area].append((other_area, interarea_beta))
        self.update_plasticities(area_update_map=custom_plasticities)

    def initialize_states(self):
        """初始化脑区和纤维的抑制状态"""
        for from_area in self.all_areas:
            for to_area in self.all_areas:
                self.fiber_states[from_area][to_area].add(0)  # 初始抑制因子0
        for area in self.all_areas:
            self.area_states[area].add(0)
        # 初始激活脑区解除抑制
        for area in self.initial_areas:
            self.area_states[area].discard(0)

    def applyFiberRule(self, rule):
        """应用纤维规则（抑制/去抑制）"""
        if rule.action == INHIBIT:
            self.fiber_states[rule.area1][rule.area2].add(rule.index)
            self.fiber_states[rule.area2][rule.area1].add(rule.index)
        elif rule.action == DISINHIBIT:
            self.fiber_states[rule.area1][rule.area2].discard(rule.index)
            self.fiber_states[rule.area2][rule.area1].discard(rule.index)

    def applyAreaRule(self, rule):
        """应用脑区规则（抑制/去抑制）"""
        if rule.action == INHIBIT:
            self.area_states[rule.area].add(rule.index)
        elif rule.action == DISINHIBIT:
            self.area_states[rule.area].discard(rule.index)

    def applyRule(self, rule):
        """统一应用规则"""
        if isinstance(rule, FiberRule):
            self.applyFiberRule(rule)
            return True
        if isinstance(rule, AreaRule):
            self.applyAreaRule(rule)
            return True
        return False

    def activateWord(self, word):
        """激活词汇区中指定单词的神经集"""
        if word not in self.lexeme_dict:
            raise ValueError(f"词汇 {word} 未在词汇表中定义")
        lexeme = self.lexeme_dict[word]
        area = self.area_by_name[LEX]
        k = area.k
        assembly_start = lexeme["index"] * k
        area.winners = list(range(assembly_start, assembly_start + k))
        area.fix_assembly()
        if self.verbose:
            print(f"激活词汇 {word}，神经集范围：[{assembly_start}, {assembly_start + k - 1}]")

    def getProjectMap(self):
        """生成投影映射"""
        proj_map = defaultdict(set)
        for area1 in self.all_areas:
            # 脑区未被抑制且有活跃神经集
            if len(self.area_states[area1]) == 0 and self.area_by_name[area1].winners:
                for area2 in self.all_areas:
                    if area1 == area2:
                        continue
                    # 目标脑区未被抑制，且纤维未被抑制
                    if len(self.area_states[area2]) == 0 and len(self.fiber_states[area1][area2]) == 0:
                        proj_map[area1].add(area2)
        return proj_map

    def parse_project(self):
        """生成投影映射并执行投影"""
        project_map = self.getProjectMap()
        self.remember_fibers(project_map)
        self.project({}, project_map)

    def remember_fibers(self, project_map):
        """记录激活的纤维（用于读出）"""
        for from_area, to_areas in project_map.items():
            self.activated_fibers[from_area].update(to_areas)

    def getWord(self, area_name, min_overlap=0.7):
        """从脑区神经集映射到单词（基于重叠度）"""
        area = self.area_by_name[area_name]
        if not area.winners:
            return "<无>"
        winners = set(area.winners)
        # print(winners)
        area_k = area.k
        threshold = min_overlap * area_k

        # 遍历词汇表，找到重叠度最高的单词
        for word, lexeme in self.lexeme_dict.items():
            word_index = lexeme["index"]
            word_assembly_start = word_index * area_k
            word_assembly = set(range(word_assembly_start, word_assembly_start + area_k))
            overlap = len(winners & word_assembly)
            if overlap >= threshold:
                return word
        return "<未知词汇>"

    def read_out(self, root_area, dependencies):
        """递归读出依赖关系（深度优先遍历）"""
        to_areas = self.readout_rules[root_area]
        # 根脑区向词汇区投影，获取根单词
        self.project({}, {root_area: [LEX]})
        root_word = self.getWord(LEX)
        if self.verbose:
            print(f"读出根脑区 {root_area} 对应单词：{root_word}")
        
        # 根脑区向目标脑区投影
        self.project({}, {root_area: to_areas})
        for to_area in to_areas:
            # 目标脑区向词汇区投影，获取目标单词
            self.project({}, {to_area: [LEX]})
            target_word = self.getWord(LEX)
            # print(root_area, to_area, target_word)
            if target_word not in ["<无>", "<未知词汇>"]:
                dependencies.append([root_word, target_word, to_area])
                # 递归读出目标脑区的依赖
                self.read_out(to_area, dependencies)
                
    def read_out2(self, dependencies_verb_adj):
        self.project({}, {VERB: [LEX]})
        root_word_verb = self.getWord(LEX)
        if self.verbose:
            print(f"读出根脑区 {VERB} 对应单词：{root_word_verb}")
        self.project({}, {VERB: [ADJ]})
        self.project({}, {ADJ: [LEX]})
        target_word_adj1 = self.getWord(LEX)
        # print(VERB, ADJ, target_word_adj1)
        if target_word_adj1 not in ["<无>", "<未知词汇>"]:
            target_word_adj2 = self.read_out3()
            if target_word_adj1 != target_word_adj2:
                dependencies_verb_adj.append([root_word_verb, target_word_adj1, ADJ])
                
    def read_out3(self):
        try:
            self.project({}, {OBJ: [ADJ]})
            target_word_adj2 = self.getWord(LEX)
            return target_word_adj2
        except:
            return "<无>"
    
        

# 解析器调试类
class ParserDebugger():
    def __init__(self, brain, all_areas, explicit_areas):
        self.b = brain
        self.all_areas = all_areas
        self.explicit_areas = explicit_areas

    def run(self):
        command = input("DEBUGGER: 按回车继续，输入'P'查看详细信息\n")
        while command:
            if command == "P":
                self.peak()
                return
            else:
                print("DEBUGGER: 未知命令")
                command = input("DEBUGGER: 按回车继续，输入'P'查看详细信息\n")
        return

    def peak(self):
        # 临时禁用可塑性和保存神经集
        self.b.disable_plasticity = True
        self.b.save_winners = True
        for area in self.all_areas:
            self.b.area_by_name[area].unfix_assembly()

        while True:
            test_proj_map_str = input("DEBUGGER: 输入投影映射（如{\"VERB\":[\"LEX\"]}），回车退出\n")
            if not test_proj_map_str:
                break
            try:
                test_proj_map = json.loads(test_proj_map_str)
            except json.JSONDecodeError:
                print("DEBUGGER: 格式错误，请重新输入")
                continue

            # 保存当前神经集
            to_areas = set()
            for _, tas in test_proj_map.items():
                to_areas.update(tas)
            for ta in to_areas:
                if not self.b.area_by_name[ta].saved_winners:
                    self.b.area_by_name[ta].saved_winners.append(self.b.area_by_name[ta].winners)

            # 执行测试投影
            self.b.project({}, test_proj_map)

            # 打印显式脑区结果
            for area in self.explicit_areas:
                if area in to_areas:
                    word = self.b.getWord(area)
                    print(f"DEBUGGER: 显式脑区 {area} 对应单词：{word}")

            # 打印指定脑区神经集
            print_areas = input("DEBUGGER: 输入要打印的脑区（如LEX,VERB），回车继续\n")
            if print_areas:
                for area in print_areas.split(","):
                    if area in self.b.area_by_name:
                        print(f"DEBUGGER: 脑区 {area} 神经集：{self.b.area_by_name[area].winners}")

        # 恢复状态
        self.b.disable_plasticity = False
        self.b.save_winners = False
        for area in self.all_areas:
            self.b.area_by_name[area].saved_winners = []
            
def tokenize_chinese(sentence):
    """使用jieba对中文句子进行分词，返回合并“的/地/了”后的分词列表"""
    raw_tokens = list(jieba.cut(sentence))
    merged_tokens = []
    i = 0
    while i < len(raw_tokens):
        if raw_tokens[i] in ["的", "地", "了"] and i > 0:
            merged_tokens[-1] += raw_tokens[i]
        else:
            merged_tokens.append(raw_tokens[i])
        i += 1
    return merged_tokens

# 中文句子解析函数
def chinese_parse(sentence="我踢硬邦邦的球", project_rounds=20, verbose=True, debug=False):
    """中文句子解析主函数"""
    # 中文分词
    words = tokenize_chinese(sentence)
    if verbose:
        print(f"分词结果：{words}")

    # 初始化中文解析器大脑
    brain = ChineseParserBrain(p=0.1, verbose=verbose)
    debugger = ParserDebugger(brain, AREAS, EXPLICIT_AREAS)

    # 逐词处理：激活词汇+应用规则+投影
    for idx, word in enumerate(words):
        if word not in LEXEME_DICT:
            print(f"警告：词汇 {word} 未定义，跳过")
            continue
        
        # 激活词汇
        brain.activateWord(word)
        
        # 应用PRE_RULES
        lexeme = LEXEME_DICT[word]
        for rule in lexeme["PRE_RULES"]:
            brain.applyRule(rule)
        if verbose:
            print(f"单词 {word} 应用PRE_RULES完成")
        
        # 生成投影映射
        proj_map = brain.getProjectMap()
        if verbose:
            print(f"投影映射：{dict(proj_map)}")
        
        # 固定/清空脑区神经集
        for area in proj_map:
            if area != LEX and LEX not in proj_map[area]:
                brain.area_by_name[area].fix_assembly()
                if verbose:
                    print(f"固定脑区 {area} 神经集")
            elif area != LEX:
                brain.area_by_name[area].unfix_assembly()
                brain.area_by_name[area].winners = []
                if verbose:
                    print(f"清空脑区 {area} 神经集")
        
        # 多轮投影（迭代收敛）
        for i in range(project_rounds):
            brain.parse_project()
            if verbose and (i + 1) % 5 == 0:
                print(f"第 {i+1} 轮投影完成")
            if idx == len(words) - 1:
                brain.parse_project()
                if verbose and (i + 1) % 5 == 0:
                    print(f"第 {i+1} 轮投影完成")
        
        # 应用POST_RULES
        for rule in lexeme["POST_RULES"]:
            brain.applyRule(rule)
        if verbose:
            print(f"单词 {word} 应用POST_RULES完成\n")
        
        # 调试模式
        if debug:
            debugger.run()

    # 读出依赖关系
    brain.disable_plasticity = True
    for area in AREAS:
        brain.area_by_name[area].unfix_assembly()
        
    dependencies_verb_adj = []
    brain.read_out2(dependencies_verb_adj)
    
    dependencies = []
    brain.read_out(VERB, dependencies)  # 以动词为根读出
    
    dependencies += dependencies_verb_adj

    # 输出结果
    # print("\n" + "="*50)
    print("="*50)
    print(f"输入句子：{sentence}")
    print(f"分词结果：{words}")
    print("依赖解析结果：")
    print(dependencies)
    print("="*50)

# 测试用例
def test_all_cases():
    """测试所有要求的中文句式"""
    test_cases = [
        # 1. 主语 + （副词） + 不及物动词
        "我红温了",
        "我无可奈何地红温了",
        # 2. 主语 + 表语/及物动词 + 宾语
        "我并非人类",
        "我踢球",
        # 3. 主语 + 表语 + 形容词
        "你变得善良",
        # 4. 形容词 + 主语 + 表语/及物动词 + 宾语
        "愚蠢的我并非人类",
        "愚蠢的我踢球",
        # 5. 主语 + 表语/及物动词 + 形容词 + 宾语
        "我并非愚蠢的人类",
        "我踢硬邦邦的球",
        # 6. 形容词 + 主语 + 表语/及物动词 + 形容词 + 宾语
        "聪明的我并非愚蠢的人类",
        "愚蠢的我踢硬邦邦的球",
        # 7. （形容词） + 主语 + （副词） + 及物动词 + （量词）+ （形容词） + 宾语
        "愚蠢的我愤怒地踢一颗硬邦邦的球",
        # 加分项：主语 + 表语 + 连续多个形容词
        "你变得温柔善良大度"
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n【测试用例 {i}】")
        chinese_parse(case, verbose=False, debug=False)
    
    #chinese_parse("你变得善良", verbose=True, debug=False)
    #chinese_parse("我并非愚蠢的人类", verbose=True, debug=False)
    
    
    
if __name__ == "__main__":
    # 运行所有测试用例
    test_all_cases()