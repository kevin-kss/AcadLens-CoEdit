"""公式/引文双向无损掩码器。

必须在分词、计算句长、计算 PPL 和调用大模型之前执行 —— 否则公式里的 `$x_1$`
会被当成正文词汇计入句长，引文角标 `[12]` 会被大模型"顺手改写"。

无损性是硬验收指标（公式与引文还原一致率 100%），因此这里有两条纪律：
1. 占位符格式 `{{KIND_N}}` 中的 N 全局自增，绝不复用，避免还原时前缀冲突；
2. 还原按 Token 长度降序，确保 `{{CITE_1}}` 不会先被 `{{CITE_11}}` 的前缀吃掉。
"""

from __future__ import annotations

import re

from app.schemas.document import MaskedText, MaskRegistry

MASK_TOKEN_RE = re.compile(r"\{\{[A-Z_]+_\d+\}\}")

# 按优先级从长到短匹配：行间公式 -> 行内公式 -> LaTeX 命令 -> 方括号引文
# -> 著者年份引文 -> URL/DOI。顺序即优先级，先匹配者占位，后者不再进入其内部。
MASK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 行间公式：$$...$$、\[...\]、display math 环境
    ("MATH", re.compile(r"\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]")),
    (
        "MATH",
        re.compile(
            r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}"
            r"[\s\S]+?\\end\{\1\}"
        ),
    ),
    # 行内公式：$...$、\(...\)
    ("MATH", re.compile(r"\$[^$\n]+\$|\\\([\s\S]+?\\\)")),
    # LaTeX 引用与标签命令
    ("LATEX_CMD", re.compile(r"\\(?:cite[tp]?|ref|eqref|label|autoref|nameref)\{[^}]+\}")),
    # URL / DOI 先于引文匹配：避免 doi 中的括号数字被引文规则截断
    ("URL", re.compile(r"https?://\S+|doi:\s*10\.\d{4,9}/\S+|\b10\.\d{4,9}/[^\s,;]+")),
    # 方括号数字引文：[12]、[1-3]、[1, 5, 9]
    ("CITE", re.compile(r"\[\s*\d+(?:\s*[-–~,，、]\s*\d+)*\s*\]")),
    # 著者年份引文：(Smith et al., 2024)、(Smith & Lee, 2023a)、（李雷等, 2024）
    # 中文文献常用全角括号（），故两种括号都需匹配，且用前向断言保证左右配对。
    (
        "CITE",
        re.compile(
            r"(?:\((?=[^()]*\))|（(?=[^（）]*）))"
            r"(?:[A-Z][A-Za-z\-']+|[\u4e00-\u9fff]{1,6})"
            r"(?:[\s,，&和与]+(?:et\s+al\.?|[A-Z][A-Za-z\-']+|[\u4e00-\u9fff]{1,6}|等))*"
            r"[,，]?\s*\d{4}[a-z]?[)）]"
        ),
    ),
    # 纯年份补充式引文：Smith (2024) 中的 (2024)、（2024）
    ("CITE", re.compile(r"[(（]\s*\d{4}[a-z]?(?:\s*[-–,，]\s*\d{4}[a-z]?)*\s*[)）]")),
]


def mask_academic_text(text: str, registry: MaskRegistry | None = None) -> MaskedText:
    """把非正文元素替换为不可变占位符。

    registry 为 None 时新建；传入已有 registry 则在其上继续累加编号，
    便于同一文档的多个段落共享一张全局映射表。
    """
    reg = registry if registry is not None else MaskRegistry()
    masked = text
    for kind, pattern in MASK_PATTERNS:
        masked = pattern.sub(lambda m, k=kind: reg.register(k, m.group(0)), masked)  # type: ignore[misc]
    return MaskedText(masked=masked, registry=reg)


def unmask_academic_text(masked_text: str, registry: MaskRegistry) -> str:
    """还原所有占位符。

    按 Token 长度降序替换：`{{CITE_11}}` 必须先于 `{{CITE_1}}` 处理，
    否则前者会被后者的前缀匹配破坏。
    """
    restored = masked_text
    for token, raw in sorted(registry.mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
        restored = restored.replace(token, raw)
    return restored


def strip_mask_tokens(text: str) -> str:
    """移除占位符本身，用于计算真实字数。"""
    return MASK_TOKEN_RE.sub("", text)


def find_mask_tokens(text: str) -> list[str]:
    """列出文本中出现的占位符，供 term_guard 校验频次一致。"""
    return MASK_TOKEN_RE.findall(text)


def verify_roundtrip(original: str) -> bool:
    """自检：掩码后再还原必须与原文逐字节一致。导入流程可开启断言。"""
    result = mask_academic_text(original)
    return unmask_academic_text(result.masked, result.registry) == original
