# Architectural Paradigms and Forensic Methodologies for AI-Generated Text Detection

The proliferation of frontier large language models (LLMs) has transformed automated text authorship attribution from a binary classification problem into a complex forensic challenge. Early detection systems relied on surface-level token predictability, operating on the assumption that machine-generated prose uniformly selects high-probability tokens while human writing exhibits high entropy. As modern foundation models incorporate multi-stage post-training, reinforcement learning from human feedback (RLHF), dynamic temperature scaling, and collaborative human-AI editing workflows, naive probability thresholding has collapsed in reliability.

Building a reliable Model Context Protocol (MCP) server capable of evaluating text on a calibrated 0-to-100 scale—and accompanying that numerical judgment with verifiable, human-interpretable explanations—requires synthesizing three distinct paradigms of computational linguistics: zero-shot token probability curvature, supervised sequence-and-token classification, and forensic stylometry. While commercial systems such as Pangram Labs achieve high laboratory accuracy through massive-scale supervised active learning, they operate largely as black-box classifiers that output span probabilities rather than explicit linguistic explanations. A custom MCP architecture can bridge this gap by pairing orthogonal neural scoring engines with a deterministic syntactic and stylometric feature extractor that surfaces concrete evidence for every classification decision.

---

## Deconstructing Commercial State-of-the-Art: How Pangram Labs Operates

To understand what makes a modern commercial detector effective—and how to replicate or complement its capabilities in a self-hosted MCP server—it is necessary to examine the architecture of Pangram Labs, specifically its Pangram 4 release from July 2026. Founded in 2023 by Max Spero and Bradley Emi (originally launched as Checkfor.ai before rebranding in April 2024), Pangram explicitly rejects zero-shot perplexity thresholding in favor of a deeply supervised, multi-task Mixture-of-Experts (MoE) neural architecture trained on paired synthetic datasets.

### Synthetic Mirroring and Hard-Negative Active Learning

A foundational failure mode of early supervised AI detectors—such as OpenAI's original RoBERTa detector—was spurious correlation with topic, domain, and length. If a training corpus contains human-written pre-2022 journalism alongside AI-generated technical summaries, a neural classifier inevitably learns to flag technical topics rather than machine syntax. Pangram resolves this through a data generation methodology termed **synthetic mirroring**, originally introduced by Emi and Spero (2024) and expanded in Pangram 4. For every human-written document in the training corpus, the pipeline generates matched synthetic counterparts controlled for exact topic, structural format, length, and rhetorical tone across a diverse array of frontier models. Furthermore, to capture collaborative workflows, the human source texts undergo controlled transformations—including single-pass and multi-pass AI polishing, style transfer, translation, and adversarial paraphrasing ("humanizer" attacks)—to populate intermediate provenance classes.

To drive down the false positive rate (FPR)—the most damaging error mode in academic and publishing deployments—Pangram employs an iterative **offline active learning loop** focused on hard-negative mining. After training an initial candidate checkpoint, the system runs inference across a massive reserved corpus of verified human writing. Any human text that the candidate checkpoint erroneously flags as AI-generated is isolated as a "hard negative". Rather than simply re-weighting these human texts (which would shift the decision boundary and inflate false negatives), the pipeline generates new, synthetically mirrored AI counterparts specifically matched to those misclassified human texts and injects the paired examples back into the training set. This forces the neural network to discard superficial stylistic shortcuts and isolate deeper distributional boundaries between human and machine prose.

### Multi-Head Mixture-of-Experts and the EditLens Continuum

Architecturally, Pangram 4 builds upon **EditLens** (Thai et al., ICLR 2026), which reformulated AI detection from a rigid binary decision into a continuous regression problem over mixed-authorship text. EditLens established a vital distinction between *heterogeneous* mixed text, where discrete paragraphs are copied and pasted from an LLM into a human document to allow clear boundary attribution, and *homogeneous* mixed text, where a human drafts an original argument and an LLM rewrites or polishes the prose, entangling human ideation with machine lexical choices at every token. During training data preparation, target texts are segmented into fine-grained clauses using Claude Haiku 4.5—which proved substantially less brittle on syntactic edge cases than classical dependency parsers—and evaluated using phrase-level pairwise cosine similarity against source drafts to quantify the exact magnitude of AI editing.

To process these signals at scale, Pangram 4 uses an open-weight Mixture-of-Experts (MoE) causal language model as its shared backbone. Because standard causal (decoder-only) language models employ unidirectional causal attention masks—meaning token $t_i$ can only attend to preceding tokens $t_{<i}$—they are naturally suboptimal for token-level classification where the provenance of a word at the start of a sentence depends on clauses at the end of the paragraph. Pangram 4 overcomes this architectural limitation by adopting the **Repeat2** prompt repetition technique (Leviathan et al., 2025), which duplicates the input sequence context so that tokens in the second pass attend to the full bidirectional document context within a causal MoE backbone.

Documents of at least 50 words are segmented into overlapping sliding windows of $S = 512$ tokens. At the final hidden layer of the MoE backbone ($H \in \mathbb{R}^{S \times D}$), three distinct linear classification heads operate simultaneously on each window $w$:

1. **Segment-Level Bucketed Regression Head ($\mathbf{p}_w^s \in \mathbb{R}^{15}$):** Computes a probability distribution over 15 ordered buckets representing the continuous **weighted AI fraction** ($f_{AI} \in [0, 1]$) of the 512-token segment:
$$f_{AI} = \frac{0.5 C_{AA} + C_{AG}}{C_H + C_{AA} + C_{AG}}$$
where $C_H$, $C_{AA}$, and $C_{AG}$ denote the character counts within the segment attributed to human writing, AI-assisted editing, and fully AI-generated prose, respectively.
2. **Ternary Tokenwise Classification Head ($\mathbf{z}_{w,i}^t \in \mathbb{R}^3$):** Projects the hidden state of every individual token $i$ in window $w$ into a three-class logit vector corresponding to `{human, AI-assisted, AI-generated}` provenance.
3. **Binary Mixed-Authorship Auxiliary Head ($q_w \in [0, 1]$):** Predicts `True` if more than 15% of the supervised tokens within the 512-token window belong to provenance classes other than the dominant class, and `False` if the window exhibits homogeneous authorship.

| Architectural & Benchmark Dimension | Pangram 4 Specification / Empirical Result | Operational Significance |
| :--- | :--- | :--- |
| **Backbone Architecture** | Open-weight Mixture-of-Experts (MoE) Causal LM with `Repeat2` context | Enables bidirectional context awareness for token-level tagging while leveraging pretrained LLM representations. |
| **Minimum Input Length** | 50 words (sliding windows of 512 tokens) | Below 50 words, statistical variance causes rapid degradation in attribution reliability. |
| **Overall Laboratory Benchmark** | AUROC: 0.9916 \| FPR: 0.0041% (~1 in 24,300) \| FNR: 0.3396% | Demonstrates near-zero false accusations on clean, in-distribution English benchmarks. |
| **Frontier Model Accuracy (July 2026)** | 99.7% macro-average across 26 models (Claude Opus 5: 99.8%, GPT 5.6: 99.6%, Gemini 3: 99.5%, Grok 4.3: 99.4%) | Requires continuous monthly retraining as new model checkpoints alter token distributions. |
| **Independent Field Evaluation** | 94.2% overall accuracy \| 3.8% FPR \| 87% accuracy on short texts (<200 words) | Illustrates the laboratory-to-deployment gap when encountering noisy, short, or out-of-domain real-world prose. |
| **Explainability Output** | Highlighted character spans (`Human`, `AI-Assisted`, `AI-Generated`) and uncalibrated confidence labels | Lacks explicit linguistic or diagnostic explanations detailing *why* a passage was flagged. |

---

## Taxonomy of Detection Methodologies: Mathematical Foundations and Mechanics

### 1. Zero-Shot Statistical and White-Box Probability Methods

Zero-shot statistical detectors require no labeled training data from the target generator. Instead, they pass the candidate text through a surrogate open-weight language model (such as `Falcon-7B`, `GPT-Neo`, or `Qwen`) and analyze the next-token log-probability distributions.

#### Log-Perplexity and the Capybara Problem
Given a token sequence $X = (x_1, x_2, \dots, x_T)$, a causal language model $M$ with parameters $\theta$ outputs a conditional probability distribution $p_\theta(x_i | x_{<i})$ over the vocabulary $V$ at each step $i$. Log-perplexity measures the average negative log-likelihood of the sequence:
$$\log \text{PPL}(M, X) = -\frac{1}{T} \sum_{i=1}^{T} \log p_\theta(x_i | x_{<i})$$
Hans et al. (ICML 2024) exposed a critical flaw in raw perplexity thresholding known as the **Capybara Problem**: when an LLM is prompted with an unusual, surreal, or highly specific premise (e.g., *"Write an essay about Dr. Capy Cosmos, a capybara astrophysicist"*), the resulting tokens exhibit high perplexity (high surprise) to any observer model that lacks the original prompt. Consequently, naive perplexity detectors misclassify prompt-eccentric AI text as human while misclassifying predictable or memorized human text as machine-generated.

#### Conditional Probability Curvature: Fast-DetectGPT
**Fast-DetectGPT** (Bao et al., ICLR 2024) evaluates whether a text sits at a local maximum (negative curvature) of log-probability by holding the preceding context $x_{<j}$ fixed at each token position $j$ and analytically evaluating the distribution of all possible alternative tokens $\tilde{x}_j \sim \tilde{p}_\theta(\cdot | x_{<j})$ directly from the model's vocabulary logits in a single forward pass:
$$d(x, p_\theta, \tilde{p}_\theta) = \frac{\log p_\theta(x) - \tilde{\mu}}{\tilde{\sigma}}$$
where the expected log-probability $\tilde{\mu}$ and variance $\tilde{\sigma}^2$ are summed across all token positions $j \in [1, T]$ analytically over the vocabulary $V$:
$$\tilde{\mu} = \sum_{j=1}^{T} \sum_{v \in V} \tilde{p}_\theta(v | x_{<j}) \log p_\theta(v | x_{<j}), \quad \tilde{\sigma}^2 = \sum_{j=1}^{T} \left( \sum_{v \in V} \tilde{p}_\theta(v | x_{<j}) \left( \log p_\theta(v | x_{<j}) \right)^2 - \tilde{\mu}_j^2 \right)$$
Fast-DetectGPT runs 340 times faster than perturbation-based DetectGPT while improving relative detection accuracy by roughly 75%.

#### Dual-Model Cross-Perplexity: Binoculars
**Binoculars** (Hans et al., ICML 2024) normalizes prompt-induced perplexity by contrasting two closely related language models that share a tokenizer: an observer model $M_1$ (e.g., `Falcon-7B` or a base model) and a performer model $M_2$ (e.g., `Falcon-7B-Instruct` or its instruction-tuned twin). Binoculars computes the ratio of self-log-perplexity to **log cross-perplexity**:
$$B_{M_1, M_2}(s) = \frac{\log \text{PPL}_{M_1}(s)}{\log \text{X-PPL}_{M_1, M_2}(s)}$$
where the log cross-perplexity $\log \text{X-PPL}_{M_1, M_2}(s)$ is the average per-token cross-entropy between the full vocabulary probability distribution predicted by $M_1$ and the log-probabilities assigned by $M_2$:
$$\log \text{X-PPL}_{M_1, M_2}(s) = -\frac{1}{L} \sum_{i=1}^{L} \sum_{j \in V} M_1(j | s_{<i}) \cdot \log M_2(j | s_{<i})$$
Lower Binoculars ratios $B$ indicate AI generation, whereas higher ratios indicate human authorship. Without training on ChatGPT data, Binoculars detects over 90% of ChatGPT generations at a false positive rate of 0.01%.

### 2. Supervised Neural Classifiers and Continuous Edit Regression

Supervised classifiers fine-tune bidirectional encoders (`RoBERTa-base`, `RoBERTa-large`, `DeBERTa-v3`, or `Longformer`) or instruction-tuned backbones (`RADAR`, `EditLens`) on labeled corpora. Unlike zero-shot logit calculators, which inspect one-step conditional token probabilities, supervised classifiers attend to holistic phrase structure, syntactic transitions, and semantic coherence across an entire 512-token window, making them resilient against word-level synonym substitution attacks.

### 3. Forensic Stylometry, Lexical Constancy, and Structural Heuristics

* **Sentence-Length Burstiness ($B_{\text{len}}$):** Defined as the coefficient of variation of sentence word counts across a document containing $S$ sentences:
$$B_{\text{len}} = \frac{\sigma_L}{\mu_L}$$
Default outputs from Gemini average 0.15–0.22, ChatGPT-4o averages 0.18–0.25, and Claude averages 0.20–0.30, whereas natural human writing averages 0.65–0.85+. A burstiness score below 0.30 is a strong indicator of unedited AI output.
* **Yule's Characteristic $K$:** A length-invariant measure of lexical repetition (higher $K$ means more repetitive vocabulary):
$$K = 10^4 \times \frac{\left(\sum_{m=1}^{m_{\max}} m^2 V(m, N)\right) - N}{N^2}$$
* **Measure of Textual Lexical Diversity (MTLD):** Evaluates the mean length of sequential token strings that maintain a running Type-Token Ratio (TTR) above 0.72 across forward and backward passes (higher MTLD means richer vocabulary).
* **Persistent 2026 Syntactic Artifacts:** While famous single words like *delve* and *tapestry* were largely scrubbed by 2026 (<0.06% of LinkedIn posts), structural habits persist: **negative parallelisms** (*"It's not just X, it's Y"*), **trailing present-participle clauses** (*", highlighting..."*, *", underscoring..."*, *", ensuring..."*), **mechanical signpost connectors** (*Furthermore,*, *Moreover,*, *Additionally,*), elevated **em-dash density**, and clustered **corporate Latinate verbs** (*leverage*, *utilize*, *facilitate*, *streamline*, *foster*, *empower*).

---

## Empirical Evaluation: RAID Benchmark and Adversarial Robustness

| Detection Paradigm | Representative Methods | Clean Text Performance | Paraphrase / Polish Resilience | Synonym-Swap Resilience | Short-Text Stability (<100w) | Native Explainability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dual-Model Cross-Perplexity** | `Binoculars` (Falcon-7B / 7B-Instruct, Qwen pairs) | **Very High** (>90% TPR @ 0.01% FPR; 0.74 TPR @ 1% FPR on RAID) | **High** (0.935 AUROC; 0.68 TPR @ 1% FPR) | **Low** (Drops to 0.30 TPR @ 1% FPR; -0.181 AUROC) | **High** (Stable down to 50 words) | **Low** (Outputs a single scalar ratio, plus token surprises) |
| **Conditional Probability Curvature** | `Fast-DetectGPT`, `Glimpse` | **High** (0.95–0.99 AUROC in-domain; 0.69 TPR @ 1% FPR on RAID) | **Moderate-High** (0.913 AUROC; 0.59 TPR @ 1% FPR) | **Low** (Drops to 0.22 TPR @ 1% FPR; -0.245 AUROC) | **Moderate** (Improved via Short-PHD extensions) | **Moderate** (Can highlight tokens sitting at local probability peaks) |
| **Supervised Multi-Head / Edit Regression** | `Pangram 4`, `EditLens`, `RoBERTa-large`, `DeBERTa` | **Very High In-Domain** (0.9916 AUROC for Pangram 4) | **Very High** (Trained explicitly on AI-polished and humanized spans) | **High** (Only -0.028 AUROC loss under synonym attack) | **Low-Moderate** (Requires >=50 words; RADAR has 61% FPR at 50w) | **Low-Moderate** (Provides character-aligned span labels, not linguistic reasons) |
| **Forensic Stylometry & Structural Heuristics** | `Burstiness`, `Yule's K`, `MTLD`, Syntactic Regexes | **Moderate** (0.84–0.90 on H3C+ stylometric baselines) | **Low-Moderate** (Vulnerable if sentence lengths are manually split) | **Very High** (Word swaps do not alter sentence lengths or clause syntax) | **Low** (Requires >=5 sentences for stable variance) | **Very High** (Directly generates concrete, verifiable linguistic reasons) |
| **Direct Zero-Shot LLM-as-a-Judge** | Prompting `Claude Opus 4.6`, `GPT 5.5`, `Qwen 3.6` directly | **Low-Moderate** (50% F1 to 77.9% accuracy) | **Low** (Easily fooled by tone shifts and rubric artifacts) | **Moderate** | **Low** | **High** (Fluent text explanations, but prone to hallucinating rationales) |

### Critical Adversarial Findings (RAID & DetectRL)
1. **Orthogonal Vulnerability to Synonym Swaps:** Word-level synonym substitution drops `Binoculars` from 0.74 to 0.30 TPR@1%FPR and `Fast-DetectGPT` from 0.69 to 0.22, whereas supervised `RoBERTa` and `RADAR` lose almost no accuracy (-0.028 and -0.008 AUROC). Combining zero-shot and supervised tiers closes this gap.
2. **The Two-Sided Adversarial Paradox & Unicode Sanitization:** When character-level perturbations (zero-width spaces or homoglyphs) are present in *human* text, stronger zero-shot backbones like `Falcon-7B` falsely accuse up to 100% of innocent human writers. Stripping zero-width characters and applying `NFKC` + homoglyph normalization before scoring returns the false positive rate to 1.0%.
3. **Non-Native Speaker (NNS) Guardrail:** Perplexity detectors frequently misclassify L2/TOEFL writing due to simpler vocabulary. Dampening the AI score when sentence-length burstiness is naturally high (>0.55) and zero LLM syntactic tacks occur protects against NNS false positives.

---

## Complete Production Implementation Using FastMCP (`ai_detector_mcp.py`)

```python
# /// script
# dependencies = [
#   "fastmcp>=2.0.0",
#   "pydantic>=2.7.0",
#   "torch>=2.2.0",
#   "transformers>=4.44.0",
#   "numpy>=1.26.0",
# ]
# ///

"""
Forensic AI Text Detection MCP Server.
Implements a Tri-Tier Ensemble combining:
1. Zero-Shot Cross-Perplexity (Binoculars) & Conditional Probability Curvature (Fast-DetectGPT)
2. Supervised Sliding-Window Sequence Classification (EditLens / RoBERTa)
3. Forensic Stylometry (Burstiness, Yule's K, MTLD) & Syntactic Artifact Scanning
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Optional PyTorch / HuggingFace imports with graceful availability flag
try:
    import torch
    import torch.nn.functional as F
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# 1. STRUCTURED OUTPUT SCHEMAS (PYDANTIC MODELS FOR FASTMCP)
# ============================================================================

class DiagnosticReason(BaseModel):
    category: Literal[
        "Token Probability & Curvature",
        "Supervised Neural Signature",
        "Sentence Cadence & Burstiness",
        "Lexical Diversity & Repetition",
        "Syntactic & Rhetorical Artifacts",
        "Adversarial Perturbation & Sanitization",
        "Calibration & Guardrail Notice",
    ] = Field(description="The forensic category of the detection signal.")
    direction: Literal["AI_INDICATOR", "HUMAN_INDICATOR", "WARNING"] = Field(
        description="Whether this specific finding pushes the score toward AI, Human, or acts as a reliability warning."
    )
    impact_weight: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="The relative contribution of this reason to the overall 0-100 score."
    )
    metric_name: str = Field(description="Name of the quantitative metric or linguistic pattern.")
    observed_value: str = Field(description="Measured value in the candidate text.")
    reference_benchmark: str = Field(description="Expected baseline range for human vs. AI writing.")
    explanation: str = Field(
        description="Detailed, human-readable forensic explanation with specific examples or quotes from the text."
    )


class SpanAttribution(BaseModel):
    sentence_index: int = Field(description="Zero-based index of the sentence in the document.")
    text_excerpt: str = Field(description="The sentence text.")
    ai_likelihood_score: int = Field(
        ge=0, le=100, description="Estimated 0-100 AI likelihood for this specific sentence."
    )
    classification: Literal["Human", "AI-Assisted", "AI-Generated"] = Field(
        description="Fine-grained ternary classification aligned with EditLens/Pangram provenance tiers."
    )


class DetectionReport(BaseModel):
    ai_generation_score: int = Field(
        ge=0,
        le=100,
        description="Calibrated 0-100 opinion score where 0 is definitely human-written and 100 is definitely AI-generated.",
    )
    provenance_classification: Literal[
        "Human-Written (0-24)",
        "Lightly AI-Assisted / Polished (25-54)",
        "Moderately to Heavily AI-Edited (55-79)",
        "AI-Generated (80-100)",
    ] = Field(description="Categorical authorship classification across the human-AI continuum.")
    confidence_level: Literal["HIGH", "MODERATE", "LOW (Short Text < 50 Words)"] = Field(
        description="Reliability confidence of the assessment based on document length and signal agreement."
    )
    reasons: List[DiagnosticReason] = Field(
        description="Prioritized list of specific quantitative and linguistic reasons explaining why the text received its score."
    )
    sub_scores: Dict[str, float] = Field(
        description="Component sub-scores (0-100) across Zero-Shot Probability, Supervised Classifier, and Stylometry."
    )
    quantitative_metrics: Dict[str, float] = Field(
        description="Raw forensic measurements including Binoculars ratio, Fast-DetectGPT curvature, Burstiness, Yule's K, and MTLD."
    )
    flagged_sentences: List[SpanAttribution] = Field(
        description="Sentence-level provenance breakdown highlighting the most machine-like passages."
    )


# ============================================================================
# 2. STAGE 0: ADVERSARIAL UNICODE SANITIZER & PRE-FLIGHT GATE
# ============================================================================

ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]")

HOMOGLYPH_MAP = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "і": "i", "І": "I", "ј": "j",
})


@dataclass
class SanitizationResult:
    clean_text: str
    zero_width_count: int
    homoglyph_count: int
    em_dash_count: int
    curly_quote_count: int


def sanitize_and_profile_typography(raw_text: str) -> SanitizationResult:
    """Captures raw typographic tells and strips adversarial character perturbations."""
    em_dashes = raw_text.count("—") + raw_text.count(" -- ")
    curly_quotes = sum(raw_text.count(q) for q in ("“", "”", "‘", "’"))

    zw_matches = ZERO_WIDTH_PATTERN.findall(raw_text)
    text_no_zw = ZERO_WIDTH_PATTERN.sub("", raw_text)

    homoglyph_hits = sum(1 for ch in text_no_zw if ch in HOMOGLYPH_MAP)
    text_dehomoglyphed = text_no_zw.translate(HOMOGLYPH_MAP)

    clean_text = unicodedata.normalize("NFKC", text_dehomoglyphed)
    clean_text = re.sub(r"[ \t]+", " ", clean_text).strip()

    return SanitizationResult(
        clean_text=clean_text,
        zero_width_count=len(zw_matches),
        homoglyph_count=homoglyph_hits,
        em_dash_count=em_dashes,
        curly_quote_count=curly_quotes,
    )


# ============================================================================
# 3. TIER 3: FORENSIC STYLOMETRY & SYNTACTIC ARTIFACT ENGINE
# ============================================================================

NEGATIVE_PARALLELISM_PATTERNS = [
    re.compile(r"\b(?:it['’]?s|this is|that['’]?s)\s+not\s+(?:just|merely|simply|only)\b[^.!?]+?\b(?:it['’]?s|but\s+rather|it\s+is)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+only\s+(?:does|is|do|can|will|did)\b[^.!?]+?\bbut\s+(?:it\s+)?also\b", re.IGNORECASE),
]

TRAILING_PARTICIPLE_PATTERN = re.compile(
    r",\s+(?:highlighting|underscoring|emphasizing|ensuring|fostering|showcasing|demonstrating|"
    r"paving\s+the\s+way|reflecting|contributing\s+to|driving|enabling|ultimately\s+\w+ing)\b[^.!?]*[.!?]",
    re.IGNORECASE,
)

SIGNPOST_CONNECTOR_PATTERN = re.compile(
    r"(?:^|\n\s*|[.!?]\s+)(?:Furthermore|Moreover|Additionally|Consequently|Notably|"
    r"Importantly|In\s+summary|In\s+conclusion|Ultimately|To\s+summarize|It\s+is\s+worth\s+noting\s+that|"
    r"It['’]?s\s+important\s+to\s+note\s+that)\s*,",
    re.IGNORECASE,
)

STOCK_CLICHE_PATTERNS = [
    re.compile(r"\bin\s+today['’]?s\s+(?:fast-paced|digital|ever-evolving|modern|rapidly\s+changing)\s+(?:world|landscape|era|age|environment)\b", re.IGNORECASE),
    re.compile(r"\ba\s+(?:testament\s+to|tapestry\s+of|myriad\s+of|plethora\s+of)\b", re.IGNORECASE),
    re.compile(r"\b(?:navigat(?:e|ing)\s+the\s+(?:complex\s+)?(?:landscape|complexities)|unlock(?:ing)?\s+the\s+(?:full\s+)?potential)\b", re.IGNORECASE),
    re.compile(r"\b(?:plays\s+a\s+(?:crucial|pivotal|vital|significant)\s+role\s+in|in\s+the\s+realm\s+of|delv(?:e|es|ing)\s+(?:deeper\s+)?into)\b", re.IGNORECASE),
    re.compile(r"(?:^|[.!?]\s+)(?:The\s+result\?|The\s+best\s+part\?|The\s+catch\?|Here['’]?s\s+the\s+(?:thing|kicker|deal):)", re.IGNORECASE),
]

LATINATE_CORPORATE_VERBS = {
    "delve", "delves", "delving", "underscore", "underscores", "underscoring",
    "leverage", "leverages", "leveraging", "utilize", "utilizes", "utilizing",
    "facilitate", "facilitates", "facilitating", "streamline", "streamlines",
    "empower", "empowers", "empowering", "foster", "fosters", "fostering",
    "harness", "harnesses", "harnessing", "revolutionize", "elevate", "meticulous",
    "intricate", "multifaceted", "seamless", "seamlessly", "pivotal", "paramount",
}


def split_sentences(text: str) -> List[str]:
    """Splits prose into sentences while preserving abbreviations."""
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘])", text)
    return [s.strip() for s in raw_sentences if len(s.strip().split()) >= 2]


def extract_word_tokens(text: str) -> List[str]:
    """Extracts lowercase alphabetic word tokens for lexical diversity calculations."""
    return re.findall(r"\b[a-zA-Z]+(?:['’][a-zA-Z]+)?\b", text.lower())


def compute_yules_k(tokens: List[str]) -> float:
    """Computes Yule's Characteristic K (Yule, 1944; Tweedie & Baayen, 1998)."""
    n_tokens = len(tokens)
    if n_tokens < 10:
        return 0.0
    freq_counts = Counter(tokens)
    spectrum = Counter(freq_counts.values())
    s2 = sum((m ** 2) * v_m for m, v_m in spectrum.items())
    return float(10000.0 * (s2 - n_tokens) / (n_tokens ** 2))


def compute_mtld(tokens: List[str], threshold: float = 0.72) -> float:
    """Computes Measure of Textual Lexical Diversity (MTLD; McCarthy & Jarvis, 2010)."""
    if len(tokens) < 15:
        return 50.0

    def _mtld_pass(seq: List[str]) -> float:
        factors = 0.0
        types = set()
        token_count = 0
        for token in seq:
            token_count += 1
            types.add(token)
            ttr = len(types) / token_count
            if ttr <= threshold:
                factors += 1.0
                types.clear()
                token_count = 0
        if token_count > 0:
            ttr = len(types) / token_count
            factors += (1.0 - ttr) / (1.0 - threshold)
        return len(seq) / factors if factors > 0 else float(len(seq))

    forward_mtld = _mtld_pass(tokens)
    backward_mtld = _mtld_pass(list(reversed(tokens)))
    return float((forward_mtld + backward_mtld) / 2.0)


# ============================================================================
# 4. TIER 1 & TIER 2: NEURAL SCORING ENGINES (BINOCULARS, FAST-DETECTGPT, EDITLENS)
# ============================================================================

class NeuralDetectionEngines:
    """
    Lazy-loading singleton container for Tier 1 (Zero-Shot Causal Pair)
    and Tier 2 (Supervised Sequence Classifier) models.
    """

    def __init__(self) -> None:
        self.initialized = False
        self.device = "cpu"
        self.obs_model = None
        self.perf_model = None
        self.causal_tokenizer = None
        self.sup_model = None
        self.sup_tokenizer = None

    def ensure_loaded(self) -> bool:
        if self.initialized:
            return self.obs_model is not None
        self.initialized = True

        if not TORCH_AVAILABLE or os.environ.get("MCP_DISABLE_NEURAL_MODELS", "0") == "1":
            return False

        try:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
            obs_id = os.environ.get("MCP_OBSERVER_MODEL", "Qwen/Qwen2.5-0.5B")
            perf_id = os.environ.get("MCP_PERFORMER_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
            sup_id = os.environ.get(
                "MCP_SUPERVISED_MODEL", "openai-community/roberta-base-openai-detector"
            )

            dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32

            self.causal_tokenizer = AutoTokenizer.from_pretrained(obs_id)
            if self.causal_tokenizer.pad_token is None:
                self.causal_tokenizer.pad_token = self.causal_tokenizer.eos_token

            self.obs_model = AutoModelForCausalLM.from_pretrained(
                obs_id, torch_dtype=dtype
            ).to(self.device).eval()
            self.perf_model = AutoModelForCausalLM.from_pretrained(
                perf_id, torch_dtype=dtype
            ).to(self.device).eval()

            self.sup_tokenizer = AutoTokenizer.from_pretrained(sup_id)
            self.sup_model = AutoModelForSequenceClassification.from_pretrained(
                sup_id
            ).to(self.device).eval()
            return True
        except Exception:
            self.obs_model = None
            return False

    def score_zero_shot_and_supervised(
        self, text: str, sentences: List[str]
    ) -> Tuple[Optional[float], Optional[float], Optional[float], List[float]]:
        if not self.ensure_loaded():
            return None, None, None, []

        with torch.no_grad():
            enc = self.causal_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.device)
            input_ids = enc["input_ids"]

            if input_ids.shape[1] < 4:
                return None, None, None, []

            logits_obs = self.obs_model(input_ids).logits[:, :-1, :].float()
            logits_perf = self.perf_model(input_ids).logits[:, :-1, :].float()
            target_ids = input_ids[:, 1:]

            log_probs_obs = F.log_softmax(logits_obs, dim=-1)
            probs_obs = torch.exp(log_probs_obs)
            log_probs_perf = F.log_softmax(logits_perf, dim=-1)

            token_nll_obs = -log_probs_obs.gather(
                dim=-1, index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
            log_ppl_obs = token_nll_obs.mean().item()

            token_xppl = -(probs_obs * log_probs_perf).sum(dim=-1)
            log_xppl = token_xppl.mean().item()

            binoculars_score = float(log_ppl_obs / max(log_xppl, 1e-6))

            observed_log_p = log_probs_perf.gather(
                dim=-1, index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
            expected_log_p = (probs_obs * log_probs_perf).sum(dim=-1)
            expected_sq_log_p = (probs_obs * (log_probs_perf ** 2)).sum(dim=-1)
            token_var = torch.clamp(expected_sq_log_p - (expected_log_p ** 2), min=1e-6)

            curvature_d = float(
                (observed_log_p.sum() - expected_log_p.sum()).item()
                / math.sqrt(token_var.sum().item())
            )

            sup_tokens = self.sup_tokenizer(
                text, return_tensors="pt", truncation=False
            )["input_ids"][0]
            window_size, stride = 510, 256
            window_probs: List[float] = []

            for start in range(0, max(1, len(sup_tokens)), stride):
                chunk = sup_tokens[start : start + window_size]
                if len(chunk) < 10 and window_probs:
                    break
                chunk_enc = self.sup_tokenizer.prepare_for_model(
                    chunk.tolist(), return_tensors="pt", truncation=True, max_length=512
                )
                chunk_input = {k: v.unsqueeze(0).to(self.device) for k, v in chunk_enc.items()}
                sup_logits = self.sup_model(**chunk_input).logits
                probs = F.softmax(sup_logits, dim=-1)[0]
                id2label = getattr(self.sup_model.config, "id2label", {})
                ai_idx = 0 if "fake" in str(id2label.get(0, "")).lower() else (probs.shape[0] - 1)
                window_probs.append(float(probs[ai_idx].item()))
                if start + window_size >= len(sup_tokens):
                    break

            supervised_prob = float(np.mean(window_probs)) if window_probs else 0.5

            sentence_scores: List[float] = []
            for sent in sentences[:25]:
                s_enc = self.causal_tokenizer(
                    sent, return_tensors="pt", truncation=True, max_length=128
                ).to(self.device)
                if s_enc["input_ids"].shape[1] < 4:
                    sentence_scores.append(supervised_prob)
                    continue
                s_logits = self.perf_model(s_enc["input_ids"]).logits[:, :-1, :].float()
                s_targets = s_enc["input_ids"][:, 1:]
                s_log_probs = F.log_softmax(s_logits, dim=-1)
                s_nll = -s_log_probs.gather(-1, s_targets.unsqueeze(-1)).mean().item()
                sent_prob = 1.0 / (1.0 + math.exp(1.5 * (s_nll - 2.4)))
                sentence_scores.append(float(0.6 * sent_prob + 0.4 * supervised_prob))

            return binoculars_score, curvature_d, supervised_prob, sentence_scores


NEURAL_ENGINES = NeuralDetectionEngines()


# ============================================================================
# 5. FASTMCP SERVER DEFINITION & DIAGNOSTIC ORCHESTRATOR
# ============================================================================

mcp = FastMCP(
    "ForensicAIDetector",
    instructions=(
        "Provides forensic AI text detection on a calibrated 0-100 scale with "
        "quantitative and linguistic reasons. Call analyze_text_provenance(text) "
        "to evaluate authorship across human, AI-assisted, and AI-generated tiers."
    ),
)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0)))


@mcp.tool
def analyze_text_provenance(
    text: str = Field(
        description="The text passage to analyze for AI generation (recommended length: 50+ words)."
    ),
) -> DetectionReport:
    """
    Evaluates whether text content is human-written, AI-assisted, or AI-generated on a 0-100 scale,
    returning a prioritized list of forensic reasons, quantitative metrics, and sentence attributions.
    """
    reasons: List[DiagnosticReason] = []

    sanitized = sanitize_and_profile_typography(text)
    clean_text = sanitized.clean_text
    word_tokens = extract_word_tokens(clean_text)
    word_count = len(word_tokens)
    sentences = split_sentences(clean_text)
    if not sentences and clean_text:
        sentences = [clean_text]

    if sanitized.zero_width_count > 0 or sanitized.homoglyph_count > 0:
        reasons.append(
            DiagnosticReason(
                category="Adversarial Perturbation & Sanitization",
                direction="WARNING",
                impact_weight="HIGH",
                metric_name="Character Obfuscation Artifacts",
                observed_value=f"{sanitized.zero_width_count} zero-width chars, {sanitized.homoglyph_count} homoglyphs",
                reference_benchmark="0 in unmanipulated prose",
                explanation=(
                    f"Detected and stripped {sanitized.zero_width_count} invisible zero-width characters "
                    f"and {sanitized.homoglyph_count} Cyrillic/Greek homoglyph substitutions prior to scoring."
                ),
            )
        )

    sent_lengths = [len(s.split()) for s in sentences]
    mean_sent_len = float(np.mean(sent_lengths)) if sent_lengths else float(word_count)
    std_sent_len = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0
    burstiness_coeff = std_sent_len / max(mean_sent_len, 1.0)

    yules_k = compute_yules_k(word_tokens)
    mtld_score = compute_mtld(word_tokens)

    neg_parallelisms: List[str] = []
    for pat in NEGATIVE_PARALLELISM_PATTERNS:
        neg_parallelisms.extend(m.group(0).strip() for m in pat.finditer(clean_text))

    trailing_participles = [m.group(0).strip() for m in TRAILING_PARTICIPLE_PATTERN.finditer(clean_text)]
    signpost_connectors = [m.group(0).strip() for m in SIGNPOST_CONNECTOR_PATTERN.finditer(clean_text)]

    cliche_matches: List[str] = []
    for pat in STOCK_CLICHE_PATTERNS:
        cliche_matches.extend(m.group(0).strip() for m in pat.finditer(clean_text))

    latinate_hits = [w for w in word_tokens if w in LATINATE_CORPORATE_VERBS]
    latinate_counts = Counter(latinate_hits)
    em_dash_per_100w = (sanitized.em_dash_count / max(word_count, 1)) * 100.0

    p_burst = _sigmoid(-8.5 * (burstiness_coeff - 0.36)) if len(sentences) >= 3 else 0.50

    artifact_points = (
        1.4 * len(neg_parallelisms)
        + 1.1 * len(trailing_participles)
        + 0.8 * max(0, len(signpost_connectors) - 1)
        + 1.3 * len(cliche_matches)
        + 0.45 * max(0, len(latinate_hits) - 1)
        + (0.9 if em_dash_per_100w > 1.5 else 0.0)
        + (0.7 if yules_k > 105.0 and mtld_score < 85.0 else 0.0)
    )
    norm_factor = max(1.0, word_count / 150.0)
    artifact_density = artifact_points / norm_factor
    p_artifacts = _sigmoid(1.25 * (artifact_density - 1.4))

    p_stylo = 0.50 * p_burst + 0.50 * p_artifacts

    if len(sentences) >= 3:
        if burstiness_coeff < 0.30:
            reasons.append(
                DiagnosticReason(
                    category="Sentence Cadence & Burstiness",
                    direction="AI_INDICATOR",
                    impact_weight="HIGH",
                    metric_name="Sentence-Length Burstiness (SD / Mean)",
                    observed_value=f"{burstiness_coeff:.2f} (Mean: {mean_sent_len:.1f}w, SD: {std_sent_len:.1f}w)",
                    reference_benchmark="AI Default: 0.15-0.30 | Natural Human: 0.60-1.00+",
                    explanation=(
                        f"Sentence lengths exhibit metronomic uniformity across {len(sentences)} sentences "
                        f"(clustering closely around {mean_sent_len:.1f} words with a coefficient of variation of {burstiness_coeff:.2f})."
                    ),
                )
            )
        elif burstiness_coeff >= 0.58:
            reasons.append(
                DiagnosticReason(
                    category="Sentence Cadence & Burstiness",
                    direction="HUMAN_INDICATOR",
                    impact_weight="HIGH",
                    metric_name="Sentence-Length Burstiness (SD / Mean)",
                    observed_value=f"{burstiness_coeff:.2f} (Mean: {mean_sent_len:.1f}w, SD: {std_sent_len:.1f}w)",
                    reference_benchmark="AI Default: 0.15-0.30 | Natural Human: 0.60-1.00+",
                    explanation=(
                        f"High sentence-length variation ({burstiness_coeff:.2f}, ranging from {min(sent_lengths)} to {max(sent_lengths)} words) "
                        "reflects natural human rhetorical pacing."
                    ),
                )
            )

    if neg_parallelisms:
        reasons.append(
            DiagnosticReason(
                category="Syntactic & Rhetorical Artifacts",
                direction="AI_INDICATOR",
                impact_weight="HIGH",
                metric_name="Negative Parallelism Framing",
                observed_value=f"{len(neg_parallelisms)} instance(s) detected",
                reference_benchmark="Rare in unprompted human prose",
                explanation=(
                    f"Detected characteristic LLM contrastive framing ('not just X, it's Y' / 'not only... but also'): "
                    f"\"{neg_parallelisms[0][:110]}\"."
                ),
            )
        )

    if trailing_participles:
        reasons.append(
            DiagnosticReason(
                category="Syntactic & Rhetorical Artifacts",
                direction="AI_INDICATOR",
                impact_weight="HIGH" if len(trailing_participles) >= 2 else "MEDIUM",
                metric_name="Trailing Present-Participle Clauses (-ing Tacks)",
                observed_value=f"{len(trailing_participles)} instance(s) detected",
                reference_benchmark="<= 1 per 500 words in human writing",
                explanation=(
                    f"Sentences repeatedly terminate with tacked-on participle commentary (e.g., \"{trailing_participles[0]}\")."
                ),
            )
        )

    if len(signpost_connectors) >= 2:
        examples = ", ".join(f'"{c.strip()}"' for c in signpost_connectors[:3])
        reasons.append(
            DiagnosticReason(
                category="Syntactic & Rhetorical Artifacts",
                direction="AI_INDICATOR",
                impact_weight="MEDIUM",
                metric_name="Mechanical Discourse Signposting",
                observed_value=f"{len(signpost_connectors)} formal transitions ({examples})",
                reference_benchmark="0-1 per 300 words in organic prose",
                explanation=f"Frequent sentence-initial conjunctive adverbs ({examples}) act as mechanical probability bridges.",
            )
        )

    if cliche_matches or len(latinate_hits) >= 2:
        top_words = ", ".join(f"'{w}' ({c}x)" for w, c in latinate_counts.most_common(5))
        cliche_str = f" Stock phrases: {', '.join(repr(c) for c in cliche_matches[:2])}." if cliche_matches else ""
        reasons.append(
            DiagnosticReason(
                category="Lexical Diversity & Repetition",
                direction="AI_INDICATOR",
                impact_weight="HIGH" if (cliche_matches and len(latinate_hits) >= 3) else "MEDIUM",
                metric_name="Clustered AI Register & Stock Idioms",
                observed_value=f"{len(latinate_hits)} Latinate/AI markers; {len(cliche_matches)} stock phrases",
                reference_benchmark="Isolated occurrences only in human text",
                explanation=f"Passage clusters multiple high-probability LLM lexical choices ({top_words or 'none'}).{cliche_str}",
            )
        )

    if em_dash_per_100w > 1.5:
        reasons.append(
            DiagnosticReason(
                category="Syntactic & Rhetorical Artifacts",
                direction="AI_INDICATOR",
                impact_weight="LOW",
                metric_name="Em-Dash Punctuation Density",
                observed_value=f"{sanitized.em_dash_count} em-dashes ({em_dash_per_100w:.1f} per 100 words)",
                reference_benchmark="< 0.5 per 100 words in standard human prose",
                explanation="Elevated frequency of em-dashes used to splice parenthetical clauses within paragraphs.",
            )
        )

    if word_count >= 40:
        if yules_k > 110.0 or mtld_score < 72.0:
            reasons.append(
                DiagnosticReason(
                    category="Lexical Diversity & Repetition",
                    direction="AI_INDICATOR",
                    impact_weight="MEDIUM",
                    metric_name="Yule's K & MTLD Lexical Constancy",
                    observed_value=f"Yule's K: {yules_k:.1f} | MTLD: {mtld_score:.1f}",
                    reference_benchmark="Human Prompted: K ~ 66.8, MTLD ~ 134 | Synthetic: K ~ 115.4, MTLD ~ 93.9",
                    explanation=f"Elevated token recurrence (Yule's K = {yules_k:.1f}) paired with compressed sequential diversity (MTLD = {mtld_score:.1f}).",
                )
            )
        elif yules_k < 75.0 and mtld_score > 110.0:
            reasons.append(
                DiagnosticReason(
                    category="Lexical Diversity & Repetition",
                    direction="HUMAN_INDICATOR",
                    impact_weight="MEDIUM",
                    metric_name="Yule's K & MTLD Lexical Constancy",
                    observed_value=f"Yule's K: {yules_k:.1f} | MTLD: {mtld_score:.1f}",
                    reference_benchmark="Human Prompted: K ~ 66.8, MTLD ~ 134 | Synthetic: K ~ 115.4, MTLD ~ 93.9",
                    explanation=f"High sequential vocabulary richness (MTLD = {mtld_score:.1f}) and low token repetition (Yule's K = {yules_k:.1f}).",
                )
            )

    bino_score, curvature_d, sup_prob, sent_probs = NEURAL_ENGINES.score_zero_shot_and_supervised(
        clean_text, sentences
    )

    if bino_score is not None and curvature_d is not None and sup_prob is not None:
        p_bino = _sigmoid(-14.0 * (bino_score - 0.90))
        p_curv = _sigmoid(1.6 * (curvature_d - 1.35))
        p_stat = 0.55 * p_bino + 0.45 * p_curv
        p_sup = sup_prob

        reasons.append(
            DiagnosticReason(
                category="Token Probability & Curvature",
                direction="AI_INDICATOR" if p_stat >= 0.55 else "HUMAN_INDICATOR",
                impact_weight="HIGH",
                metric_name="Binoculars Cross-Perplexity & Fast-DetectGPT Curvature",
                observed_value=f"Binoculars B = {bino_score:.3f} | Curvature d = {curvature_d:.2f}",
                reference_benchmark="AI: B < 0.88, d > 1.50 | Human: B > 0.94, d < 0.60",
                explanation=(
                    f"Zero-shot dual-model analysis yielded a cross-perplexity ratio of {bino_score:.3f} and conditional probability "
                    f"curvature of {curvature_d:.2f} standard deviations."
                ),
            )
        )

        reasons.append(
            DiagnosticReason(
                category="Supervised Neural Signature",
                direction="AI_INDICATOR" if p_sup >= 0.55 else "HUMAN_INDICATOR",
                impact_weight="HIGH",
                metric_name="Sliding-Window Sequence Classifier Probability",
                observed_value=f"{p_sup * 100:.1f}% AI-like discourse representation",
                reference_benchmark="< 25% for unedited human prose",
                explanation="Bidirectional transformer sequence classification across 512-token windows.",
            )
        )

        if p_sup > 0.78 and p_stylo > 0.62 and p_stat < 0.42:
            composite_prob = 0.20 * p_stat + 0.50 * p_sup + 0.30 * p_stylo
            reasons.append(
                DiagnosticReason(
                    category="Calibration & Guardrail Notice",
                    direction="WARNING",
                    impact_weight="MEDIUM",
                    metric_name="Synonym-Substitution Evasion Pattern",
                    observed_value="High Supervised/Stylometric AI signal vs. suppressed Token Curvature",
                    reference_benchmark="Consistent across tiers in unattacked text",
                    explanation="Divergence indicates likely word-level synonym substitution or light humanizer editing.",
                )
            )
        else:
            composite_prob = 0.45 * p_stat + 0.35 * p_sup + 0.20 * p_stylo
    else:
        p_stat = p_stylo
        p_sup = p_stylo
        composite_prob = p_stylo

    if (
        burstiness_coeff > 0.55
        and len(neg_parallelisms) == 0
        and len(trailing_participles) == 0
        and len(cliche_matches) == 0
        and composite_prob > 0.45
    ):
        composite_prob *= 0.72
        reasons.append(
            DiagnosticReason(
                category="Calibration & Guardrail Notice",
                direction="HUMAN_INDICATOR",
                impact_weight="MEDIUM",
                metric_name="Non-Native / Simple-Lexicon Guardrail",
                observed_value=f"Burstiness {burstiness_coeff:.2f} with 0 syntactic AI cliches",
                reference_benchmark="Active when high sentence variance pairs with simple vocabulary",
                explanation="Applied false-positive dampening to protect against L2/non-native speaker bias.",
            )
        )

    if word_count < 50:
        confidence: Literal["HIGH", "MODERATE", "LOW (Short Text < 50 Words)"] = (
            "LOW (Short Text < 50 Words)"
        )
        weight = max(0.25, word_count / 50.0)
        composite_prob = weight * composite_prob + (1.0 - weight) * 0.50
        reasons.append(
            DiagnosticReason(
                category="Calibration & Guardrail Notice",
                direction="WARNING",
                impact_weight="MEDIUM",
                metric_name="Minimum Sample Length Threshold",
                observed_value=f"{word_count} words",
                reference_benchmark=">= 50 words required for high-confidence attribution",
                explanation=f"Passage contains only {word_count} words (below the 50-word forensic floor).",
            )
        )
    elif word_count < 120:
        confidence = "MODERATE"
    else:
        confidence = "HIGH"

    final_score = int(round(max(0.0, min(100.0, composite_prob * 100.0))))

    if final_score < 25:
        prov_tier: Literal[
            "Human-Written (0-24)",
            "Lightly AI-Assisted / Polished (25-54)",
            "Moderately to Heavily AI-Edited (55-79)",
            "AI-Generated (80-100)",
        ] = "Human-Written (0-24)"
    elif final_score < 55:
        prov_tier = "Lightly AI-Assisted / Polished (25-54)"
    elif final_score < 80:
        prov_tier = "Moderately to Heavily AI-Edited (55-79)"
    else:
        prov_tier = "AI-Generated (80-100)"

    flagged_spans: List[SpanAttribution] = []
    for idx, sent in enumerate(sentences[:25]):
        if sent_probs and idx < len(sent_probs):
            s_prob = sent_probs[idx]
        else:
            len_diff = abs(len(sent.split()) - 17.0)
            has_local_artifact = any(
                pat.search(sent) for pat in NEGATIVE_PARALLELISM_PATTERNS + STOCK_CLICHE_PATTERNS
            ) or bool(TRAILING_PARTICIPLE_PATTERN.search(sent))
            s_prob = min(0.98, composite_prob + (0.18 if has_local_artifact else 0.0) - (0.01 * len_diff))

        s_score = int(round(max(0.0, min(100.0, s_prob * 100.0))))
        s_label: Literal["Human", "AI-Assisted", "AI-Generated"] = (
            "AI-Generated" if s_score >= 75 else ("AI-Assisted" if s_score >= 40 else "Human")
        )
        flagged_spans.append(
            SpanAttribution(
                sentence_index=idx,
                text_excerpt=sent,
                ai_likelihood_score=s_score,
                classification=s_label,
            )
        )

    weight_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    reasons.sort(key=lambda r: weight_order.get(r.impact_weight, 3))

    return DetectionReport(
        ai_generation_score=final_score,
        provenance_classification=prov_tier,
        confidence_level=confidence,
        reasons=reasons,
        sub_scores={
            "zero_shot_probability_score": round(p_stat * 100.0, 1),
            "supervised_classifier_score": round(p_sup * 100.0, 1),
            "forensic_stylometry_score": round(p_stylo * 100.0, 1),
        },
        quantitative_metrics={
            "word_count": float(word_count),
            "sentence_count": float(len(sentences)),
            "burstiness_coefficient": round(burstiness_coeff, 4),
            "yules_k_repetition": round(yules_k, 2),
            "mtld_lexical_diversity": round(mtld_score, 2),
            "em_dashes_per_100_words": round(em_dash_per_100w, 2),
            "binoculars_ratio": round(bino_score, 4) if bino_score is not None else -1.0,
            "fast_detectgpt_curvature": round(curvature_d, 4) if curvature_d is not None else 0.0,
        },
        flagged_sentences=flagged_spans,
    )


if __name__ == "__main__":
    mcp.run()