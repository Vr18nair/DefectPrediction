import os
# Resolve OMP: Error #15 (Multiple OpenMP runtimes)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import re
import requests
import json
import random
import hashlib
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
print("🔥 NEW VERSION OF app.py LOADED 🔥")

# ------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------
app = FastAPI(title="Defect Prediction API")

# ✅ CORS (safe for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# MODEL SETUP
# ------------------------------------------------------------
MODEL_DIR = "./final_graphcodebert_balanced_best"

print("Device set to use CPU")
device = -1  # CPU safe

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
    device=device
)

# ------------------------------------------------------------
# EXPLAINABLE AI (SHAP) SETUP (READ-ONLY, STARTUP INIT)
# ------------------------------------------------------------
XAI_TOP_K = 10
XAI_TOKEN_MAX_CHARS = 32
XAI_MAX_EVALS = 256
XAI_MAX_INPUT_TOKENS = 256
XAI_RANDOM_SEED = 0
XAI_OUTPUT_NAMES = ["clean_probability", "defect_probability"]

# Minimal, static background dataset (hardcoded)
XAI_BACKGROUND: List[str] = [
    "int main() { return 0; }",
    "if (ptr == NULL) return -1;",
    "snprintf(buf, sizeof(buf), \"%s\", input);",
    "strncpy(dst, src, n);",
]

try:
    import shap  # type: ignore

    random.seed(XAI_RANDOM_SEED)
    np.random.seed(XAI_RANDOM_SEED)
    torch.manual_seed(XAI_RANDOM_SEED)

    def _xai_predict_proba(texts: List[str]) -> np.ndarray:
        outputs = pipe(list(texts))
        probs: List[List[float]] = []
        for item in outputs:
            item_sorted = sorted(item, key=lambda d: d["label"])
            probs.append([float(item_sorted[0]["score"]), float(item_sorted[1]["score"])])
        return np.asarray(probs, dtype=float)

    _xai_masker = None
    if hasattr(shap, "maskers") and hasattr(shap.maskers, "Text"):
        _xai_masker = shap.maskers.Text(tokenizer)

    XAI_EXPLAINER = None
    # Try multiple constructor signatures for compatibility across SHAP versions.
    for _init in (
        lambda: shap.Explainer(_xai_predict_proba, _xai_masker, XAI_BACKGROUND, output_names=XAI_OUTPUT_NAMES, seed=XAI_RANDOM_SEED),
        lambda: shap.Explainer(_xai_predict_proba, _xai_masker, output_names=XAI_OUTPUT_NAMES, seed=XAI_RANDOM_SEED),
        lambda: shap.Explainer(_xai_predict_proba, _xai_masker, output_names=XAI_OUTPUT_NAMES),
        lambda: shap.Explainer(_xai_predict_proba, XAI_BACKGROUND, output_names=XAI_OUTPUT_NAMES, seed=XAI_RANDOM_SEED),
        lambda: shap.Explainer(_xai_predict_proba, XAI_BACKGROUND, output_names=XAI_OUTPUT_NAMES),
    ):
        try:
            XAI_EXPLAINER = _init()
            break
        except TypeError:
            continue
        except Exception:
            XAI_EXPLAINER = None
            break
except ModuleNotFoundError:
    shap = None  # type: ignore
    XAI_EXPLAINER = None
    print("SHAP not installed; XAI explanations disabled (install 'shap' to enable).")

@contextmanager
def _xai_determinism_context(seed: int):
    py_state = None
    np_state = None
    torch_state = None

    try:
        py_state = random.getstate()
        random.seed(seed)
    except Exception:
        py_state = None

    try:
        np_state = np.random.get_state()
        np.random.seed(seed)
    except Exception:
        np_state = None

    try:
        torch_state = torch.get_rng_state()
        torch.manual_seed(seed)
    except Exception:
        torch_state = None

    try:
        yield
    finally:
        try:
            if py_state is not None:
                random.setstate(py_state)
        except Exception:
            pass

        try:
            if np_state is not None:
                np.random.set_state(np_state)
        except Exception:
            pass

        try:
            if torch_state is not None:
                torch.set_rng_state(torch_state)
        except Exception:
            pass


def _xai_shap_explain_one(code: str, model_confidence: float) -> Optional[Dict[str, Any]]:
    """
    Purely descriptive SHAP explanation for a single input.
    Must not influence classification outcomes.
    """
    if XAI_EXPLAINER is None:
        return None

    # Strict runtime safeguards: skip XAI for large inputs (best-effort, non-blocking).
    try:
        token_ids = tokenizer(code, add_special_tokens=True, truncation=False)["input_ids"]
        if isinstance(token_ids, (list, tuple)) and len(token_ids) > XAI_MAX_INPUT_TOKENS:
            return None
    except Exception:
        return None

    try:
        digest = hashlib.sha256(code.encode("utf-8", errors="ignore")).digest()
        per_input_seed = int.from_bytes(digest[:4], byteorder="little", signed=False) ^ XAI_RANDOM_SEED
        per_input_seed = int(per_input_seed % (2**31 - 1))
    except Exception:
        return None

    try:
        with _xai_determinism_context(per_input_seed):
            try:
                exp = XAI_EXPLAINER([code], max_evals=XAI_MAX_EVALS)
            except TypeError:
                # If the explainer can't accept a bound, skip rather than risk unbounded runtime.
                return None
    except Exception:
        return None

    exp_obj: Any = exp
    if not hasattr(exp_obj, "values") and isinstance(exp_obj, (list, tuple)):
        if len(exp_obj) > 1 and hasattr(exp_obj[1], "values"):
            exp_obj = exp_obj[1]  # defect output (preferred)
        elif len(exp_obj) > 0 and hasattr(exp_obj[0], "values"):
            exp_obj = exp_obj[0]

    if not hasattr(exp_obj, "values"):
        return None

    tokens: List[str] = []
    try:
        data = getattr(exp_obj, "data", None)
        if isinstance(data, np.ndarray):
            if data.ndim == 2 and data.shape[0] == 1:
                tokens = [str(t) for t in list(data[0])]
            elif data.ndim == 1:
                tokens = [str(t) for t in list(data)]
        elif isinstance(data, list):
            if len(data) == 1 and isinstance(data[0], (list, tuple, np.ndarray)):
                tokens = [str(t) for t in list(data[0])]
            elif all(not isinstance(x, (list, tuple, np.ndarray)) for x in data):
                tokens = [str(t) for t in data]
        elif isinstance(data, tuple):
            tokens = [str(t) for t in list(data)]
    except Exception:
        tokens = []

    output_names = getattr(exp_obj, "output_names", None)
    output_dim = len(output_names) if isinstance(output_names, (list, tuple)) else None

    values = np.asarray(getattr(exp_obj, "values"), dtype=float)

    # Normalize to (tokens,) for the defect probability output.
    if values.ndim >= 2 and values.shape[0] == 1:
        values = values[0]

    if values.ndim == 2:
        if output_dim and values.shape[1] == output_dim:
            values = values[:, 1 if output_dim > 1 else 0]
        elif output_dim and values.shape[0] == output_dim:
            values = values[1 if output_dim > 1 else 0]
        elif values.shape[0] == 1:
            values = values[0]
        elif values.shape[1] == 1:
            values = values[:, 0]
        else:
            return None
    if values.ndim != 1:
        return None

    if not tokens:
        # Fall back to tokenizer tokens only for shaping output; do not expose ids.
        ids = tokenizer.encode(code, add_special_tokens=True)
        tokens = tokenizer.convert_ids_to_tokens(ids)

    n = min(len(tokens), int(values.shape[0]))
    tokens = tokens[:n]
    values = values[:n]

    skip = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"}
    items: List[tuple[str, float]] = []
    for t, v in zip(tokens, values):
        tok = str(t).strip()
        if not tok or tok in skip:
            continue
        tok = tok.replace("\u0120", "").replace("\u2581", "")
        tok = tok.strip()
        if not tok:
            continue
        items.append((tok, float(v)))

    if not items:
        return {"type": "shap", "model_confidence": round(float(model_confidence), 4), "top_tokens": []}

    abs_sum = float(sum(abs(v) for _, v in items))
    if abs_sum <= 0.0:
        normalized = [(t, 0.0) for t, _ in items]
    else:
        normalized = [(t, v / abs_sum) for t, v in items]

    top = sorted(normalized, key=lambda tv: (-abs(tv[1]), tv[0]))[:XAI_TOP_K]
    payload_tokens = [
        {"token": t[:XAI_TOKEN_MAX_CHARS], "impact": round(float(v), 4)}
        for t, v in top
    ]

    return {
        "type": "shap",
        "model_confidence": round(float(model_confidence), 4),
        "top_tokens": payload_tokens,
    }

# ------------------------------------------------------------
# GEMINI LLM CONFIGURATION (SCAFFOLDING)
# ------------------------------------------------------------
ENABLE_LLM_EXPLANATION = True
GEMINI_API_KEY = "AIzaSyC8XkDPTCqHepjc7OT7RI-acBvcHKQTG0o"
print("GEMINI_API_KEY set:", bool(GEMINI_API_KEY))
LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT_SECONDS", "20"))

# Static, deterministic prompt template
LLM_EXPLANATION_PROMPT_TEMPLATE = (
    "As an expert security researcher, summarize why the following code tokens "
    "({tokens}) contribute to a defect probability of {prob} in the following code snippet.\n\n"
    "Code:\n{code}\n\n"
    "Requirement: Return ONLY a JSON object with a single key 'summary'.\n"
    "STRICT CONSTRAINTS:\n"
    "- The summary must be exactly one sentence, concise, and technical.\n"
    "- Use cautious, probabilistic language (e.g., 'may indicate', 'suggests').\n"
    "- DO NOT reclassify the result or re-evaluate the verdict.\n"
    "- DO NOT introduce new vulnerability categories not implied by the tokens.\n"
    "- DO NOT claim certainty (e.g., avoid 'definitely', 'is vulnerable').\n"
    "- DO NOT reference rules or rule-based detection."
)

# Determine effective activation state
if not ENABLE_LLM_EXPLANATION:
    LLM_LAYER_ACTIVE = False
    print("LLM explanation layer: DISABLED")
elif not GEMINI_API_KEY:
    LLM_LAYER_ACTIVE = False
    print("LLM explanation layer: DISABLED (missing API key)")
else:
    LLM_LAYER_ACTIVE = True
    print("LLM explanation layer: ENABLED (Gemini configured)")


def call_gemini(prompt: str) -> Optional[str]:
    """
    Reliable and bounded Gemini transport function using direct HTTP.
    Must not influence classification. Catch all exceptions.
    """
    if not LLM_LAYER_ACTIVE:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "topP": 1.0,
            "topK": 1,
            "candidateCount": 1,
            "maxOutputTokens": 512,
        }
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        
        res_json = response.json()
        
        # Minimally extract text from standard Gemini response structure
        # candidates[0].content.parts[0].text
        candidates = res_json.get("candidates", [])
        if candidates and len(candidates) > 0:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts and len(parts) > 0:
                return parts[0].get("text")
                
    except requests.exceptions.Timeout:
        print(f"LLM Warning: Gemini call timed out after {LLM_TIMEOUT_SECONDS}s")
    except requests.exceptions.RequestException as e:
        print(f"LLM Warning: Gemini transport error: {e}")
    except Exception as e:
        print(f"LLM Warning: Unexpected error in Gemini transport: {e}")

    return None


def validate_gemini_response(raw_text: Optional[str]) -> Optional[str]:
    """
    Strictly validates that Gemini output reflects the required JSON schema:
    { "summary": "string" }
    Also performs defensive content checks to prevent authority leakage.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None

    # Strip markdown code fences if present (Gemini commonly wraps JSON this way)
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        # Remove opening fence (e.g. ```json or ```)
        raw_text = raw_text.split("\n", 1)[-1] if "\n" in raw_text else raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()
    print(raw_text)
    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        print("LLM Validation: rejected (invalid JSON)")
        return None

    if not isinstance(data, dict):
        return None

    # Ensure exactly one key: "summary"
    if list(data.keys()) != ["summary"]:
        return None

    summary = data.get("summary")
    if not isinstance(summary, str):
        return None

    summary = summary.strip()
    
    # Enforce non-empty and length bounds (500 chars)
    if not summary or len(summary) > 500:
        return None

    # Authority Leakage Check
    forbidden_phrases = [
        "definitely",
        "certainly",
        "is vulnerable",
        "guaranteed",
        "rule detected",
        "the rule",
        "reclassified",
        "manual checker"
    ]
    summary_lower = summary.lower()
    for phrase in forbidden_phrases:
        if phrase in summary_lower:
            return None

    return summary


def generate_fallback_summary(top_tokens: List[Dict[str, Any]], defect_probability: float) -> str:
    """
    Deterministic fallback generator. Ensures an explanation exists even if LLM fails.
    Uses SHAP tokens and cautious language. Same input -> Identical output.
    """
    if not top_tokens:
        return f"Based on the model's inference (defect probability: {defect_probability:.2f}), no specific code tokens were identified as having high individual influence."

    tok_names = [t.get("token", "unk") for t in top_tokens[:5]]
    token_list_str = ", ".join([f"'{t}'" for t in tok_names])

    prob_str = f"{defect_probability:.2f}"
    
    return (
        f"Based on the model's inference (defect probability: {prob_str}), the following tokens "
        f"were found to have high statistical influence on the classification: {token_list_str}. "
        "This indicates these specific patterns contributed most to the model's decision."
    )


# ------------------------------------------------------------
# REQUEST SCHEMA
# ------------------------------------------------------------
class CodeInput(BaseModel):
    code: str

def safe_override(code: str):
    """
    Explicitly safe patterns that must NEVER be marked defective.
    """
    safe_patterns = [
        r"\bsnprintf\s*\(",
        r"\bstrncpy\s*\(",
        r"\bmemcpy\s*\(",
    ]

    for pattern in safe_patterns:
        if re.search(pattern, code):
            return True

    return False


SAFE_API_OVERRIDE_RULES: List[Dict[str, str]] = [
    {
        "name": "safe_api_snprintf",
        "pattern": r"\bsnprintf\s*\(",
        "reason": "Uses snprintf(), a bounded formatting API that mitigates common buffer overflow risks.",
        "certainty": "certain",
        "rule_type": "override",
        "verdict": "clean",
    },
    {
        "name": "safe_api_strncpy",
        "pattern": r"\bstrncpy\s*\(",
        "reason": "Uses strncpy(), a bounded copy API that mitigates common buffer overflow risks.",
        "certainty": "certain",
        "rule_type": "override",
        "verdict": "clean",
    },
    {
        "name": "safe_api_memcpy",
        "pattern": r"\bmemcpy\s*\(",
        "reason": "Uses memcpy() with explicit size, which is commonly used in controlled, bounds-aware copying.",
        "certainty": "certain",
        "rule_type": "override",
        "verdict": "clean",
    },
]


def _safe_api_override_justification(code: str) -> Optional[Dict[str, str]]:
    """
    Returns deterministic metadata for the first safe-API override rule that matches.
    Must mirror safe_override patterns and ordering exactly.
    """
    for rule in SAFE_API_OVERRIDE_RULES:
        if re.search(rule["pattern"], code):
            return {
                "name": rule["name"],
                "reason": rule["reason"],
                "certainty": rule["certainty"],
                "rule_type": rule["rule_type"],
                "verdict": rule["verdict"],
            }
    return None

# ------------------------------------------------------------
# 🔴 MANUAL DEFECT CHECKER (HIGHEST PRIORITY)
# ------------------------------------------------------------
MANUAL_DEFECT_RULES: List[Dict[str, str]] = [
    {
        "name": "buffer_overflow_strcpy",
        "pattern": r"\bstrcpy\s*\(",
        "reason": "Use of strcpy() can overflow the destination buffer (unbounded copy).",
        "certainty": "certain",
    },
    {
        "name": "buffer_overflow_gets",
        "pattern": r"\bgets\s*\(",
        "reason": "Use of gets() is inherently unsafe (no bounds checking).",
        "certainty": "certain",
    },
    {
        "name": "buffer_overflow_sprintf",
        "pattern": r"\bsprintf\s*\(",
        "reason": "Use of sprintf() can overflow the destination buffer (unbounded formatting).",
        "certainty": "certain",
    },
    {
        "name": "scanf_unbounded_string",
        "pattern": r'\bscanf\s*\(\s*"%s',
        "reason": 'scanf("%s", ...) without a width specifier can overflow the destination buffer.',
        "certainty": "certain",
    },
    {
        "name": "use_after_free_array_access",
        "pattern": r"\bfree\s*\(\s*(\w+)\s*\).*?\1\s*\[",
        "reason": "Freed memory is accessed afterwards (use-after-free).",
        "certainty": "certain",
    },
    {
        "name": "dangling_pointer_return",
        "pattern": r"return\s*&\s*\w+",
        "reason": "Returning the address of a local variable yields a dangling pointer.",
        "certainty": "certain",
    },
]

def manual_defect_checker(code: str):
    """
    Guaranteed defects only.
    NO generic pointer rules.
    NO false positives.
    """

    # Ordered list: first match wins (authoritative, deterministic).
    for rule in MANUAL_DEFECT_RULES:
        if re.search(rule["pattern"], code, re.DOTALL):
            return True

    return False


def _manual_defect_rule_justification(code: str) -> Optional[Dict[str, str]]:
    """
    Returns deterministic metadata for the first manual defect rule that matches.
    Must mirror manual_defect_checker rule ordering and regex patterns exactly.
    """
    for rule in MANUAL_DEFECT_RULES:
        if re.search(rule["pattern"], code, re.DOTALL):
            return {
                "name": rule["name"],
                "reason": rule["reason"],
                "certainty": rule["certainty"],
            }
    return None

# ------------------------------------------------------------
# 🟠 STATIC SAFETY CHECK (SECOND PRIORITY)
# ------------------------------------------------------------
SAFE_FUNCTIONS = ["snprintf", "strncpy", "memcpy"]

STATIC_SAFETY_RULES: List[Dict[str, str]] = [
    {
        "name": "unsafe_api_strcpy",
        "category": "buffer_overflow_api",
        "pattern": r"\bstrcpy\s*\(",
        "reason": "Use of strcpy() is commonly associated with buffer overflow vulnerabilities (unbounded copy).",
        "rule_type": "heuristic",
        "confidence": "high",
    },
    {
        "name": "unsafe_api_gets",
        "category": "unbounded_input",
        "pattern": r"\bgets\s*\(",
        "reason": "Use of gets() is commonly associated with vulnerabilities (no bounds checking).",
        "rule_type": "heuristic",
        "confidence": "high",
    },
    {
        "name": "unsafe_api_sprintf",
        "category": "buffer_overflow_api",
        "pattern": r"\bsprintf\s*\(",
        "reason": "Use of sprintf() is commonly associated with buffer overflow vulnerabilities (unbounded formatting).",
        "rule_type": "heuristic",
        "confidence": "high",
    },
    {
        "name": "scanf_unbounded_string",
        "category": "unbounded_input",
        "pattern": r'\bscanf\s*\(\s*"%s',
        "reason": 'Use of scanf("%s", ...) without a width specifier is commonly associated with buffer overflows.',
        "rule_type": "heuristic",
        "confidence": "high",
    },
]

def static_safety_check(code: str):
    # If safe APIs are used, do NOT flag
    for fn in SAFE_FUNCTIONS:
        if fn in code:
            return False

    for rule in STATIC_SAFETY_RULES:
        if re.search(rule["pattern"], code):
            return True

    return False


def _static_rule_justification(code: str) -> Optional[Dict[str, str]]:
    """
    Returns deterministic metadata for the first static safety heuristic that matches.
    Must mirror static_safety_check pattern ordering and regex patterns exactly.
    """
    for rule in STATIC_SAFETY_RULES:
        if re.search(rule["pattern"], code):
            return {
                "name": rule["name"],
                "category": rule["category"],
                "reason": rule["reason"],
                "rule_type": rule["rule_type"],
                "confidence": rule["confidence"],
            }
    return None


STRUCTURAL_CLEAN_RULES: List[Dict[str, str]] = [
    {
        "name": "structural_null_check_early_return",
        "pattern": r"\bif\s*\(\s*!\s*\w+\s*\)\s*return",
        "reason": "Detected a defensive null-check with early return, reducing risk of null pointer dereference.",
        "certainty": "certain",
        "rule_type": "override",
        "verdict": "clean",
    },
    {
        "name": "structural_null_check_explicit",
        "pattern": r"\bif\s*\(\s*\w+\s*==\s*NULL\s*\)",
        "reason": "Detected an explicit NULL check, reducing risk of null pointer dereference.",
        "certainty": "certain",
        "rule_type": "override",
        "verdict": "clean",
    },
    {
        "name": "structural_bounds_check",
        "pattern": r"\bindex\s*<\s*0\s*\|\|\s*index\s*>=\s*\w+",
        "reason": "Detected a bounds check guarding index access, reducing risk of out-of-bounds memory access.",
        "certainty": "certain",
        "rule_type": "override",
        "verdict": "clean",
    },
    {
        "name": "structural_malloc_null_check",
        "pattern": r"\bmalloc\s*\(.*\)\s*;\s*if\s*\(\s*!\w+\s*\)",
        "reason": "Detected allocation followed by a null-check, reducing risk of null pointer dereference after allocation.",
        "certainty": "certain",
        "rule_type": "override",
        "verdict": "clean",
    },
    {
        "name": "structural_free_present",
        "pattern": r"\bfree\s*\(\s*\w+\s*\)",
        "reason": "Detected an explicit free() call, indicating explicit memory management and cleanup intent.",
        "certainty": "certain",
        "rule_type": "override",
        "verdict": "clean",
    },
    {
        "name": "structural_simple_arithmetic_return",
        "pattern": r"\breturn\s+\w+\s*[\*\+\-\/]\s*\w+",
        "reason": "Detected a simple arithmetic return pattern with no obvious risky memory operations.",
        "certainty": "certain",
        "rule_type": "override",
        "verdict": "clean",
    },
]


def _structural_clean_override_justification(code: str) -> Optional[Dict[str, str]]:
    """
    Returns deterministic metadata for the first structural-clean override pattern that matches.
    Must mirror structural_clean_override patterns and ordering exactly.
    """
    for rule in STRUCTURAL_CLEAN_RULES:
        if re.search(rule["pattern"], code, re.DOTALL):
            return {
                "name": rule["name"],
                "reason": rule["reason"],
                "certainty": rule["certainty"],
                "rule_type": rule["rule_type"],
                "verdict": rule["verdict"],
            }
    return None
def structural_clean_override(code: str):
    """
    Detects common SAFE coding patterns.
    If matched, ML false positives are ignored.
    """

    safe_patterns = [
        r"\bif\s*\(\s*!\s*\w+\s*\)\s*return",           # null check (!p)
        r"\bif\s*\(\s*\w+\s*==\s*NULL\s*\)",            # p == NULL
        r"\bindex\s*<\s*0\s*\|\|\s*index\s*>=\s*\w+",  # bounds check
        r"\bmalloc\s*\(.*\)\s*;\s*if\s*\(\s*!\w+\s*\)", # malloc + check
        r"\bfree\s*\(\s*\w+\s*\)",                      # proper free
        r"\breturn\s+\w+\s*[\*\+\-\/]\s*\w+"            # simple arithmetic
    ]

    for pattern in safe_patterns:
        if re.search(pattern, code, re.DOTALL):
            return True

    return False


# ------------------------------------------------------------
# 🧠 CLASSIFICATION LOGIC (PRIORITY PIPELINE)
# ------------------------------------------------------------
# ARCHITECTURAL INVARIANTS – DO NOT MODIFY
# ------------------------------------------------------------
# 1. Pipeline Order: The classification pipeline order is fixed and must not change:
#    - Manual Defect Rules (Highest Priority)
#    - Safe API Overrides
#    - Structural Clean Overrides
#    - Static Heuristics
#    - ML Inference (Lowest Priority)
# 2. Rule Authority: Rule logic (manual, safe, structural, static) supersedes ML.
# 3. ML Thresholds: Decision thresholds (e.g., 0.60) must not change.
# 4. Decision Source: 'decision_source' must remain authoritative and untouched.
# 5. Mutual Exclusivity: 'rule_justification' and ('explanation_tokens' + 'explanation_summary') 
#    are mutually exclusive.
#
# LLM layer is interpretive only. It must not influence classification.
# ------------------------------------------------------------
def classify_one(code: str):

    # 🔴 1. MANUAL DEFECT CHECKER (HIGHEST PRIORITY)
    if manual_defect_checker(code):
        return {
            "prediction": "defective",
            "clean_probability": 0.0,
            "defect_probability": 1.0,
            "decision_source": "manual_checker",
            "rule_justification": _manual_defect_rule_justification(code) or {
                "name": "unknown_manual_rule",
                "reason": "A guaranteed manual defect rule matched.",
                "certainty": "certain",
            },
        }

    # 🟢 0. SAFE API OVERRIDE
    if safe_override(code):
        return {
            "prediction": "clean",
            "clean_probability": 1.0,
            "defect_probability": 0.0,
            "decision_source": "safe_api_override",
            "rule_justification": _safe_api_override_justification(code) or {
                "name": "unknown_safe_api_rule",
                "reason": "A safe API pattern matched.",
                "certainty": "certain",
                "rule_type": "override",
                "verdict": "clean",
            },
        }

    # 🟢 0.5 STRUCTURAL CLEAN FLAG (evaluated pre-ML, applied post-ML)
    structural_flag = structural_clean_override(code)
    structural_justification = (
        _structural_clean_override_justification(code)
        if structural_flag
        else None
    )

    # 🟠 2. STATIC SAFETY CHECK
    static_flag = static_safety_check(code)
    static_justification = _static_rule_justification(code) if static_flag else None

    # 🟡 3. ML MODEL
    out = pipe(code)[0]
    out = sorted(out, key=lambda d: d["label"])

    clean_prob = out[0]["score"]
    defect_prob = out[1]["score"]

    if static_flag:
        pred = "defective"
        source = "static_rule"
    elif defect_prob >= 0.60:
        pred = "defective"
        source = "ml_high_confidence"
    else:
        pred = "clean"
        source = "ml_low_confidence"

    # 🟢 0.5 STRUCTURAL OVERRIDE (post-ML, conditional)
    # Only reinforce an already-clean ML decision. Never suppress defects.
    if structural_flag and pred == "clean" and not static_flag:
        source = "structural_clean_override"

    # --- DEBUG: Classification Flow ---
    print("SOURCE:", source)
    print("PRED:", pred)
    print("STATIC_FLAG:", static_flag)
    print("STRUCTURAL_FLAG:", structural_flag if 'structural_flag' in locals() else None)

    result: Dict[str, Any] = {
        "prediction": pred,
        "clean_probability": clean_prob,
        "defect_probability": defect_prob,
        "decision_source": source
    }

    # 1. Rule-based path (Static Heuristics)
    if source == "static_rule" and static_justification is not None:
        result["rule_justification"] = static_justification

    # 1b. Structural clean override (post-ML bias reducer)
    elif source == "structural_clean_override" and structural_justification is not None:
        result["rule_justification"] = structural_justification

    # 2. ML-defective path (Integrative Explanation Layer)
    elif isinstance(source, str) and source.startswith("ml_") and pred == "defective":
        print("Entering ML explanation block")
        explanation = _xai_shap_explain_one(code, model_confidence=defect_prob)
        top_tokens = explanation.get("top_tokens", []) if (explanation and isinstance(explanation, dict)) else []
        
        # Always provide the tokens key (even if empty) to maintain contract
        result["explanation_tokens"] = top_tokens
        
        # Synthesis Phase: Determine summary via LLM or Fallback
        summary = None
        if LLM_LAYER_ACTIVE:
            tokens_str = ", ".join([t["token"] for t in top_tokens[:5]]) if top_tokens else "no specific tokens"
            prompt = LLM_EXPLANATION_PROMPT_TEMPLATE.format(
                tokens=tokens_str,
                prob=f"{defect_prob:.2f}",
                code=code
            )
            print("Gemini call attempted")
            raw_res = call_gemini(prompt)
            summary = validate_gemini_response(raw_res)
            if summary:
                print("Gemini call succeeded")
        
        # Robust Fallback: Triggered if LLM is disabled, timed out, or rejected
        if not summary:
            print("Fallback explanation used")
            summary = generate_fallback_summary(top_tokens, defect_prob)
            
        result["explanation_summary"] = summary

    # 3. ML-clean path
    # (By default, no rule_justification or explanation fields are added)

    # --- DEFENSIVE INVARIANT ENFORCEMENT ---
    if (
        isinstance(result.get("decision_source"), str)
        and result["decision_source"].startswith("ml_")
        and result.get("prediction") == "defective"
    ):
        if "explanation_summary" not in result:
            print("Invariant enforcement triggered: explanation fields missing")
            result["explanation_tokens"] = []
            result["explanation_summary"] = generate_fallback_summary(
                [],
                result.get("defect_probability", 0.0)
            )

    # --- DEBUG: Final Payload ---
    print("FINAL RESULT:", result)

    return result

# ------------------------------------------------------------
# API ENDPOINT
# ------------------------------------------------------------
@app.post("/predict")
def predict(data: CodeInput):
    return classify_one(data.code)
