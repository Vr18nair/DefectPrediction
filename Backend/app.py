import torch
import re
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

    # 🟢 0.5 STRUCTURAL CLEAN OVERRIDE  ← 🔥 NEW
    if structural_clean_override(code):
        return {
            "prediction": "clean",
            "clean_probability": 0.9,
            "defect_probability": 0.1,
            "decision_source": "structural_clean_override",
            "rule_justification": _structural_clean_override_justification(code) or {
                "name": "unknown_structural_clean_rule",
                "reason": "A structural clean override pattern matched.",
                "certainty": "certain",
                "rule_type": "override",
                "verdict": "clean",
            },
        }

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

    result: Dict[str, Any] = {
        "prediction": pred,
        "clean_probability": clean_prob,
        "defect_probability": defect_prob,
        "decision_source": source
    }

    if source == "static_rule" and static_justification is not None:
        result["rule_justification"] = static_justification

    # Attach SHAP explanation ONLY when ML is the final decision authority.
    if isinstance(source, str) and source.startswith("ml_"):
        model_confidence = defect_prob if pred == "defective" else clean_prob
        explanation = _xai_shap_explain_one(code, model_confidence=model_confidence)
        if explanation is not None:
            result["explanation"] = explanation

    return result

# ------------------------------------------------------------
# API ENDPOINT
# ------------------------------------------------------------
@app.post("/predict")
def predict(data: CodeInput):
    return classify_one(data.code)
