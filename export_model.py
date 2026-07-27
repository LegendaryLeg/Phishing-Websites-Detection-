"""
Export the trained best_phishing_model.pkl to JavaScript via m2cgen.

Writes converted_model.js (raw m2cgen output, score renamed to rawScore)
and patches model.js with FEATURE_ORDER, CLASS_ORDER, and prediction helpers.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import m2cgen as m2c
import numpy as np

ROOT = Path(__file__).resolve().parent
PKL_PATH = ROOT / "best_phishing_model.pkl"
CONVERTED_JS_PATH = ROOT / "converted_model.js"
MODEL_JS_PATH = ROOT / "model.js"


def unwrap_estimator(model):
    """Return the estimator m2cgen should export."""
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        return model
    return model


def get_classes(model) -> list:
    if hasattr(model, "named_steps") and "clf" in getattr(model, "named_steps", {}):
        raw = model.named_steps["clf"].classes_
    elif hasattr(model, "classes_"):
        raw = model.classes_
    else:
        raise AttributeError("Could not find classes_ on model")
    # Normalize numpy scalar types for clean JS / logging.
    out = []
    for c in raw:
        if isinstance(c, (np.integer, int)):
            out.append(int(c))
        elif isinstance(c, (np.floating, float)):
            out.append(float(c))
        else:
            out.append(c)
    return out


def rename_score_to_rawscore(js_body: str) -> str:
    js_body = js_body.strip()
    if js_body.startswith("function score("):
        return "function rawScore(" + js_body[len("function score(") :]
    if "function score(" in js_body:
        return js_body.replace("function score(", "function rawScore(", 1)
    return f"function rawScore(input) {{\n  return (function() {{\n{js_body}\n  }})();\n}}"


def build_model_js(
    features: list[str],
    classes: list,
    js_body: str,
    model_name: str,
    metrics: dict,
) -> str:
    feature_order_literal = ",\n  ".join(f'"{f}"' for f in features)
    # Keep numeric classes as numbers; stringify anything else.
    class_literals = []
    for c in classes:
        if isinstance(c, (int, np.integer)):
            class_literals.append(str(int(c)))
        elif isinstance(c, float):
            class_literals.append(str(c))
        else:
            class_literals.append(f'"{c}"')
    class_order_literal = ", ".join(class_literals)

    metrics_comment = (
        f"// Exported model: {model_name}\n"
        f"// clf.classes_ = {list(classes)}\n"
        f"// Phishing recall={metrics.get('phishing_recall', float('nan')):.4f}, "
        f"phishing F1={metrics.get('phishing_f1', float('nan')):.4f}, "
        f"macro-F1={metrics.get('holdout_macro_f1', float('nan')):.4f}\n"
    )

    return f"""{metrics_comment}// Must match train_compare_models.py DEPLOYABLE_FEATURES and popup.js.
const FEATURE_ORDER = [
  {feature_order_literal}
];

// Must match clf.classes_ from the exported sklearn model (do not assume order).
const CLASS_ORDER = [{class_order_literal}];

/**
 * Convert a feature object (keys = FEATURE_ORDER names) into the dense
 * vector expected by rawScore.
 */
function featuresToVector(featureObj) {{
  return FEATURE_ORDER.map((name) => {{
    const v = featureObj[name];
    if (v === undefined || v === null || Number.isNaN(Number(v))) {{
      return 0;
    }}
    return Number(v);
  }});
}}

/**
 * Raw m2cgen model output (probability / vote vector for classifiers).
 */
{js_body}

function clamp01(x) {{
  return Math.max(0, Math.min(1, x));
}}

/**
 * P(phishing) from m2cgen output. CLASS_ORDER is [0, 1] => index 0 is phishing.
 */
function modelPhishingLikelihood(pred) {{
  if (Array.isArray(pred)) {{
    const phishIdx = CLASS_ORDER.indexOf(0);
    const idx = phishIdx >= 0 ? phishIdx : 0;
    const total = pred.reduce((a, b) => a + Math.max(0, Number(b) || 0), 0) || 1;
    return clamp01((Number(pred[idx]) || 0) / total);
  }}
  // Scalar: treat as P(legitimate) if >=0.5 style score for class 1
  return clamp01(1 - Number(pred));
}}

/**
 * Lightweight modern-web prior. Used to soften overconfident tree leaves on SPAs.
 */
function heuristicPhishingLikelihood(f) {{
  let score = 0.35;
  if (f.IsHTTPS === 0) score += 0.25;
  else score -= 0.12;
  if (f.IsDomainIP === 1) score += 0.3;
  if (f.HasObfuscation === 1) score += 0.15;
  if ((f.DegitRatioInURL || 0) > 0.35) score += 0.08;
  if (f.HasPasswordField === 1 && f.HasExternalFormSubmit === 1) score += 0.2;
  if (f.HasDescription === 1) score -= 0.05;
  if (f.HasFavicon === 1) score -= 0.05;
  if (f.HasTitle === 1) score -= 0.03;
  if (f.IsResponsive === 1) score -= 0.03;
  if (f.HasCopyrightInfo === 1) score -= 0.04;
  if (f.HasSocialNet === 1) score -= 0.04;
  if ((f.NoOfImage || 0) >= 5) score -= 0.03;
  return clamp01(score);
}}

function combinePhishingLikelihood(modelP, heurP, f) {{
  let combined = 0.55 * modelP + 0.45 * heurP;
  const modernLegitSignals =
    (f.IsHTTPS === 1 ? 1 : 0) +
    (f.HasFavicon === 1 ? 1 : 0) +
    (f.HasTitle === 1 ? 1 : 0) +
    (f.IsResponsive === 1 ? 1 : 0) +
    (f.HasDescription === 1 ? 1 : 0);
  // Dampen extreme model phishing scores on pages that look like modern apps.
  if (modernLegitSignals >= 4 && modelP >= 0.8) {{
    combined = Math.min(combined, 0.45 + 0.25 * heurP);
  }}
  return clamp01(combined);
}}

function riskLevelFromLikelihood(p) {{
  if (p >= 0.7) return "High";
  if (p >= 0.4) return "Medium";
  return "Low";
}}

function scoreToLabel(pred) {{
  if (Array.isArray(pred)) {{
    let bestIdx = 0;
    for (let i = 1; i < pred.length; i += 1) {{
      if (pred[i] > pred[bestIdx]) bestIdx = i;
    }}
    return CLASS_ORDER[bestIdx];
  }}
  const positive = CLASS_ORDER[CLASS_ORDER.length - 1];
  const negative = CLASS_ORDER[0];
  return Number(pred) >= 0.5 ? positive : negative;
}}

/**
 * Primary API for the extension UI: returns phishing likelihood (0..1), not a hard verdict.
 */
function predictFromFeatures(featureObj) {{
  const vector = featuresToVector(featureObj);
  const pred = rawScore(vector);
  const modelP = modelPhishingLikelihood(pred);
  const heurP = heuristicPhishingLikelihood(featureObj || {{}});
  const phishingLikelihood = combinePhishingLikelihood(modelP, heurP, featureObj || {{}});
  const riskLevel = riskLevelFromLikelihood(phishingLikelihood);
  const label = scoreToLabel(pred); // kept for debugging / optional use
  return {{
    phishingLikelihood,
    phishingPercent: Math.round(phishingLikelihood * 100),
    riskLevel,
    modelPhishingLikelihood: modelP,
    heuristicPhishingLikelihood: heurP,
    label: Number(label),
    labelName: riskLevel === "High" ? "High phishing risk" : riskLevel === "Medium" ? "Medium phishing risk" : "Low phishing risk",
    raw: pred,
    features: featureObj,
    vector,
  }};
}}

if (typeof window !== "undefined") {{
  window.FEATURE_ORDER = FEATURE_ORDER;
  window.CLASS_ORDER = CLASS_ORDER;
  window.rawScore = rawScore;
  window.scoreToLabel = scoreToLabel;
  window.predictFromFeatures = predictFromFeatures;
  window.featuresToVector = featuresToVector;
  window.modelPhishingLikelihood = modelPhishingLikelihood;
  window.heuristicPhishingLikelihood = heuristicPhishingLikelihood;
}}

if (typeof module !== "undefined" && module.exports) {{
  module.exports = {{
    FEATURE_ORDER,
    CLASS_ORDER,
    rawScore,
    scoreToLabel,
    predictFromFeatures,
    featuresToVector,
    modelPhishingLikelihood,
    heuristicPhishingLikelihood,
    riskLevelFromLikelihood,
  }};
}}
"""


def main() -> None:
    if not PKL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {PKL_PATH.name}. Run train_compare_models.py first."
        )

    with open(PKL_PATH, "rb") as f:
        payload = pickle.load(f)

    model = unwrap_estimator(payload["model"])
    features = payload["features"]
    model_name = payload.get("model_name", type(model).__name__)
    metrics = payload.get("metrics", {})
    classes = get_classes(model)

    print(f"Exporting model: {model_name}")
    print(f"clf.classes_ = {classes}")
    print(f"Features ({len(features)}): {features}")

    try:
        js_body = m2c.export_to_javascript(model)
    except Exception as exc:
        if hasattr(model, "named_steps"):
            print(f"Pipeline export failed ({exc}); exporting inner clf only.")
            js_body = m2c.export_to_javascript(model.named_steps["clf"])
            classes = list(model.named_steps["clf"].classes_)
            print(f"clf.classes_ (inner) = {classes}")
        else:
            raise

    raw_js = rename_score_to_rawscore(js_body)
    CONVERTED_JS_PATH.write_text(raw_js + "\n", encoding="utf-8")
    size = CONVERTED_JS_PATH.stat().st_size
    print(f"Wrote {CONVERTED_JS_PATH.name} ({size} bytes)")
    if size > 500 * 1024:
        print(
            f"WARNING: converted_model.js is {size / 1024:.1f} KB (>500KB). "
            "Large JS bundle tradeoff — proceed with integration anyway."
        )

    model_js = build_model_js(features, classes, raw_js, model_name, metrics)
    MODEL_JS_PATH.write_text(model_js, encoding="utf-8")
    print(f"Wrote {MODEL_JS_PATH.name} ({MODEL_JS_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
