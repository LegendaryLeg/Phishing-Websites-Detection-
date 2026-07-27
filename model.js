// Exported model: DecisionTree
// clf.classes_ = [0, 1]
// Phishing recall=0.9947, phishing F1=0.9953, macro-F1=0.9959
// Must match train_compare_models.py DEPLOYABLE_FEATURES and popup.js.
const FEATURE_ORDER = [
  "URLLength",
  "DomainLength",
  "IsDomainIP",
  "TLDLength",
  "NoOfSubDomain",
  "HasObfuscation",
  "NoOfObfuscatedChar",
  "ObfuscationRatio",
  "NoOfLettersInURL",
  "LetterRatioInURL",
  "NoOfDegitsInURL",
  "DegitRatioInURL",
  "NoOfEqualsInURL",
  "NoOfQMarkInURL",
  "NoOfAmpersandInURL",
  "NoOfOtherSpecialCharsInURL",
  "SpacialCharRatioInURL",
  "IsHTTPS",
  "LineOfCode",
  "LargestLineLength",
  "HasTitle",
  "HasFavicon",
  "IsResponsive",
  "HasDescription",
  "NoOfPopup",
  "NoOfiFrame",
  "HasExternalFormSubmit",
  "HasSocialNet",
  "HasSubmitButton",
  "HasHiddenFields",
  "HasPasswordField",
  "Bank",
  "Pay",
  "Crypto",
  "HasCopyrightInfo",
  "NoOfImage",
  "NoOfCSS",
  "NoOfJS",
  "NoOfSelfRef",
  "NoOfEmptyRef",
  "NoOfExternalRef"
];

// Must match clf.classes_ from the exported sklearn model (do not assume order).
const CLASS_ORDER = [0, 1];

/**
 * Convert a feature object (keys = FEATURE_ORDER names) into the dense
 * vector expected by rawScore.
 */
function featuresToVector(featureObj) {
  return FEATURE_ORDER.map((name) => {
    const v = featureObj[name];
    if (v === undefined || v === null || Number.isNaN(Number(v))) {
      return 0;
    }
    return Number(v);
  });
}

/**
 * Raw m2cgen model output (probability / vote vector for classifiers).
 */
function rawScore(input) {
    var var0;
    if (input[40] <= 4.5) {
        if (input[38] <= 4.5) {
            if (input[18] <= 108.5) {
                if (input[35] <= 1.5) {
                    if (input[35] <= 0.5) {
                        var0 = [1.0, 0.0];
                    } else {
                        if (input[18] <= 88.5) {
                            var0 = [1.0, 0.0];
                        } else {
                            var0 = [0.9833333333333333, 0.016666666666666666];
                        }
                    }
                } else {
                    if (input[16] <= 0.05049999989569187) {
                        var0 = [0.9259259259259259, 0.07407407407407407];
                    } else {
                        var0 = [1.0, 0.0];
                    }
                }
            } else {
                if (input[15] <= 2.5) {
                    if (input[23] <= 0.5) {
                        if (input[18] <= 258.5) {
                            if (input[9] <= 0.5194999873638153) {
                                if (input[17] <= 0.5) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    var0 = [0.4444444444444444, 0.5555555555555556];
                                }
                            } else {
                                if (input[16] <= 0.03749999962747097) {
                                    var0 = [0.8940397350993378, 0.10596026490066225];
                                } else {
                                    if (input[18] <= 197.5) {
                                        var0 = [1.0, 0.0];
                                    } else {
                                        var0 = [0.9919354838709677, 0.008064516129032258];
                                    }
                                }
                            }
                        } else {
                            if (input[9] <= 0.5689999759197235) {
                                if (input[35] <= 8.5) {
                                    var0 = [0.1597222222222222, 0.8402777777777778];
                                } else {
                                    var0 = [0.006756756756756757, 0.9932432432432432];
                                }
                            } else {
                                var0 = [0.6363636363636364, 0.36363636363636365];
                            }
                        }
                    } else {
                        if (input[35] <= 3.5) {
                            if (input[9] <= 0.5194999873638153) {
                                if (input[18] <= 243.0) {
                                    var0 = [0.16901408450704225, 0.8309859154929577];
                                } else {
                                    var0 = [0.040983606557377046, 0.9590163934426229];
                                }
                            } else {
                                var0 = [0.5529411764705883, 0.4470588235294118];
                            }
                        } else {
                            if (input[35] <= 9.5) {
                                if (input[19] <= 336.5) {
                                    var0 = [0.15702479338842976, 0.8429752066115702];
                                } else {
                                    var0 = [0.017167381974248927, 0.9828326180257511];
                                }
                            } else {
                                if (input[18] <= 243.0) {
                                    var0 = [0.008333333333333333, 0.9916666666666667];
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                } else {
                    if (input[0] <= 32.5) {
                        if (input[18] <= 229.0) {
                            if (input[35] <= 2.5) {
                                var0 = [1.0, 0.0];
                            } else {
                                var0 = [0.9712230215827338, 0.02877697841726619];
                            }
                        } else {
                            var0 = [0.845679012345679, 0.15432098765432098];
                        }
                    } else {
                        if (input[18] <= 321.5) {
                            var0 = [1.0, 0.0];
                        } else {
                            if (input[0] <= 43.5) {
                                var0 = [0.9375, 0.0625];
                            } else {
                                var0 = [1.0, 0.0];
                            }
                        }
                    }
                }
            }
        } else {
            if (input[4] <= 0.5) {
                var0 = [1.0, 0.0];
            } else {
                if (input[18] <= 128.5) {
                    var0 = [0.59375, 0.40625];
                } else {
                    if (input[11] <= 0.012000000104308128) {
                        if (input[38] <= 9.5) {
                            if (input[1] <= 18.5) {
                                var0 = [0.0, 1.0];
                            } else {
                                var0 = [0.07746478873239436, 0.9225352112676056];
                            }
                        } else {
                            if (input[19] <= 10196.0) {
                                var0 = [0.0, 1.0];
                            } else {
                                var0 = [0.016666666666666666, 0.9833333333333333];
                            }
                        }
                    } else {
                        var0 = [0.2786885245901639, 0.7213114754098361];
                    }
                }
            }
        }
    } else {
        if (input[15] <= 3.5) {
            if (input[17] <= 0.5) {
                var0 = [1.0, 0.0];
            } else {
                if (input[18] <= 156.5) {
                    if (input[15] <= 1.5) {
                        if (input[18] <= 119.5) {
                            var0 = [0.06349206349206349, 0.9365079365079365];
                        } else {
                            if (input[9] <= 0.48999999463558197) {
                                var0 = [0.0, 1.0];
                            } else {
                                var0 = [0.012738853503184714, 0.9872611464968153];
                            }
                        }
                    } else {
                        if (input[9] <= 0.5764999985694885) {
                            var0 = [0.4943181818181818, 0.5056818181818182];
                        } else {
                            var0 = [0.9868421052631579, 0.013157894736842105];
                        }
                    }
                } else {
                    if (input[35] <= 5.5) {
                        if (input[9] <= 0.62950000166893) {
                            if (input[11] <= 0.06149999983608723) {
                                if (input[16] <= 0.07600000128149986) {
                                    if (input[9] <= 0.5689999759197235) {
                                        var0 = [0.0, 1.0];
                                    } else {
                                        if (input[19] <= 8966.5) {
                                            var0 = [0.0, 1.0];
                                        } else {
                                            var0 = [0.014814814814814815, 0.9851851851851852];
                                        }
                                    }
                                } else {
                                    if (input[9] <= 0.507999986410141) {
                                        var0 = [0.0, 1.0];
                                    } else {
                                        var0 = [0.16666666666666666, 0.8333333333333334];
                                    }
                                }
                            } else {
                                var0 = [0.2746478873239437, 0.7253521126760564];
                            }
                        } else {
                            if (input[15] <= 1.5) {
                                var0 = [0.005847953216374269, 0.9941520467836257];
                            } else {
                                var0 = [0.725, 0.275];
                            }
                        }
                    } else {
                        if (input[19] <= 27679.5) {
                            if (input[9] <= 0.6855000257492065) {
                                if (input[18] <= 443.5) {
                                    if (input[11] <= 0.08650000020861626) {
                                        if (input[19] <= 8039.0) {
                                            var0 = [0.0, 1.0];
                                        } else {
                                            var0 = [0.001841620626151013, 0.998158379373849];
                                        }
                                    } else {
                                        var0 = [0.01652892561983471, 0.9834710743801653];
                                    }
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            } else {
                                var0 = [0.027649769585253458, 0.9723502304147466];
                            }
                        } else {
                            if (input[18] <= 1289.5) {
                                var0 = [0.19166666666666668, 0.8083333333333333];
                            } else {
                                if (input[19] <= 134935.5) {
                                    var0 = [0.0, 1.0];
                                } else {
                                    var0 = [0.01652892561983471, 0.9834710743801653];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (input[38] <= 29.5) {
                if (input[10] <= 0.5) {
                    if (input[9] <= 0.6620000004768372) {
                        var0 = [0.5555555555555556, 0.4444444444444444];
                    } else {
                        var0 = [1.0, 0.0];
                    }
                } else {
                    var0 = [1.0, 0.0];
                }
            } else {
                var0 = [0.11814345991561181, 0.8818565400843882];
            }
        }
    }
    return var0;
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

/**
 * P(phishing) from m2cgen output. CLASS_ORDER is [0, 1] => index 0 is phishing.
 */
function modelPhishingLikelihood(pred) {
  if (Array.isArray(pred)) {
    const phishIdx = CLASS_ORDER.indexOf(0);
    const idx = phishIdx >= 0 ? phishIdx : 0;
    const total = pred.reduce((a, b) => a + Math.max(0, Number(b) || 0), 0) || 1;
    return clamp01((Number(pred[idx]) || 0) / total);
  }
  // Scalar: treat as P(legitimate) if >=0.5 style score for class 1
  return clamp01(1 - Number(pred));
}

/**
 * Lightweight modern-web prior. Used to soften overconfident tree leaves on SPAs.
 */
function heuristicPhishingLikelihood(f) {
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
}

function combinePhishingLikelihood(modelP, heurP, f) {
  let combined = 0.55 * modelP + 0.45 * heurP;
  const modernLegitSignals =
    (f.IsHTTPS === 1 ? 1 : 0) +
    (f.HasFavicon === 1 ? 1 : 0) +
    (f.HasTitle === 1 ? 1 : 0) +
    (f.IsResponsive === 1 ? 1 : 0) +
    (f.HasDescription === 1 ? 1 : 0);
  // Dampen extreme model phishing scores on pages that look like modern apps.
  if (modernLegitSignals >= 4 && modelP >= 0.8) {
    combined = Math.min(combined, 0.45 + 0.25 * heurP);
  }
  return clamp01(combined);
}

function riskLevelFromLikelihood(p) {
  if (p >= 0.7) return "High";
  if (p >= 0.4) return "Medium";
  return "Low";
}

function scoreToLabel(pred) {
  if (Array.isArray(pred)) {
    let bestIdx = 0;
    for (let i = 1; i < pred.length; i += 1) {
      if (pred[i] > pred[bestIdx]) bestIdx = i;
    }
    return CLASS_ORDER[bestIdx];
  }
  const positive = CLASS_ORDER[CLASS_ORDER.length - 1];
  const negative = CLASS_ORDER[0];
  return Number(pred) >= 0.5 ? positive : negative;
}

/**
 * Primary API for the extension UI: returns phishing likelihood (0..1), not a hard verdict.
 */
function predictFromFeatures(featureObj) {
  const vector = featuresToVector(featureObj);
  const pred = rawScore(vector);
  const modelP = modelPhishingLikelihood(pred);
  const heurP = heuristicPhishingLikelihood(featureObj || {});
  const phishingLikelihood = combinePhishingLikelihood(modelP, heurP, featureObj || {});
  const riskLevel = riskLevelFromLikelihood(phishingLikelihood);
  const label = scoreToLabel(pred); // kept for debugging / optional use
  return {
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
  };
}

if (typeof window !== "undefined") {
  window.FEATURE_ORDER = FEATURE_ORDER;
  window.CLASS_ORDER = CLASS_ORDER;
  window.rawScore = rawScore;
  window.scoreToLabel = scoreToLabel;
  window.predictFromFeatures = predictFromFeatures;
  window.featuresToVector = featuresToVector;
  window.modelPhishingLikelihood = modelPhishingLikelihood;
  window.heuristicPhishingLikelihood = heuristicPhishingLikelihood;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    FEATURE_ORDER,
    CLASS_ORDER,
    rawScore,
    scoreToLabel,
    predictFromFeatures,
    featuresToVector,
    modelPhishingLikelihood,
    heuristicPhishingLikelihood,
    riskLevelFromLikelihood,
  };
}
