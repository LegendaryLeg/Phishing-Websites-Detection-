// Exported model: DecisionTree
// clf.classes_ = [0, 1]
// Phishing recall=0.9990, phishing F1=0.9990, macro-F1=0.9992
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
                        if (input[18] <= 103.0) {
                            var0 = [1.0, 0.0];
                        } else {
                            if (input[40] <= 2.5) {
                                var0 = [1.0, 0.0];
                            } else {
                                if (input[1] <= 11.5) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            }
                        }
                    }
                } else {
                    if (input[15] <= 1.5) {
                        if (input[18] <= 101.5) {
                            var0 = [1.0, 0.0];
                        } else {
                            if (input[17] <= 0.5) {
                                var0 = [1.0, 0.0];
                            } else {
                                if (input[0] <= 29.5) {
                                    var0 = [0.0, 1.0];
                                } else {
                                    var0 = [1.0, 0.0];
                                }
                            }
                        }
                    } else {
                        var0 = [1.0, 0.0];
                    }
                }
            } else {
                if (input[15] <= 2.5) {
                    if (input[23] <= 0.5) {
                        if (input[18] <= 258.5) {
                            if (input[27] <= 0.5) {
                                if (input[9] <= 0.5194999873638153) {
                                    if (input[17] <= 0.5) {
                                        var0 = [1.0, 0.0];
                                    } else {
                                        if (input[16] <= 0.05350000038743019) {
                                            if (input[40] <= 0.5) {
                                                if (input[37] <= 2.5) {
                                                    var0 = [1.0, 0.0];
                                                } else {
                                                    var0 = [0.0, 1.0];
                                                }
                                            } else {
                                                if (input[37] <= 0.5) {
                                                    var0 = [0.5, 0.5];
                                                } else {
                                                    var0 = [0.0, 1.0];
                                                }
                                            }
                                        } else {
                                            if (input[36] <= 0.5) {
                                                var0 = [1.0, 0.0];
                                            } else {
                                                if (input[4] <= 0.5) {
                                                    var0 = [1.0, 0.0];
                                                } else {
                                                    var0 = [0.3157894736842105, 0.6842105263157895];
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (input[35] <= 8.5) {
                                        if (input[37] <= 2.5) {
                                            var0 = [1.0, 0.0];
                                        } else {
                                            if (input[9] <= 0.5819999873638153) {
                                                if (input[1] <= 23.5) {
                                                    var0 = [0.9444444444444444, 0.05555555555555555];
                                                } else {
                                                    var0 = [0.0, 1.0];
                                                }
                                            } else {
                                                var0 = [1.0, 0.0];
                                            }
                                        }
                                    } else {
                                        if (input[9] <= 0.5819999873638153) {
                                            var0 = [0.0, 1.0];
                                        } else {
                                            if (input[39] <= 0.5) {
                                                var0 = [1.0, 0.0];
                                            } else {
                                                if (input[34] <= 0.5) {
                                                    var0 = [1.0, 0.0];
                                                } else {
                                                    var0 = [0.0, 1.0];
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (input[17] <= 0.5) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    if (input[11] <= 0.012000000104308128) {
                                        var0 = [0.0, 1.0];
                                    } else {
                                        if (input[9] <= 0.4519999921321869) {
                                            var0 = [0.0, 1.0];
                                        } else {
                                            var0 = [1.0, 0.0];
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[11] <= 0.009499999694526196) {
                                if (input[4] <= 0.5) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    if (input[17] <= 0.5) {
                                        var0 = [1.0, 0.0];
                                    } else {
                                        if (input[19] <= 64031.0) {
                                            if (input[9] <= 0.6759999990463257) {
                                                var0 = [0.0, 1.0];
                                            } else {
                                                if (input[37] <= 1.0) {
                                                    var0 = [1.0, 0.0];
                                                } else {
                                                    var0 = [0.0, 1.0];
                                                }
                                            }
                                        } else {
                                            if (input[18] <= 6002.0) {
                                                var0 = [1.0, 0.0];
                                            } else {
                                                var0 = [0.0, 1.0];
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (input[18] <= 812.0) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    if (input[28] <= 0.5) {
                                        var0 = [0.0, 1.0];
                                    } else {
                                        var0 = [1.0, 0.0];
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[17] <= 0.5) {
                            var0 = [1.0, 0.0];
                        } else {
                            if (input[4] <= 0.5) {
                                var0 = [1.0, 0.0];
                            } else {
                                if (input[9] <= 0.6755000054836273) {
                                    if (input[35] <= 1.5) {
                                        if (input[19] <= 3466.0) {
                                            if (input[11] <= 0.1784999966621399) {
                                                if (input[16] <= 0.02949999924749136) {
                                                    var0 = [0.16666666666666666, 0.8333333333333334];
                                                } else {
                                                    var0 = [0.0, 1.0];
                                                }
                                            } else {
                                                var0 = [1.0, 0.0];
                                            }
                                        } else {
                                            if (input[36] <= 0.5) {
                                                var0 = [1.0, 0.0];
                                            } else {
                                                var0 = [0.0, 1.0];
                                            }
                                        }
                                    } else {
                                        var0 = [0.0, 1.0];
                                    }
                                } else {
                                    if (input[18] <= 260.0) {
                                        var0 = [1.0, 0.0];
                                    } else {
                                        var0 = [0.0, 1.0];
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (input[0] <= 32.5) {
                        if (input[23] <= 0.5) {
                            if (input[18] <= 229.0) {
                                var0 = [1.0, 0.0];
                            } else {
                                if (input[4] <= 2.5) {
                                    if (input[27] <= 0.5) {
                                        if (input[37] <= 14.5) {
                                            var0 = [1.0, 0.0];
                                        } else {
                                            var0 = [0.0, 1.0];
                                        }
                                    } else {
                                        if (input[15] <= 3.5) {
                                            var0 = [0.0, 1.0];
                                        } else {
                                            var0 = [1.0, 0.0];
                                        }
                                    }
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[9] <= 0.48350000381469727) {
                                var0 = [0.0, 1.0];
                            } else {
                                if (input[18] <= 247.5) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    if (input[3] <= 2.5) {
                                        var0 = [0.0, 1.0];
                                    } else {
                                        var0 = [1.0, 0.0];
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[18] <= 321.5) {
                            var0 = [1.0, 0.0];
                        } else {
                            if (input[0] <= 41.5) {
                                if (input[35] <= 5.5) {
                                    if (input[18] <= 326.0) {
                                        var0 = [0.0, 1.0];
                                    } else {
                                        var0 = [1.0, 0.0];
                                    }
                                } else {
                                    if (input[18] <= 507.0) {
                                        if (input[38] <= 1.0) {
                                            var0 = [1.0, 0.0];
                                        } else {
                                            var0 = [0.0, 1.0];
                                        }
                                    } else {
                                        var0 = [0.0, 1.0];
                                    }
                                }
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
                if (input[17] <= 0.5) {
                    var0 = [1.0, 0.0];
                } else {
                    if (input[15] <= 2.5) {
                        if (input[18] <= 90.0) {
                            var0 = [1.0, 0.0];
                        } else {
                            var0 = [0.0, 1.0];
                        }
                    } else {
                        if (input[10] <= 0.5) {
                            if (input[9] <= 0.6175000071525574) {
                                var0 = [0.0, 1.0];
                            } else {
                                if (input[1] <= 33.0) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            }
                        } else {
                            if (input[38] <= 20.5) {
                                var0 = [1.0, 0.0];
                            } else {
                                var0 = [0.0, 1.0];
                            }
                        }
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
                        if (input[4] <= 0.5) {
                            var0 = [1.0, 0.0];
                        } else {
                            var0 = [0.0, 1.0];
                        }
                    } else {
                        if (input[38] <= 8.5) {
                            if (input[34] <= 0.5) {
                                if (input[27] <= 0.5) {
                                    if (input[4] <= 1.5) {
                                        if (input[9] <= 0.5054999887943268) {
                                            if (input[10] <= 0.5) {
                                                if (input[16] <= 0.08700000122189522) {
                                                    var0 = [0.0, 1.0];
                                                } else {
                                                    var0 = [1.0, 0.0];
                                                }
                                            } else {
                                                var0 = [1.0, 0.0];
                                            }
                                        } else {
                                            var0 = [1.0, 0.0];
                                        }
                                    } else {
                                        if (input[11] <= 0.03350000083446503) {
                                            var0 = [0.0, 1.0];
                                        } else {
                                            var0 = [1.0, 0.0];
                                        }
                                    }
                                } else {
                                    if (input[18] <= 143.5) {
                                        if (input[16] <= 0.07999999821186066) {
                                            var0 = [1.0, 0.0];
                                        } else {
                                            var0 = [0.0, 1.0];
                                        }
                                    } else {
                                        var0 = [0.0, 1.0];
                                    }
                                }
                            } else {
                                if (input[9] <= 0.5195000171661377) {
                                    if (input[16] <= 0.09750000014901161) {
                                        var0 = [0.0, 1.0];
                                    } else {
                                        if (input[35] <= 5.5) {
                                            var0 = [1.0, 0.0];
                                        } else {
                                            var0 = [0.0, 1.0];
                                        }
                                    }
                                } else {
                                    if (input[40] <= 8.5) {
                                        var0 = [1.0, 0.0];
                                    } else {
                                        if (input[18] <= 111.0) {
                                            var0 = [1.0, 0.0];
                                        } else {
                                            if (input[36] <= 3.5) {
                                                var0 = [0.0, 1.0];
                                            } else {
                                                var0 = [1.0, 0.0];
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[19] <= 6417.5) {
                                var0 = [0.0, 1.0];
                            } else {
                                var0 = [1.0, 0.0];
                            }
                        }
                    }
                } else {
                    if (input[4] <= 0.5) {
                        var0 = [1.0, 0.0];
                    } else {
                        if (input[35] <= 5.5) {
                            if (input[10] <= 1.5) {
                                if (input[9] <= 0.6935000121593475) {
                                    if (input[15] <= 2.5) {
                                        if (input[18] <= 178.5) {
                                            if (input[9] <= 0.5559999942779541) {
                                                var0 = [0.0, 1.0];
                                            } else {
                                                if (input[1] <= 20.5) {
                                                    var0 = [1.0, 0.0];
                                                } else {
                                                    var0 = [0.0, 1.0];
                                                }
                                            }
                                        } else {
                                            if (input[37] <= 0.5) {
                                                if (input[19] <= 8994.5) {
                                                    var0 = [0.0, 1.0];
                                                } else {
                                                    var0 = [1.0, 0.0];
                                                }
                                            } else {
                                                var0 = [0.0, 1.0];
                                            }
                                        }
                                    } else {
                                        if (input[3] <= 3.5) {
                                            if (input[19] <= 40039.0) {
                                                var0 = [0.0, 1.0];
                                            } else {
                                                var0 = [1.0, 0.0];
                                            }
                                        } else {
                                            var0 = [1.0, 0.0];
                                        }
                                    }
                                } else {
                                    if (input[16] <= 0.036999999545514584) {
                                        if (input[18] <= 290.0) {
                                            var0 = [1.0, 0.0];
                                        } else {
                                            var0 = [0.0, 1.0];
                                        }
                                    } else {
                                        var0 = [1.0, 0.0];
                                    }
                                }
                            } else {
                                if (input[15] <= 1.5) {
                                    var0 = [0.0, 1.0];
                                } else {
                                    if (input[8] <= 12.5) {
                                        if (input[16] <= 0.05650000087916851) {
                                            var0 = [1.0, 0.0];
                                        } else {
                                            var0 = [0.0, 1.0];
                                        }
                                    } else {
                                        if (input[40] <= 50.5) {
                                            var0 = [1.0, 0.0];
                                        } else {
                                            var0 = [0.0, 1.0];
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[19] <= 94885.0) {
                                if (input[18] <= 335.5) {
                                    if (input[19] <= 16626.0) {
                                        if (input[9] <= 0.6855000257492065) {
                                            var0 = [0.0, 1.0];
                                        } else {
                                            if (input[36] <= 0.5) {
                                                var0 = [1.0, 0.0];
                                            } else {
                                                var0 = [0.0, 1.0];
                                            }
                                        }
                                    } else {
                                        var0 = [1.0, 0.0];
                                    }
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            } else {
                                if (input[18] <= 2329.5) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    if (input[19] <= 923337.0) {
                                        var0 = [0.0, 1.0];
                                    } else {
                                        var0 = [1.0, 0.0];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (input[38] <= 29.5) {
                if (input[11] <= 0.004000000189989805) {
                    if (input[9] <= 0.6110000014305115) {
                        if (input[18] <= 174.0) {
                            if (input[8] <= 14.0) {
                                var0 = [0.0, 1.0];
                            } else {
                                var0 = [1.0, 0.0];
                            }
                        } else {
                            if (input[19] <= 33735.0) {
                                var0 = [0.0, 1.0];
                            } else {
                                var0 = [1.0, 0.0];
                            }
                        }
                    } else {
                        if (input[1] <= 35.5) {
                            var0 = [1.0, 0.0];
                        } else {
                            if (input[8] <= 31.5) {
                                if (input[36] <= 2.0) {
                                    var0 = [1.0, 0.0];
                                } else {
                                    var0 = [0.0, 1.0];
                                }
                            } else {
                                var0 = [1.0, 0.0];
                            }
                        }
                    }
                } else {
                    var0 = [1.0, 0.0];
                }
            } else {
                if (input[0] <= 56.5) {
                    if (input[19] <= 27017.0) {
                        var0 = [0.0, 1.0];
                    } else {
                        if (input[35] <= 862.5) {
                            var0 = [1.0, 0.0];
                        } else {
                            var0 = [0.0, 1.0];
                        }
                    }
                } else {
                    var0 = [1.0, 0.0];
                }
            }
        }
    }
    return var0;
}

/**
 * Map m2cgen output to a class label using CLASS_ORDER.
 * Tree models typically return [P(class0), P(class1), ...] aligned with CLASS_ORDER.
 */
function scoreToLabel(pred) {
  if (Array.isArray(pred)) {
    let bestIdx = 0;
    for (let i = 1; i < pred.length; i += 1) {
      if (pred[i] > pred[bestIdx]) bestIdx = i;
    }
    return CLASS_ORDER[bestIdx];
  }
  // Scalar fallback: treat as score for the positive class (last in CLASS_ORDER).
  const positive = CLASS_ORDER[CLASS_ORDER.length - 1];
  const negative = CLASS_ORDER[0];
  return Number(pred) >= 0.5 ? positive : negative;
}

function predictFromFeatures(featureObj) {
  const vector = featuresToVector(featureObj);
  const pred = rawScore(vector);
  const label = scoreToLabel(pred);
  const labelName = Number(label) === 1 ? "Legitimate" : "Phishing";
  return {
    label: Number(label),
    labelName,
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
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    FEATURE_ORDER,
    CLASS_ORDER,
    rawScore,
    scoreToLabel,
    predictFromFeatures,
    featuresToVector,
  };
}
