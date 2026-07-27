/*
 * localRisk.js
 * ------------
 * Local hybrid risk engine:
 *   allowlist + lookalike/red-flag rules + ML likelihood + heuristics
 * Returns phishingLikelihood plus human-readable reasons.
 */

// Brands often impersonated. Compared against the registrable domain label.
const WATCHED_BRANDS = [
  "google", "gmail", "youtube", "microsoft", "outlook", "office", "apple", "icloud",
  "amazon", "paypal", "ebay", "facebook", "instagram", "whatsapp", "netflix", "spotify",
  "chase", "wellsfargo", "bankofamerica", "citibank", "americanexpress", "amex",
  "claude", "anthropic", "openai", "chatgpt", "github", "linkedin", "dropbox", "adobe",
  "walmart", "target", "coinbase", "binance", "kraken", "steam", "discord", "slack",
];

function levenshtein(a, b) {
  const s = String(a);
  const t = String(b);
  const m = s.length;
  const n = t.length;
  if (Math.abs(m - n) > 2) return 99;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = s[i - 1] === t[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[m][n];
}

function normalizeBrandish(label) {
  return String(label || "")
    .toLowerCase()
    .replace(/0/g, "o")
    .replace(/1/g, "l")
    .replace(/3/g, "e")
    .replace(/4/g, "a")
    .replace(/5/g, "s")
    .replace(/7/g, "t")
    .replace(/\$/g, "s")
    .replace(/[^a-z]/g, "");
}

function findLookalikeBrand(registrableDomain) {
  if (!registrableDomain) return null;
  const base = registrableDomain.split(".")[0];
  const candidates = [base, ...base.split(/[-_]/)].filter(Boolean);

  for (const candidate of candidates) {
    const raw = String(candidate || "")
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "");
    const norm = normalizeBrandish(candidate);
    if (!norm || norm.length < 4) continue;

    for (const brand of WATCHED_BRANDS) {
      // Exact clean brand token on its own isn't a lookalike by itself.
      if (raw === brand) continue;

      // Leetspeak / substitutions that normalize to the brand (paypa1 -> paypal)
      if (norm === brand && raw !== brand) return brand;

      const dist = levenshtein(norm, brand);
      if (dist > 0 && dist <= 2 && Math.abs(norm.length - brand.length) <= 2) {
        return brand;
      }
      if (norm.includes(brand) && norm.length <= brand.length + 10) {
        return brand;
      }
    }
  }
  return null;
}

function collectRuleSignals(url, features) {
  const reasons = [];
  let boost = 0;
  const f = features || {};
  const host = typeof getHostname === "function" ? getHostname(url) : "";
  const domain = typeof getRegistrableDomain === "function" ? getRegistrableDomain(url) : host;

  if (f.IsHTTPS === 0) {
    reasons.push("Page is not using HTTPS");
    boost += 0.18;
  }
  if (f.IsDomainIP === 1) {
    reasons.push("Website uses a raw IP address instead of a domain name");
    boost += 0.28;
  }
  if (f.HasObfuscation === 1) {
    reasons.push("URL contains encoded/obfuscated characters");
    boost += 0.12;
  }
  if ((f.NoOfSubDomain || 0) >= 4) {
    reasons.push("Unusually deep subdomain structure");
    boost += 0.1;
  }
  if ((f.DegitRatioInURL || 0) > 0.4) {
    reasons.push("URL contains a high ratio of digits");
    boost += 0.06;
  }
  if (f.HasPasswordField === 1 && f.HasExternalFormSubmit === 1) {
    reasons.push("Login/password form submits data to another domain");
    boost += 0.25;
  } else if (f.HasPasswordField === 1 && f.IsHTTPS === 0) {
    reasons.push("Password field on a non-HTTPS page");
    boost += 0.2;
  }

  const lookalike = findLookalikeBrand(domain);
  if (lookalike) {
    reasons.push(`Domain looks similar to “${lookalike}” (possible impersonation)`);
    boost += 0.3;
  }

  // Suspicious free-hosting style patterns in hostname
  if (/\.(tk|ml|ga|cf|gq|zip|mov)(\.|$)/i.test(host)) {
    reasons.push("Domain uses a frequently abused top-level domain");
    boost += 0.12;
  }

  return { reasons, boost, lookalikeBrand: lookalike, domain, host };
}

function clamp01Local(x) {
  return Math.max(0, Math.min(1, Number(x) || 0));
}

/**
 * Main local evaluator used by background/popup.
 * @param {string} url
 * @param {object} features
 * @param {Set|string[]} userAllowlist
 */
function evaluateLocalRisk(url, features, userAllowlist) {
  const base = predictFromFeatures(features || {});
  const rules = collectRuleSignals(url, features);
  const allowlisted =
    typeof isAllowlisted === "function" ? isAllowlisted(url, userAllowlist) : false;
  const userTrusted =
    typeof isUserAllowlisted === "function"
      ? isUserAllowlisted(url, userAllowlist)
      : false;
  const builtinTrusted =
    typeof isBuiltinAllowlisted === "function" ? isBuiltinAllowlisted(url) : false;

  let likelihood = clamp01Local(base.phishingLikelihood);
  const reasons = [];

  // Strong red flags first
  if (rules.boost > 0) {
    likelihood = clamp01Local(likelihood + rules.boost * 0.65);
    reasons.push(...rules.reasons);
  }

  if (allowlisted) {
    // Cap risk for trusted domains unless there are strong red flags.
    const strongFlags =
      (features && features.IsDomainIP === 1) ||
      (features && features.HasPasswordField === 1 && features.HasExternalFormSubmit === 1) ||
      !!rules.lookalikeBrand;

    if (!strongFlags) {
      likelihood = Math.min(likelihood, 0.22);
      reasons.unshift(
        userTrusted
          ? "Domain is on your personal trust list"
          : "Domain is on the built-in trusted sites list"
      );
    } else {
      reasons.unshift(
        "Domain is usually trusted, but strong warning signs were found"
      );
      likelihood = Math.max(likelihood, 0.55);
    }
  } else if (reasons.length === 0 && likelihood < 0.35) {
    reasons.push("No major local warning signs detected");
  } else if (reasons.length === 0 && likelihood >= 0.4) {
    reasons.push("Model/heuristic score is elevated for this page’s structure");
  }

  // Deduplicate reasons
  const uniqReasons = [];
  for (const r of reasons) {
    if (r && !uniqReasons.includes(r)) uniqReasons.push(r);
  }

  const riskLevel =
    likelihood >= 0.7 ? "High" : likelihood >= 0.4 ? "Medium" : "Low";

  return {
    ...base,
    phishingLikelihood: likelihood,
    phishingPercent: Math.round(likelihood * 100),
    riskLevel,
    labelName:
      riskLevel === "High"
        ? "High phishing risk"
        : riskLevel === "Medium"
          ? "Medium phishing risk"
          : "Low phishing risk",
    reasons: uniqReasons,
    allowlisted,
    userTrusted,
    builtinTrusted,
    domain: rules.domain,
    lookalikeBrand: rules.lookalikeBrand,
    url: url || "",
  };
}

if (typeof globalThis !== "undefined") {
  globalThis.evaluateLocalRisk = evaluateLocalRisk;
  globalThis.findLookalikeBrand = findLookalikeBrand;
  globalThis.collectRuleSignals = collectRuleSignals;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    evaluateLocalRisk,
    findLookalikeBrand,
    collectRuleSignals,
  };
}
