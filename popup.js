/**
 * popup.js — displays cached hybrid risk results and manages Trust this site.
 * Feature extraction for fallback scans still uses extractFeaturesFromPage below.
 */

function extractFeaturesFromPage() {
  const href = location.href || "";
  let urlObj;
  try {
    urlObj = new URL(href);
  } catch (e) {
    urlObj = null;
  }

  const hostname = urlObj ? urlObj.hostname : "";
  const urlForStats = href;
  const domain = hostname.replace(/\.$/, "");

  const letters = (urlForStats.match(/[A-Za-z]/g) || []).length;
  const digits = (urlForStats.match(/\d/g) || []).length;
  const equals = (urlForStats.match(/=/g) || []).length;
  const qmarks = (urlForStats.match(/\?/g) || []).length;
  const amps = (urlForStats.match(/&/g) || []).length;
  const otherSpecial = (urlForStats.match(/[^A-Za-z0-9?=&\-._~:/]/g) || []).length;
  const urlLength = urlForStats.length || 1;

  const labels = domain.split(".").filter(Boolean);
  const tld = labels.length ? labels[labels.length - 1] : "";
  const noOfSubDomain = Math.max(labels.length - 2, 0);

  const obfuscatedMatches = urlForStats.match(/%[0-9a-fA-F]{2}|@/g) || [];
  const noOfObfuscatedChar = obfuscatedMatches.join("").replace(/@/g, "@").length;
  const hasObfuscation = obfuscatedMatches.length > 0 ? 1 : 0;
  const obfuscationRatio = noOfObfuscatedChar / urlLength;

  const isDomainIP =
    /^(\d{1,3}\.){3}\d{1,3}$/.test(domain) || domain.includes(":") ? 1 : 0;

  const html = document.documentElement
    ? document.documentElement.outerHTML
    : document.body
      ? document.body.innerHTML
      : "";
  const lines = html.split(/\r\n|\r|\n/);
  const lineOfCode = lines.length;
  let largestLineLength = 0;
  for (const line of lines) {
    if (line.length > largestLineLength) largestLineLength = line.length;
  }

  const textLower = (
    document.body && document.body.innerText ? document.body.innerText : ""
  ).toLowerCase();

  const socialHosts = [
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "linkedin.com",
    "x.com",
  ];
  const anchors = Array.from(document.querySelectorAll("a[href]"));
  const hasSocialNet = anchors.some((a) =>
    socialHosts.some((s) =>
      (a.getAttribute("href") || "").toLowerCase().includes(s)
    )
  )
    ? 1
    : 0;

  const forms = Array.from(document.querySelectorAll("form"));
  const hasExternalFormSubmit = forms.some((f) => {
    const action = f.getAttribute("action") || "";
    if (!action || action.startsWith("#") || action.startsWith("/")) return false;
    try {
      const actionHost = new URL(action, location.href).hostname;
      return actionHost && actionHost !== hostname;
    } catch (e) {
      return false;
    }
  })
    ? 1
    : 0;

  const selfRefs = anchors.filter((a) => {
    try {
      return new URL(a.href, location.href).hostname === hostname;
    } catch (e) {
      return false;
    }
  }).length;
  const emptyRefs = anchors.filter((a) => {
    const h = (a.getAttribute("href") || "").trim();
    return !h || h === "#" || h.toLowerCase().startsWith("javascript:void");
  }).length;
  const externalRefs = Math.max(anchors.length - selfRefs - emptyRefs, 0);

  return {
    URLLength: urlLength,
    DomainLength: domain.length,
    IsDomainIP: isDomainIP,
    TLDLength: tld.length,
    NoOfSubDomain: noOfSubDomain,
    HasObfuscation: hasObfuscation,
    NoOfObfuscatedChar: noOfObfuscatedChar,
    ObfuscationRatio: obfuscationRatio,
    NoOfLettersInURL: letters,
    LetterRatioInURL: letters / urlLength,
    NoOfDegitsInURL: digits,
    DegitRatioInURL: digits / urlLength,
    NoOfEqualsInURL: equals,
    NoOfQMarkInURL: qmarks,
    NoOfAmpersandInURL: amps,
    NoOfOtherSpecialCharsInURL: otherSpecial,
    SpacialCharRatioInURL: otherSpecial / urlLength,
    IsHTTPS: location.protocol === "https:" ? 1 : 0,
    LineOfCode: lineOfCode,
    LargestLineLength: largestLineLength,
    HasTitle: document.title && document.title.trim() ? 1 : 0,
    HasFavicon: document.querySelector('link[rel*="icon"]') ? 1 : 0,
    IsResponsive: document.querySelector('meta[name="viewport"]') ? 1 : 0,
    HasDescription: document.querySelector('meta[name="description"]') ? 1 : 0,
    NoOfPopup: (html.match(/window\.open\s*\(/g) || []).length,
    NoOfiFrame: document.querySelectorAll("iframe").length,
    HasExternalFormSubmit: hasExternalFormSubmit,
    HasSocialNet: hasSocialNet,
    HasSubmitButton: document.querySelector(
      'button[type="submit"], input[type="submit"]'
    )
      ? 1
      : 0,
    HasHiddenFields: document.querySelector('input[type="hidden"]') ? 1 : 0,
    HasPasswordField: document.querySelector('input[type="password"]') ? 1 : 0,
    Bank: /bank|banco|compte|routing/i.test(textLower) ? 1 : 0,
    Pay: /pay|payment|checkout|billing|paypal/i.test(textLower) ? 1 : 0,
    Crypto: /bitcoin|crypto|ethereum|wallet|blockchain/i.test(textLower) ? 1 : 0,
    HasCopyrightInfo:
      textLower.includes("copyright") ||
      html.includes("©") ||
      html.includes("&copy;")
        ? 1
        : 0,
    NoOfImage: document.querySelectorAll("img").length,
    NoOfCSS: document.querySelectorAll('link[rel="stylesheet"], style').length,
    NoOfJS: document.querySelectorAll("script").length,
    NoOfSelfRef: selfRefs,
    NoOfEmptyRef: emptyRefs,
    NoOfExternalRef: externalRefs,
  };
}

let currentTab = null;
let currentPrediction = null;

async function getActiveTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0];
}

function setStatus(text, className) {
  const el = document.getElementById("status");
  if (!el) return;
  el.textContent = text;
  el.className = className || "";
}

function setLikelihood(prediction) {
  const textEl = document.getElementById("likelihoodText");
  const bar = document.getElementById("bar");
  const fill = document.getElementById("barFill");
  if (!textEl || !bar || !fill) return;

  if (!prediction || prediction.phishingPercent == null) {
    textEl.textContent = "Phishing likelihood: —";
    fill.style.width = "0%";
    bar.className = "bar";
    return;
  }

  const pct = prediction.phishingPercent;
  const level = (prediction.riskLevel || "Low").toLowerCase();
  textEl.textContent = `Phishing likelihood: ${pct}%`;
  fill.style.width = `${pct}%`;
  bar.className = `bar ${level}`;
}

function setReasons(prediction) {
  const list = document.getElementById("reasons");
  if (!list) return;
  list.innerHTML = "";
  const reasons = (prediction && prediction.reasons) || [];
  for (const reason of reasons.slice(0, 5)) {
    const li = document.createElement("li");
    li.textContent = reason;
    list.appendChild(li);
  }
}

function setTrustUi(prediction) {
  const btn = document.getElementById("trustBtn");
  const note = document.getElementById("trustNote");
  if (!btn || !note) return;

  if (!prediction || !prediction.domain) {
    btn.disabled = true;
    btn.textContent = "Trust this site";
    note.textContent = "";
    return;
  }

  btn.disabled = false;
  if (prediction.userTrusted) {
    btn.textContent = "Remove trust";
    note.textContent = `Trusted by you: ${prediction.domain}`;
  } else if (prediction.builtinTrusted) {
    btn.textContent = "Trust this site";
    note.textContent = `Already on built-in allowlist: ${prediction.domain}`;
    btn.disabled = true;
  } else {
    btn.textContent = "Trust this site";
    note.textContent = `Not trusted yet: ${prediction.domain}`;
  }
}

function renderPrediction(prediction) {
  currentPrediction = prediction || null;
  if (!prediction) {
    setStatus("No prediction.", "error");
    setLikelihood(null);
    setReasons(null);
    setTrustUi(null);
    return;
  }
  const level = (prediction.riskLevel || "Low").toLowerCase();
  setStatus(`${prediction.riskLevel} risk`, level);
  setLikelihood(prediction);
  setReasons(prediction);
  setTrustUi(prediction);
}

function renderFeatures(features) {
  const pre = document.getElementById("features");
  if (!pre) return;
  pre.textContent = JSON.stringify(features, null, 2);
}

function applyCachedResult(cached, tabUrl) {
  if (!cached) {
    setStatus("Not scanned yet.", "error");
    setLikelihood(null);
    setReasons(null);
    setTrustUi(null);
    const urlEl = document.getElementById("url");
    if (urlEl && tabUrl) urlEl.textContent = tabUrl;
    return;
  }

  const { features, prediction } = cached;
  if (features) renderFeatures(features);
  renderPrediction(prediction);

  const urlEl = document.getElementById("url");
  if (urlEl && tabUrl) urlEl.textContent = tabUrl;
}

async function loadCachedResult() {
  const tab = await getActiveTab();
  currentTab = tab;
  if (!tab || !tab.id) {
    setStatus("No active tab.", "error");
    return;
  }
  const tabUrl = tab.url || "";
  if (!tabUrl || !(tabUrl.startsWith("http://") || tabUrl.startsWith("https://"))) {
    setStatus("Open an http(s) page to analyze.", "error");
    return;
  }

  setStatus("Loading cached result…", "pending");
  try {
    const cached = await chrome.runtime.sendMessage({
      type: "GET_CACHED_RESULT",
      tabId: tab.id,
    });
    applyCachedResult(cached, tabUrl);
  } catch (e) {
    await analyzeActiveTabViaBackgroundOrFallback(true);
  }
}

async function analyzeActiveTab() {
  await analyzeActiveTabViaBackgroundOrFallback(false);
}

async function analyzeActiveTabViaBackgroundOrFallback(onlyOnFallback) {
  const tab = await getActiveTab();
  currentTab = tab;
  if (!tab || !tab.id) {
    setStatus("No active tab.", "error");
    return;
  }
  const tabUrl = tab.url || "";
  if (!tabUrl || !(tabUrl.startsWith("http://") || tabUrl.startsWith("https://"))) {
    setStatus("Open an http(s) page to analyze.", "error");
    return;
  }

  if (!onlyOnFallback) {
    setStatus("Rescanning…", "pending");
    try {
      const cached = await chrome.runtime.sendMessage({
        type: "RESCAN_TAB",
        tabId: tab.id,
        url: tabUrl,
      });
      applyCachedResult(cached, tabUrl);
      return;
    } catch (e) {
      // fall through
    }
  }

  setStatus("Analyzing…", "pending");
  let results;
  try {
    results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractFeaturesFromPage,
    });
  } catch (err) {
    setStatus(`Could not access page: ${err.message}`, "error");
    return;
  }

  const features = results && results[0] && results[0].result;
  if (!features) {
    setStatus("Feature extraction returned nothing.", "error");
    return;
  }

  renderFeatures(features);
  let userList = [];
  try {
    const stored = await chrome.storage.local.get(["userAllowlist"]);
    userList = Array.isArray(stored.userAllowlist) ? stored.userAllowlist : [];
  } catch (e) {
    userList = [];
  }

  const prediction =
    typeof evaluateLocalRisk === "function"
      ? evaluateLocalRisk(tabUrl, features, userList)
      : predictFromFeatures(features);
  renderPrediction(prediction);

  const urlEl = document.getElementById("url");
  if (urlEl) urlEl.textContent = tabUrl;
}

async function toggleTrust() {
  if (!currentTab || !currentTab.id || !currentTab.url) return;
  const trusted = currentPrediction && currentPrediction.userTrusted;
  const type = trusted ? "UNTRUST_SITE" : "TRUST_SITE";
  setStatus(trusted ? "Removing trust…" : "Trusting site…", "pending");
  try {
    const resp = await chrome.runtime.sendMessage({
      type,
      tabId: currentTab.id,
      url: currentTab.url,
    });
    if (resp && resp.cached) {
      applyCachedResult(resp.cached, currentTab.url);
    } else {
      await analyzeActiveTab();
    }
  } catch (e) {
    setStatus("Could not update trust list.", "error");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const analyzeBtn = document.getElementById("analyze");
  if (analyzeBtn) analyzeBtn.addEventListener("click", analyzeActiveTab);
  const trustBtn = document.getElementById("trustBtn");
  if (trustBtn) trustBtn.addEventListener("click", toggleTrust);
  loadCachedResult();
});
