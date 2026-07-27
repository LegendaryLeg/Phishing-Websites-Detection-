/**
 * Chrome extension popup logic.
 * extractFeaturesFromPage() must return the same feature names (and semantics)
 * as DEPLOYABLE_FEATURES / FEATURE_ORDER used by the trained model.
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
  const pathname = urlObj ? urlObj.pathname + urlObj.search : "";
  const urlForStats = href;
  const domain = hostname.replace(/\.$/, "");

  const letters = (urlForStats.match(/[A-Za-z]/g) || []).length;
  const digits = (urlForStats.match(/\d/g) || []).length;
  const equals = (urlForStats.match(/=/g) || []).length;
  const qmarks = (urlForStats.match(/\?/g) || []).length;
  const amps = (urlForStats.match(/&/g) || []).length;
  // "Other" special chars roughly matching PhiUSIIL-style URL punctuation.
  const otherSpecial = (urlForStats.match(/[^A-Za-z0-9?=&\-._~:/]/g) || []).length;
  const urlLength = urlForStats.length || 1;

  const labels = domain.split(".").filter(Boolean);
  const tld = labels.length ? labels[labels.length - 1] : "";
  const noOfSubDomain = Math.max(labels.length - 2, 0);

  // Simple obfuscation heuristic: percent-encoding / @ in host / hex escapes.
  const obfuscatedMatches = urlForStats.match(/%[0-9a-fA-F]{2}|@/g) || [];
  const noOfObfuscatedChar = obfuscatedMatches.join("").replace(/@/g, "@").length;
  const hasObfuscation = obfuscatedMatches.length > 0 ? 1 : 0;
  const obfuscationRatio = noOfObfuscatedChar / urlLength;

  const isDomainIP = /^(\d{1,3}\.){3}\d{1,3}$/.test(domain) || domain.includes(":") ? 1 : 0;

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

  const textLower = (document.body && document.body.innerText
    ? document.body.innerText
    : ""
  ).toLowerCase();

  const socialHosts = ["facebook.com", "twitter.com", "instagram.com", "linkedin.com", "x.com"];
  const anchors = Array.from(document.querySelectorAll("a[href]"));
  const hasSocialNet = anchors.some((a) =>
    socialHosts.some((s) => (a.getAttribute("href") || "").toLowerCase().includes(s))
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
      textLower.includes("copyright") || html.includes("©") || html.includes("&copy;")
        ? 1
        : 0,
    NoOfImage: document.querySelectorAll("img").length,
    NoOfCSS: document.querySelectorAll('link[rel="stylesheet"], style').length,
    NoOfJS: document.querySelectorAll("script").length,
    NoOfSelfRef: selfRefs,
    NoOfEmptyRef: emptyRefs,
    NoOfExternalRef: externalRefs
  };
}

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

function renderFeatures(features) {
  const pre = document.getElementById("features");
  if (!pre) return;
  pre.textContent = JSON.stringify(features, null, 2);
}

async function analyzeActiveTab() {
  // Preferred path: ask background.js for cached result / trigger a re-scan.
  // Fallback path: run the old in-popup executeScript-based extraction.
  await analyzeActiveTabViaBackgroundOrFallback(false);
}

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("analyze");
  if (btn) btn.addEventListener("click", analyzeActiveTab);
  loadCachedResult();
});

function applyCachedResult(cached, tabUrl) {
  if (!cached) {
    setStatus("Not scanned yet.", "error");
    const urlEl = document.getElementById("url");
    if (urlEl && tabUrl) urlEl.textContent = tabUrl;
    return;
  }

  const { features, prediction } = cached;
  if (features) renderFeatures(features);

  const cls = prediction && prediction.label === 1 ? "legit" : "phish";
  const labelName = prediction ? prediction.labelName : "Unknown";
  setStatus(`${labelName}`, cls);

  const urlEl = document.getElementById("url");
  if (urlEl && tabUrl) urlEl.textContent = tabUrl;
}

async function loadCachedResult() {
  const tab = await getActiveTab();
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
    // background.js not registered/available yet => fall back to manual scan
    await analyzeActiveTabViaBackgroundOrFallback(true);
  }
}

async function analyzeActiveTabViaBackgroundOrFallback(onlyOnFallback) {
  const tab = await getActiveTab();
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
    setStatus("Rescanning (background)…", "pending");
    try {
      const cached = await chrome.runtime.sendMessage({
        type: "RESCAN_TAB",
        tabId: tab.id,
        url: tabUrl,
      });
      applyCachedResult(cached, tabUrl);
      return;
    } catch (e) {
      // fall through to manual scan
    }
  }

  // Manual scan fallback (works even if background isn't enabled).
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
  const prediction = predictFromFeatures(features);
  const cls = prediction && prediction.label === 1 ? "legit" : "phish";
  setStatus(`${prediction.labelName}`, cls);

  const urlEl = document.getElementById("url");
  if (urlEl) urlEl.textContent = tabUrl;
}
