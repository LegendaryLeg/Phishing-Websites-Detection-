/*
 * background.js
 * -------------
 * Auto-scans pages, applies local hybrid risk (allowlist + rules + ML),
 * caches results per tab, and stores the user's personal trust list.
 */
importScripts("model.js", "featureExtractor.js", "allowlist.js", "localRisk.js");

const tabResults = new Map(); // tabId -> { features, prediction, url }
let userAllowlist = new Set();

async function loadUserAllowlist() {
  try {
    const data = await chrome.storage.local.get(["userAllowlist"]);
    const list = Array.isArray(data.userAllowlist) ? data.userAllowlist : [];
    userAllowlist = new Set(list.map((d) => String(d).toLowerCase()));
  } catch (e) {
    userAllowlist = new Set();
  }
}

async function saveUserAllowlist() {
  await chrome.storage.local.set({
    userAllowlist: Array.from(userAllowlist),
  });
}

loadUserAllowlist();

function isScannable(url) {
  return typeof url === "string" && /^https?:\/\//.test(url);
}

function setBadgeIdle(tabId, title) {
  chrome.action.setBadgeText({ tabId, text: "" });
  chrome.action.setTitle({ tabId, title: title || "Phishing Detector" });
}

function setBadgeScanning(tabId) {
  chrome.action.setBadgeText({ tabId, text: "…" });
  chrome.action.setBadgeBackgroundColor({ tabId, color: "#999999" });
  chrome.action.setTitle({ tabId, title: "Phishing Detector — scanning…" });
}

function setBadgeForPrediction(tabId, prediction) {
  const pct =
    prediction && typeof prediction.phishingPercent === "number"
      ? prediction.phishingPercent
      : Math.round((prediction?.phishingLikelihood || 0) * 100);
  const risk = prediction?.riskLevel || "Low";

  let color = "#34a853";
  if (risk === "Medium") color = "#b08900";
  if (risk === "High") color = "#e03131";

  const text = risk === "Low" ? "" : String(pct);
  chrome.action.setBadgeText({ tabId, text });
  chrome.action.setBadgeBackgroundColor({ tabId, color });

  const trusted = prediction?.allowlisted ? " · trusted domain" : "";
  chrome.action.setTitle({
    tabId,
    title: `Phishing likelihood: ${pct}% (${risk} risk)${trusted}`,
  });
}

async function scanTab(tabId, url) {
  if (!isScannable(url)) {
    tabResults.delete(tabId);
    setBadgeIdle(tabId, "Phishing Detector (this page can't be scanned)");
    return;
  }

  setBadgeScanning(tabId);
  try {
    const [injection] = await chrome.scripting.executeScript({
      target: { tabId },
      func: extractFeaturesFromPage,
    });

    const features = injection && injection.result ? injection.result : null;
    if (!features) throw new Error("Feature extraction returned nothing");

    const prediction = evaluateLocalRisk(url, features, userAllowlist);
    tabResults.set(tabId, { features, prediction, url });
    setBadgeForPrediction(tabId, prediction);
  } catch (err) {
    tabResults.delete(tabId);
    setBadgeIdle(tabId, "Phishing Detector (scan failed)");
  }
}

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete") {
    scanTab(tabId, tab.url);
  }
});

chrome.tabs.onActivated.addListener(({ tabId }) => {
  const cached = tabResults.get(tabId);
  if (cached) {
    setBadgeForPrediction(tabId, cached.prediction);
  } else {
    chrome.tabs.get(tabId, (tab) => {
      if (chrome.runtime.lastError || !tab) return;
      scanTab(tabId, tab.url);
    });
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  tabResults.delete(tabId);
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.userAllowlist) {
    const list = Array.isArray(changes.userAllowlist.newValue)
      ? changes.userAllowlist.newValue
      : [];
    userAllowlist = new Set(list.map((d) => String(d).toLowerCase()));
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || !message.type) return;

  if (message.type === "GET_CACHED_RESULT") {
    sendResponse(tabResults.get(message.tabId) || null);
    return;
  }

  if (message.type === "RESCAN_TAB") {
    scanTab(message.tabId, message.url).then(() => {
      sendResponse(tabResults.get(message.tabId) || null);
    });
    return true;
  }

  if (message.type === "TRUST_SITE") {
    const domain = getRegistrableDomain(message.url || message.domain || "");
    if (!domain) {
      sendResponse({ ok: false, error: "No domain" });
      return;
    }
    userAllowlist.add(domain);
    saveUserAllowlist().then(async () => {
      if (message.tabId && message.url) {
        await scanTab(message.tabId, message.url);
      }
      sendResponse({
        ok: true,
        domain,
        cached: message.tabId ? tabResults.get(message.tabId) || null : null,
      });
    });
    return true;
  }

  if (message.type === "UNTRUST_SITE") {
    const domain = getRegistrableDomain(message.url || message.domain || "");
    if (!domain) {
      sendResponse({ ok: false, error: "No domain" });
      return;
    }
    userAllowlist.delete(domain);
    saveUserAllowlist().then(async () => {
      if (message.tabId && message.url) {
        await scanTab(message.tabId, message.url);
      }
      sendResponse({
        ok: true,
        domain,
        cached: message.tabId ? tabResults.get(message.tabId) || null : null,
      });
    });
    return true;
  }
});
