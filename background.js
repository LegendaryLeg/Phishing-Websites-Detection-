/*
 * background.js
 * -------------
 * Auto-scans each loaded page and caches { features, prediction } per tab.
 * The popup reads from this cache so it stays consistent with the badge.
 *
 * This file is intended to be registered as the MV3 service worker.
 */
importScripts("model.js", "featureExtractor.js");

// tabId -> { features, prediction }
const tabResults = new Map();

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
  // model.js predictFromFeatures: label 0=Phishing, 1=Legitimate
  const isPhishing = prediction && Number(prediction.label) === 0;
  if (isPhishing) {
    chrome.action.setBadgeText({ tabId, text: "!" });
    chrome.action.setBadgeBackgroundColor({ tabId, color: "#e03131" });
    chrome.action.setTitle({ tabId, title: "Likely Phishing" });
  } else {
    chrome.action.setBadgeText({ tabId, text: "" });
    chrome.action.setBadgeBackgroundColor({ tabId, color: "#34a853" });
    chrome.action.setTitle({ tabId, title: "Looks Legitimate" });
  }
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

    const prediction = predictFromFeatures(features);
    tabResults.set(tabId, { features, prediction });
    setBadgeForPrediction(tabId, prediction);
  } catch (err) {
    tabResults.delete(tabId);
    setBadgeIdle(tabId, "Phishing Detector (scan failed)");
  }
}

// Re-scan whenever a page finishes loading.
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete") {
    scanTab(tabId, tab.url);
  }
});

// Restore cached badge instantly when switching tabs.
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

// Popup asks for cached result instead of re-scanning everything.
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message && message.type === "GET_CACHED_RESULT") {
    sendResponse(tabResults.get(message.tabId) || null);
  }

  if (message && message.type === "RESCAN_TAB") {
    scanTab(message.tabId, message.url).then(() => {
      sendResponse(tabResults.get(message.tabId) || null);
    });
    return true; // keep sendResponse alive for async scan
  }
});

