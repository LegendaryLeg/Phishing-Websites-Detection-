/*
 * featureExtractor.js
 * --------------------
 * Single source of truth for turning a live page into the feature vector
 * expected by `model.js` (m2cgen export).
 *
 * NOTE: This function is injected into the target page context via
 * `chrome.scripting.executeScript({ func: extractFeaturesFromPage })`, so
 * it must be fully self-contained.
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

  const textLower = (document.body && document.body.innerText ? document.body.innerText : "").toLowerCase();

  const socialHosts = [
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "linkedin.com",
    "x.com",
  ];
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
    Crypto:
      /bitcoin|crypto|ethereum|wallet|blockchain/i.test(textLower) ? 1 : 0,
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

