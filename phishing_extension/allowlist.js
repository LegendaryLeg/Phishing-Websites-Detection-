/*
 * allowlist.js
 * ------------
 * Built-in trusted registrable domains + helpers.
 * User-trusted domains live in chrome.storage (managed by background.js).
 */

// Curated everyday trusted domains (not a full Tranco dump — keep the extension small).
// Matching is on registrable domain (eTLD+1 style), so mail.google.com => google.com.
const BUILTIN_ALLOWLIST = [
  // AI / productivity
  "claude.ai", "anthropic.com", "openai.com", "chatgpt.com", "notion.so", "notion.com",
  "figma.com", "canva.com", "miro.com", "slack.com", "zoom.us", "zoom.com",
  "dropbox.com", "box.com", "evernote.com", "todoist.com", "asana.com", "trello.com",
  "atlassian.com", "jira.com", "bitbucket.org", "linear.app", "cursor.com", "cursor.sh",
  // Google
  "google.com", "gmail.com", "youtube.com", "youtu.be", "googleapis.com", "gstatic.com",
  "googleusercontent.com", "withgoogle.com", "chrome.com", "chromium.org", "blogger.com",
  "blogspot.com", "drive.google.com",
  // Microsoft / Apple / Meta
  "microsoft.com", "live.com", "outlook.com", "office.com", "office365.com", "onedrive.com",
  "sharepoint.com", "github.com", "github.io", "linkedin.com", "skype.com", "bing.com",
  "apple.com", "icloud.com", "me.com", "mzstatic.com",
  "facebook.com", "fb.com", "instagram.com", "whatsapp.com", "meta.com", "messenger.com",
  "threads.net",
  // Amazon / commerce / payments
  "amazon.com", "amazon.ca", "amazon.co.uk", "amazon.de", "aws.amazon.com", "amazonaws.com",
  "paypal.com", "stripe.com", "square.com", "shopify.com", "ebay.com", "etsy.com",
  "walmart.com", "target.com", "bestbuy.com", "aliexpress.com", "alibaba.com",
  // Social / media / news
  "x.com", "twitter.com", "reddit.com", "tiktok.com", "twitch.tv", "discord.com", "discord.gg",
  "spotify.com", "netflix.com", "hulu.com", "disneyplus.com", "vimeo.com", "pinterest.com",
  "medium.com", "substack.com", "wikipedia.org", "wikimedia.org", "nytimes.com", "bbc.com",
  "bbc.co.uk", "cnn.com", "theguardian.com",
  // Dev / cloud
  "stackoverflow.com", "stackexchange.com", "npmjs.com", "pypi.org", "python.org",
  "mozilla.org", "firefox.com", "cloudflare.com", "cloudflare.net", "vercel.com", "netlify.com",
  "heroku.com", "digitalocean.com", "gitlab.com", "sourceforge.net", "docker.com",
  "kubernetes.io", "ubuntu.com", "debian.org", "archlinux.org", "rust-lang.org",
  "huggingface.co", "kaggle.com", "colab.research.google.com",
  // Education / reference
  "edu", // special-cased below via TLD helper? skip - too broad
  "coursera.org", "udemy.com", "edx.org", "khanacademy.org", "mit.edu", "harvard.edu",
  "stanford.edu", "ox.ac.uk", "cam.ac.uk",
  // Finance (common)
  "chase.com", "bankofamerica.com", "wellsfargo.com", "capitalone.com", "americanexpress.com",
  "visa.com", "mastercard.com", "wise.com", "revolut.com",
  // Misc common
  "yahoo.com", "aol.com", "duckduckgo.com", "brave.com", "opera.com", "samsung.com",
  "intel.com", "nvidia.com", "adobe.com", "acrobat.com", "salesforce.com", "oracle.com",
  "ibm.com", "cisco.com", "wordpress.com", "wordpress.org", "wix.com", "squarespace.com",
  "godaddy.com", "namecheap.com", "cloud.google.com", "azure.com", "azure.microsoft.com",
].filter((d) => d !== "edu");

const MULTI_PART_SUFFIXES = [
  "co.uk", "org.uk", "ac.uk", "gov.uk", "co.jp", "co.kr", "co.in", "com.au", "net.au",
  "com.br", "com.mx", "co.nz", "co.za", "com.sg", "com.hk", "com.tw", "com.tr",
];

const BUILTIN_ALLOWLIST_SET = new Set(BUILTIN_ALLOWLIST.map((d) => d.toLowerCase()));

function getHostname(urlOrHost) {
  if (!urlOrHost) return "";
  try {
    if (urlOrHost.includes("://")) return new URL(urlOrHost).hostname.toLowerCase();
  } catch (e) {
    /* fall through */
  }
  return String(urlOrHost).toLowerCase().replace(/^\.+|\.+$/g, "");
}

/**
 * Approximate registrable domain (eTLD+1). Good enough for allowlisting common sites.
 */
function getRegistrableDomain(urlOrHost) {
  const host = getHostname(urlOrHost).replace(/\.$/, "");
  if (!host) return "";
  if (/^(\d{1,3}\.){3}\d{1,3}$/.test(host)) return host; // IP

  const labels = host.split(".").filter(Boolean);
  if (labels.length <= 2) return host;

  const lastTwo = labels.slice(-2).join(".");
  const lastThree = labels.slice(-3).join(".");
  if (MULTI_PART_SUFFIXES.includes(lastTwo) && labels.length >= 3) {
    return labels.slice(-3).join(".");
  }
  // handle foo.co.uk style already covered; also amazon.co.uk etc.
  if (MULTI_PART_SUFFIXES.some((s) => host.endsWith("." + s) || host === s)) {
    return lastThree;
  }
  return lastTwo;
}

function isBuiltinAllowlisted(urlOrHost) {
  const domain = getRegistrableDomain(urlOrHost);
  if (!domain) return false;
  if (BUILTIN_ALLOWLIST_SET.has(domain)) return true;
  // also allow exact host matches that were listed as full hosts (rare)
  return BUILTIN_ALLOWLIST_SET.has(getHostname(urlOrHost));
}

function isUserAllowlisted(urlOrHost, userAllowlist) {
  const domain = getRegistrableDomain(urlOrHost);
  if (!domain || !userAllowlist) return false;
  if (userAllowlist instanceof Set) return userAllowlist.has(domain);
  if (Array.isArray(userAllowlist)) return userAllowlist.includes(domain);
  return false;
}

function isAllowlisted(urlOrHost, userAllowlist) {
  return isBuiltinAllowlisted(urlOrHost) || isUserAllowlisted(urlOrHost, userAllowlist);
}

if (typeof globalThis !== "undefined") {
  globalThis.BUILTIN_ALLOWLIST = BUILTIN_ALLOWLIST;
  globalThis.getHostname = getHostname;
  globalThis.getRegistrableDomain = getRegistrableDomain;
  globalThis.isBuiltinAllowlisted = isBuiltinAllowlisted;
  globalThis.isUserAllowlisted = isUserAllowlisted;
  globalThis.isAllowlisted = isAllowlisted;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    BUILTIN_ALLOWLIST,
    getHostname,
    getRegistrableDomain,
    isBuiltinAllowlisted,
    isUserAllowlisted,
    isAllowlisted,
  };
}
