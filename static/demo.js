// In-browser fallback for the static demo (GitHub Pages).
// When no FastAPI backend answers /api/*, the same trained model runs here from
// model.json (written by export_web_model.py), so app.js works unchanged.
// The Groq second opinion needs a server-side key, so it is off in this mode.

(() => {
  const LABELS = ["LEGITIMATE", "PHISHING"];
  const NO_LLM = "LLM second opinion runs on the full server version only";

  // features.py, ported line for line.
  const URL_RE = /https?:\/\/\S+|www\.\S+/gi;
  const IP_URL_RE = /https?:\/\/\d{1,3}(?:\.\d{1,3}){3}/i;
  const SHORTENERS = ["bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly"];
  const FREEMAIL = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "mail.ru"];
  const URGENCY = ["urgent", "immediately", "verify", "suspended", "expire", "expires", "act now",
    "confirm", "password", "account", "click here", "final notice", "within 24 hours",
    "limited time", "security alert", "deactivated", "terminated", "winner", "prize"];

  // app.py SIGNAL_TEXT: [threshold, message]
  const SIGNAL_TEXT = {
    has_ip_url: [0, () => "Link points at a raw IP address instead of a domain"],
    has_shortener: [0, () => "Uses a URL shortener that hides the real destination"],
    n_urgency_terms: [1, (v) => `Urgency and credential language (${v} matches)`],
    uppercase_ratio: [0.15, (v) => `Heavy use of capitals (${Math.round(v * 100)}% of letters)`],
    n_exclamations: [2, (v) => `Excessive exclamation marks (${v})`],
    sender_is_freemail: [0, () => "Sent from a free mail provider, not an organisation domain"],
    subject_has_urgency: [0, () => "Subject line pressures the reader to act"],
    n_urls: [2, (v) => `Contains an unusual number of links (${v})`],
  };

  const count = (hay, needle) => hay.split(needle).length - 1;
  const round = (x, d) => Math.round(x * 10 ** d) / 10 ** d;

  function extractMetadata(text, subject, sender) {
    sender = sender.toLowerCase();
    const lowered = (text + " " + subject).toLowerCase();
    const letters = [...text].filter((c) => /\p{L}/u.test(c));
    return {
      n_urls: (text.match(URL_RE) || []).length,
      has_ip_url: +IP_URL_RE.test(text),
      has_shortener: +SHORTENERS.some((s) => lowered.includes(s)),
      n_urgency_terms: URGENCY.reduce((n, t) => n + count(lowered, t), 0),
      uppercase_ratio: letters.length ? round(letters.filter((c) => /\p{Lu}/u.test(c)).length / letters.length, 4) : 0,
      n_exclamations: count(text, "!"),
      sender_is_freemail: +FREEMAIL.some((d) => sender.endsWith("@" + d) || sender.includes(d)),
      subject_has_urgency: +URGENCY.some((t) => subject.toLowerCase().includes(t)),
      text_length: [...text].length,
    };
  }

  // scikit-learn TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True):
  // lowercase, tokens = runs of 2+ word characters, drop stop words, add bigrams,
  // tf -> 1 + ln(tf), multiply by idf, L2-normalise.
  const TOKEN_RE = /[\p{L}\p{M}\p{N}_]{2,}/gu;
  function tfidf(m, text) {
    const tokens = (text.toLowerCase().match(TOKEN_RE) || []).filter((t) => !m.stop.has(t));
    const counts = new Map();
    const add = (term) => {
      const i = m.index.get(term);
      if (i !== undefined) counts.set(i, (counts.get(i) || 0) + 1);
    };
    tokens.forEach((t, k) => {
      add(t);
      if (k + 1 < tokens.length) add(t + " " + tokens[k + 1]);
    });
    const row = [...counts].map(([i, tf]) => [i, (1 + Math.log(tf)) * m.idf[i]]);
    const norm = Math.hypot(...row.map(([, v]) => v)) || 1;
    return row.map(([i, v]) => [i, v / norm]);
  }

  // Captured before the override below, so loading model.json never loops back into it.
  const realFetch = window.fetch.bind(window);

  // model.json sits next to this script, wherever the site is mounted.
  const MODEL_URL = new URL("model.json", document.currentScript.src);
  let modelPromise;
  const loadModel = () =>
    (modelPromise ||= realFetch(MODEL_URL)
      .then((r) => r.json())
      .then((m) => ({ ...m, index: new Map(m.terms.map((t, i) => [t, i])), stop: new Set(m.stop_words) })));

  function classify(m, body, subject, sender) {
    const meta = extractMetadata(body, subject, sender);
    let z = m.intercept;
    for (const [i, v] of tfidf(m, body)) z += m.text_coef[i] * v;
    m.meta_columns.forEach((col, j) => {
      z += m.meta_coef[j] * (meta[col] * m.meta_scale[j] + m.meta_min[j]);
    });
    const p1 = 1 / (1 + Math.exp(-z));
    const label = p1 > 0.5 ? 1 : 0;
    const reasons = Object.entries(SIGNAL_TEXT)
      .filter(([key, [threshold]]) => meta[key] > threshold)
      .map(([key, [, message]]) => message(meta[key]));
    return {
      label: LABELS[label],
      confidence: round(label ? p1 : 1 - p1, 4),
      probabilities: { LEGITIMATE: round(1 - p1, 4), PHISHING: round(p1, 4) },
      signals: meta,
      reasons,
    };
  }

  const json = (data, status = 200) =>
    new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });

  async function local(path, init) {
    const m = await loadModel();
    if (path === "/api/info") {
      return json({ ...m.info, model: "TF-IDF + structural features + Logistic Regression",
        classes: LABELS, llm_available: false, llm_model: null, llm_note: NO_LLM });
    }
    const email = JSON.parse(init.body);
    const body = String(email.text || "").trim();
    if (body.length < 20) return json({ detail: "Please paste at least 20 characters of the email body." }, 400);
    const ml = classify(m, body, String(email.subject || ""), String(email.sender || ""));
    const llm = email.engine === "ml" ? null : { error: NO_LLM };
    return json({ label: ml.label, confidence: ml.confidence, engine: "ml", ml, llm, agreement: null });
  }

  let offline = location.hostname.endsWith("github.io");
  window.fetch = async (input, init = {}) => {
    const path = input instanceof Request ? input.url : String(input);
    if (!path.startsWith("/api/")) return realFetch(input, init);
    if (!offline) {
      try {
        const response = await realFetch(input, init);
        if (response.status !== 404 && response.status !== 405) return response;
      } catch {
        /* no backend reachable: fall through to the in-browser model */
      }
      offline = true;
    }
    return local(path, init);
  };
})();
