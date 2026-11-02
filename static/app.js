const $ = (id) => document.getElementById(id);

const SAMPLES = {
  legit: {
    subject: "Q3 planning notes",
    sender: "jane.doe@corp.example.com",
    text: "Hi team, attaching the agenda for Thursday. Please review the draft budget " +
          "section before the meeting and send corrections. The room booking has been " +
          "updated in the calendar invite. Thanks.",
  },
  phish: {
    subject: "URGENT: Your account has been suspended",
    sender: "security-alert@gmail.com",
    text: "Dear Customer, unusual activity was detected. Your account will be SUSPENDED " +
          "within 24 hours. Verify your password immediately at " +
          "http://192.168.4.11/secure/login to avoid permanent closure!!!",
  },
};

const SIGNAL_LABELS = {
  n_urls: "Links",
  has_ip_url: "Raw IP link",
  has_shortener: "URL shortener",
  n_urgency_terms: "Urgency terms",
  uppercase_ratio: "Uppercase ratio",
  n_exclamations: "Exclamation marks",
  sender_is_freemail: "Free-mail sender",
  subject_has_urgency: "Urgent subject",
  text_length: "Body length",
};

fetch("/api/info")
  .then((r) => r.json())
  .then((d) => {
    $("stat-acc").textContent = (d.accuracy * 100).toFixed(1) + "%";
    $("stat-fpr").textContent = (d.false_positive_rate * 100).toFixed(2) + "%";
    $("stat-n").textContent = d.training_samples.toLocaleString();
    setEngineAvailability(d.llm_available, d.llm_model, d.llm_note);
  })
  .catch(() => { $("stat-acc").textContent = "offline"; });

// The LLM options only work once GROQ_API_KEY is configured on the server.
function setEngineAvailability(available, model, offNote) {
  const note = $("engine-note");
  if (available) {
    note.textContent = `Groq model: ${model}`;
    return;
  }
  document.querySelectorAll('input[name="engine"]').forEach((input) => {
    if (input.value !== "ml") input.disabled = true;
  });
  document.querySelector('input[value="ml"]').checked = true;
  note.textContent = offNote || "Groq disabled — set GROQ_API_KEY to enable";
}

const engine = () => document.querySelector('input[name="engine"]:checked').value;

const text = $("text");
text.addEventListener("input", () => {
  $("count").textContent = `${text.value.length} characters · minimum 20`;
});

$("sample-legit").onclick = () => fillSample(SAMPLES.legit);
$("sample-phish").onclick = () => fillSample(SAMPLES.phish);

function fillSample(sample) {
  $("subject").value = sample.subject;
  $("sender").value = sample.sender;
  text.value = sample.text;
  text.dispatchEvent(new Event("input"));
  text.focus();
}

$("form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("submit");
  $("error").hidden = true;
  button.disabled = true;
  button.textContent = "Scanning…";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text.value,
        subject: $("subject").value,
        sender: $("sender").value,
        engine: engine(),
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Request failed.");
    render(data);
  } catch (err) {
    $("error").textContent = err.message;
    $("error").hidden = false;
    $("result").hidden = true;
  } finally {
    button.disabled = false;
    button.textContent = "Scan email";
  }
});

function render(data) {
  const badge = $("badge");
  badge.textContent = data.label;
  badge.className = "badge " + data.label.toLowerCase();

  $("conf").textContent = (data.confidence * 100).toFixed(1) + "%";
  $("bar").style.width = data.confidence * 100 + "%";

  renderModel(data.ml);
  renderLlm(data.llm, data.agreement);

  $("result").hidden = false;
  $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderModel(ml) {
  const shown = Boolean(ml);
  $("probs").hidden = !shown;
  $("ml-block").hidden = !shown;
  if (!shown) return;

  $("p-legit").textContent = (ml.probabilities.LEGITIMATE * 100).toFixed(1) + "%";
  $("p-phish").textContent = (ml.probabilities.PHISHING * 100).toFixed(1) + "%";

  $("reasons").innerHTML = ml.reasons.length
    ? ml.reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")
    : '<li class="clear">No structural warning signs found in this message.</li>';

  $("signals").innerHTML = Object.entries(SIGNAL_LABELS)
    .map(([key, label]) => {
      const value = ml.signals[key];
      const on = value ? ' class="on"' : "";
      return `<div${on}><span>${label}</span><b>${formatValue(key, value)}</b></div>`;
    })
    .join("");
}

function renderLlm(result, agreement) {
  const block = $("llm-block");
  if (!result) { block.hidden = true; return; }
  block.hidden = false;

  if (result.error) {
    $("llm").innerHTML = `<p class="meta">${escapeHtml(result.error)}</p>`;
    return;
  }

  const agree = agreement === null || agreement === undefined ? "" :
    agreement
      ? '<div class="agree yes">Both engines agree.</div>'
      : '<div class="agree no">The engines disagree — worth a human look.</div>';

  $("llm").innerHTML =
    `<div class="head">
       <span class="tag ${result.label.toLowerCase()}">${result.label}</span>
       <span class="meta">${(result.confidence * 100).toFixed(0)}% confidence · ${escapeHtml(result.model)}</span>
     </div>
     <p>${escapeHtml(result.reason)}</p>${agree}`;
}

function formatValue(key, value) {
  if (key === "uppercase_ratio") return (value * 100).toFixed(1) + "%";
  if (key.startsWith("has_") || key.endsWith("_is_freemail") || key === "subject_has_urgency") {
    return value ? "yes" : "no";
  }
  return value;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
