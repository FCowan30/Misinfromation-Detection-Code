const path = require("path");
const { spawn } = require("child_process");

function safeToString(x) {
  if (typeof x === "string") return x;
  if (Buffer.isBuffer(x)) return x.toString("utf8");
  return String(x);
}

function safePercentDisplay(value) {
  if (value === null || value === undefined || value === "") return "N/A";
  const n = Number(value);
  if (Number.isNaN(n)) return "N/A";
  return `${Math.round(n)}%`;
}

function safePercentValue(value) {
  if (value === null || value === undefined || value === "") return 0;
  const n = Number(value);
  if (Number.isNaN(n)) return 0;
  return Math.max(0, Math.min(100, Math.round(n)));
}

function safeRawDisplay(value, decimals = 4) {
  if (value === null || value === undefined || value === "") return "N/A";
  const n = Number(value);
  if (Number.isNaN(n)) return "N/A";
  return n.toFixed(decimals);
}

function getDriverShort(driver, analysisRoute) {
  if (analysisRoute === "clip_descriptive") return "Visual route";
  if (driver === "text") return "Text-led";
  if (driver === "image_mismatch") return "Image mismatch";
  if (driver === "both") return "Combined evidence";
  return "Mixed";
}

function getAnalysisRouteLabel(analysisRoute) {
  if (analysisRoute === "clip_descriptive") return "Visual consistency check";
  if (analysisRoute === "full_multimodal") return "Full multimodal analysis";
  if (analysisRoute === "text_only") return "Text-only analysis";
  if (analysisRoute === "text_only_fallback") return "Text-only fallback";
  return "Standard analysis";
}

function getRiskFlags(riskLevel) {
  const risk = (riskLevel || "").toLowerCase();

  const risk_low =
    risk.includes("lower") ||
    risk.includes("strong visual support") ||
    risk.includes("moderate visual support") ||
    risk.includes("weak visual support");

  const risk_high =
    risk.includes("high misinformation risk") ||
    risk.includes("strong visual conflict");

  const risk_medium =
    !risk_low && !risk_high && !!risk;

  return { risk_low, risk_medium, risk_high };
}

function getHeroClass(label, finalConfidencePct, analysisRoute) {
  const conf = Number(finalConfidencePct ?? 0);

  if (analysisRoute === "clip_descriptive") {
    if (label === "TRUE") {
      return conf >= 80 ? "hero-true-strong" : "hero-true-moderate";
    }
    if (label === "FAKE") {
      return conf >= 80 ? "hero-fake-strong" : "hero-fake-moderate";
    }
    return "hero-neutral";
  }

  if (label === "TRUE" || label === "REAL") {
    return conf >= 80 ? "hero-true-strong" : "hero-true-moderate";
  }

  if (label === "FAKE") {
    return conf >= 80 ? "hero-fake-strong" : "hero-fake-moderate";
  }

  return "hero-neutral";
}

function buildCommonViewModel(obj, text, explain) {
  const mode = obj.mode || "unknown";
  const isMultimodal =
    mode === "multimodal" || mode === "multimodal_descriptive";

  const prediction = obj.prediction || {};
  const label = prediction.label || "UNKNOWN";
  const conf = prediction.confidence ?? null;
  const analysisRoute = prediction.analysis_route || "";

  const fusion = obj.fusion || {};
  const fusionSignals = fusion.signals || {};
  const clipModel = fusion.clip_model || {};

  const driver = isMultimodal ? (fusionSignals.driver || "") : "";
  const textContrib = isMultimodal
    ? (fusionSignals.contributions?.text ?? null)
    : null;
  const imgContrib = isMultimodal
    ? (fusionSignals.contributions?.image_mismatch ?? null)
    : null;

  const clipSim01 = isMultimodal ? (clipModel.similarity_01 ?? null) : null;
  const clipMismatch = isMultimodal ? (clipModel.p_mismatch ?? null) : null;
  const clipRawSimilarity = isMultimodal ? (clipModel.raw_similarity ?? null) : null;

  const explanation = obj.explanation;
  const hasExplanation = explanation && !explanation.error;

  const detailed = obj.explanation_detailed;
  const hasDetailed = detailed && !detailed.error;

  const gradcam = hasDetailed ? detailed.gradcam : (hasExplanation ? explanation.gradcam : null);
  const gradcamUsable = !!(gradcam && !gradcam.error);

  let gradcamPath = "";
  let gradcamSummary = "";
  let gradcamStrength = "";
  let hasGradcamImage = false;

  if (gradcamUsable) {
    gradcamSummary = gradcam.top_region_summary || "";
    gradcamStrength =
      gradcam.activation_strength !== undefined
        ? Number(gradcam.activation_strength).toFixed(3)
        : "";

    if (gradcam.heatmap_path) {
      gradcamPath =
        "/" + gradcam.heatmap_path
          .replace(/^FrontEnd\/public\//, "")
          .replace(/\\/g, "/");
      hasGradcamImage = true;
    }
  }

  const scores = hasDetailed ? (detailed.scores || {}) : {};

  const finalConfidencePctValue =
    scores.final_confidence_pct ?? (conf !== null ? Math.round(Number(conf) * 100) : null);
  const textConfidencePctValue = scores.text_confidence_pct ?? null;
  const similarityPctValue =
    scores.similarity_pct ?? (clipSim01 !== null ? Math.round(Number(clipSim01) * 100) : null);
  const mismatchPctValue =
    scores.mismatch_pct ?? (clipMismatch !== null ? Math.round(Number(clipMismatch) * 100) : null);

  const riskLevel = hasDetailed ? (detailed.risk_level || "") : "";
  const { risk_low, risk_medium, risk_high } = getRiskFlags(riskLevel);

  const driverShort = getDriverShort(driver, analysisRoute);
  const analysisRouteLabel = getAnalysisRouteLabel(analysisRoute);
  const heroClass = getHeroClass(label, finalConfidencePctValue, analysisRoute);

  return {
    form_text: text,
    form_explain: explain,
    has_result: true,

    mode,
    result_label: label,

    // User-friendly display values (used by explainable page)
    result_confidence: safePercentDisplay(finalConfidencePctValue),

    // Raw developer values (used by baseline page)
    raw_confidence: conf !== null ? Number(conf).toFixed(4) : "N/A",
    raw_text_contrib: textContrib !== null ? Number(textContrib).toFixed(4) : "N/A",
    raw_img_contrib: imgContrib !== null ? Number(imgContrib).toFixed(4) : "N/A",
    raw_clip_similarity_01: clipSim01 !== null ? Number(clipSim01).toFixed(4) : "N/A",
    raw_clip_mismatch: clipMismatch !== null ? Number(clipMismatch).toFixed(4) : "N/A",
    raw_clip_similarity: clipRawSimilarity !== null ? Number(clipRawSimilarity).toFixed(4) : "N/A",

    is_true: label === "TRUE" || label === "REAL",
    is_fake: label === "FAKE",
    is_uncertain: label === "UNCERTAIN",

    is_multimodal: isMultimodal,
    driver,
    driver_short: driverShort,

    text_contrib: textContrib !== null ? Number(textContrib).toFixed(3) : "",
    img_contrib: imgContrib !== null ? Number(imgContrib).toFixed(3) : "",
    clip_sim01: clipSim01 !== null ? Number(clipSim01).toFixed(3) : "",
    clip_mismatch: clipMismatch !== null ? Number(clipMismatch).toFixed(3) : "",

    has_explanation: hasExplanation,
    explain_summary: hasExplanation ? explanation.summary : (explanation?.error || ""),
    has_flags: hasExplanation && Array.isArray(explanation.flags) && explanation.flags.length > 0,
    flags: hasExplanation ? explanation.flags : [],
    shap_tokens: hasExplanation ? (explanation.shap?.top_tokens || []) : [],

    has_gradcam: gradcamUsable,
    has_gradcam_image: hasGradcamImage,
    gradcam_path: gradcamPath,
    gradcam_summary: gradcamSummary,
    gradcam_strength: gradcamStrength,

    has_detailed: hasDetailed,
    detailed_summary: hasDetailed ? detailed.summary : "",
    prediction_text: hasDetailed ? (detailed.prediction_text || "") : "",
    flag_text: hasDetailed ? (detailed.flag_text || "") : "",
    similarity_text: hasDetailed ? (detailed.similarity_text || "") : "",
    mismatch_text: hasDetailed ? (detailed.mismatch_text || "") : "",
    driver_text: hasDetailed ? (detailed.driver_text || "") : "",
    visual_text: hasDetailed ? (detailed.visual_text || "") : "",

    risk_level: riskLevel,
    risk_low,
    risk_medium,
    risk_high,

    analysis_route_text: hasDetailed ? (detailed.analysis_route_text || "") : "",
    analysis_route_label: analysisRouteLabel,

    evidence_for: hasDetailed ? (detailed.evidence_for || "") : "",
    evidence_against: hasDetailed ? (detailed.evidence_against || "") : "",
    top_reasons: hasDetailed && Array.isArray(detailed.top_reasons) ? detailed.top_reasons : [],
    decision_pathway: hasDetailed && Array.isArray(detailed.decision_pathway) ? detailed.decision_pathway : [],

    confidence_meaning_text: hasDetailed ? (detailed.confidence_meaning_text || "") : "",
    uncertainty_text: hasDetailed ? (detailed.uncertainty_text || "") : "",
    limitations_text: hasDetailed ? (detailed.limitations_text || "") : "",

    final_confidence_pct: safePercentValue(finalConfidencePctValue),
    text_confidence_pct: safePercentDisplay(textConfidencePctValue),
    text_confidence_pct_value: safePercentValue(textConfidencePctValue),
    similarity_pct: safePercentDisplay(similarityPctValue),
    similarity_pct_value: safePercentValue(similarityPctValue),
    mismatch_pct: safePercentDisplay(mismatchPctValue),
    mismatch_pct_value: safePercentValue(mismatchPctValue),

    hero_class: heroClass,

    raw_json: JSON.stringify(obj, null, 2)
  };
}

function runPythonAndRender(req, res, viewName, options = {}) {
  const text = (req.body.text || "").trim();
  const explain =
    options.forceExplain === true
      ? true
      : options.forceExplain === false
        ? false
        : !!req.body.explain;

  if (!text) return res.status(400).send("Text is required.");

  const imagePath = req.file ? req.file.path : null;
  const projectRoot = path.join(__dirname, "..", "..");

  const args = ["-m", "Backend.run", "--text", text];
  if (imagePath) args.push("--image", imagePath);
  if (explain) args.push("--explain", "--top-n", "10");
  args.push("--pretty");

  const pythonPath = path.join(projectRoot, ".venv", "Scripts", "python.exe");

  const py = spawn(pythonPath, args, {
    cwd: projectRoot,
    shell: false
  });

  let stdout = "";
  let stderr = "";

  py.stdout.on("data", (d) => (stdout += safeToString(d)));
  py.stderr.on("data", (d) => (stderr += safeToString(d)));

  py.on("close", (code) => {
    const outStr = stdout.trim();

    if (!outStr) {
      return res.status(500).send(
        `Python returned no stdout.\nExit code: ${code}\n\nSTDERR:\n${stderr}`
      );
    }

    let obj;
    try {
      obj = JSON.parse(outStr);
    } catch (e) {
      return res.status(500).send(
        `Failed to parse JSON from Python.\n\nError: ${e}\n\nSTDOUT:\n${outStr}\n\nSTDERR:\n${stderr}`
      );
    }

    const viewModel = buildCommonViewModel(obj, text, explain);
    res.render(viewName, viewModel);
  });

  py.on("error", (err) => {
    res.status(500).send(`Failed to start Python process: ${err}`);
  });
}

exports.analyzeWithPython = (req, res) => {
  // Page 1 = developer/raw mode
  runPythonAndRender(req, res, "home", { forceExplain: false });
};

exports.analyzeWithPythonDetailed = (req, res) => {
  // Page 2 = full explainable mode
  runPythonAndRender(req, res, "home_explain", { forceExplain: true });
};