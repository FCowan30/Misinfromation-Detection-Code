const path = require("path");
const { spawn } = require("child_process");

function safeToString(x) {
  if (typeof x === "string") return x;
  if (Buffer.isBuffer(x)) return x.toString("utf8");
  return String(x);
}

exports.analyzeWithPython = (req, res) => {
  const text = (req.body.text || "").trim();
  const explain = !!req.body.explain;

  if (!text) return res.status(400).send("Text is required.");

  const imagePath = req.file ? req.file.path : null;

  // Project root (one level above FrontEnd)
  const projectRoot = path.join(__dirname, "..", "..");

  // Build args for CLI mode
  const args = ["-m", "Backend.run", "--text", text];

  if (imagePath) {
    args.push("--image", imagePath);
  }

  if (explain) {
    args.push("--explain", "--top-n", "10");
  }

  args.push("--pretty");

  const pythonPath = path.join(projectRoot, ".venv", "Scripts", "python.exe");
  console.log("Using Python:", pythonPath);

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

    // --- Render the home page with results ---
    const mode = obj.mode || "unknown";
    const isMultimodal = mode === "multimodal";

    const prediction = obj.prediction || {};
    const label = prediction.label || "UNKNOWN";
    const conf = prediction.confidence ?? null;

    const driver = isMultimodal ? (obj.fusion?.signals?.driver || "") : "";
    const textContrib = isMultimodal ? (obj.fusion?.signals?.contributions?.text ?? null) : null;
    const imgContrib = isMultimodal ? (obj.fusion?.signals?.contributions?.image_mismatch ?? null) : null;
    const clipSim01 = isMultimodal ? (obj.fusion?.clip_model?.similarity_01 ?? null) : null;
    const clipMismatch = isMultimodal ? (obj.fusion?.clip_model?.p_mismatch ?? null) : null;

    const explanation = obj.explanation;
    const hasExplanation = explanation && !explanation.error;

    const gradcam = hasExplanation ? explanation.gradcam : null;
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

    res.render("home", {
      form_text: text,
      form_explain: explain,
      has_result: true,

      mode,
      result_label: label,
      result_confidence: conf !== null ? Number(conf).toFixed(3) : "N/A",
      is_true: label === "TRUE",
      is_fake: label === "FAKE",

      is_multimodal: isMultimodal,
      driver,
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

      raw_json: JSON.stringify(obj, null, 2)
    });
  });

  py.on("error", (err) => {
    res.status(500).send(`Failed to start Python process: ${err}`);
  });
};