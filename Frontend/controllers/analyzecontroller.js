const path = require("path");
const { spawn } = require("child_process");
const { stdout } = require("process");

function extractJsonFromStdout(Stdout) {
    const first = stdout.indexOf("{");
    const last = stdout.lastIndexOf("}");
    if (first !== -1 && last !== -1) return stdout.slice(first, last + 1);
    return null;
}

function renderHome(res, form_test,form_expalin, obj) {
    const mode = obj.mode || "unknown";
    const isMultimodal = mode === "multimodal";

    const prediction = obj.prediction || {};
    const label = prediction.label || "unknown";
    const conf = prediction.confidence ?? null;

    const driver = isMultimodal ? (obj.fusion?.signlas?.driver || "") : "";
    const textContrib = isMultimodal ? (obj.fusion?.signals?.contributions?.text ?? null) : nulll;
    const imgcontrib = isMultimodal ? (obj.fusion?.signals?.contributions?.image_mismatch ?? null) : null;
    const clipSim01 = isMultimodal ? (obj.fusion?.clip_modal?.similarity01 ?? null) : null;
    const clipMismatch = isMultimodal ? (obj.fusion?.clip_modal?.p_mismatch ?? null) : null;

    const explanation = obj.explanation;
    const hasExplanation = explanation && !explanation.error;

    res.render("home", {
        from_text,
        form_explain,
        has_result: true,

        mode,
        result_label: label,
        result_confidence: conf !== null ? Number(conf).toFixed(3) : "N/A",
        is_true: label === "true",
        is_fake: label === "fake",

        isMultimodal: isMultimodal,
        driver,
        textContrib: textContrib !== null ? Number(textContrib).toFixed(3) : "",
        imgContrib: imgContrib !== null ? Number(imgContrib).toFixed(3) : "",
        clipSim01: clipSim01 !== null ? Number(clipSim01).toFixed(3) : "",
        clipMismatch: clipMismatch !== null ? Number(clipMismatch).toFixed(3) : "",

        hasExplanation: hasExplanation,
        explain_summery: hasExplaination? explanation.summary : (explanation?.error || ""),
        has_flags: hasExplanation && Array.isArray(explanation.flags) && explanation.flags.length > 0,
        shap_tokens: hasExplanation ? (explanation.shap?.top_tokens || []) : [],

        raw_json: JSON.stringify(obj, null, 2)
    });
}

exports.analyzeWithPython = (req, res) => {
    const text = (req.body.text || "").trim();
    const explain = !!req.body.explain;

    if (!text) return res.status(400).send("this text is required");

    const imagePath = req.file ? req.file.path : "";

    const projectRoot = path.join(__dirname, "..", "..");

    const py = spawn("python", ["-m", "Backend.run"], {
        cws: projectRoot,
        shell: true
    });

    let stdout = "";
    let stderr = "";

    py.stdout.on("data", (d) => (stdout += d.tostring()));
    py.stderr.on("data", (d) => (stderr += d.toString()));
    
    py.on("close", () => {
        const jsonStr = extractJsonFromStdout(stdout);
        if (!jsonStr) {
            return res.status(500).send(
                `Python did not return JSON.\n\nSTDERR:\n${stderr}\n\nSTDOUT:\n${stdout}`
            );
    }

    try {
      const obj = JSON.parse(jsonStr);
      renderHome(res, text, explain, obj);
    } catch (e) {
      return res.status(500).send(
        `Failed to parse JSON.\n\n${e}\n\nRAW JSON:\n${jsonStr}\n\nSTDERR:\n${stderr}`
      );
    }
  });

    // Feed the interactive prompts in the exact order Backend.run expects:
  py.stdin.write(text + "\n");
  py.stdin.write((imagePath || "") + "\n");
  py.stdin.write((explain ? "y" : "n") + "\n");
  py.stdin.end();
};
