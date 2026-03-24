const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

const projectRoot = path.join(__dirname, "..", "..");
const pythonPath = path.join(projectRoot, ".venv", "Scripts", "python.exe");

const metricsPath = path.join(projectRoot, "Backend", "Evaluation", "metrics.json");
const statusPath = path.join(projectRoot, "Backend", "Evaluation", "status.json");

function readJsonSafe(filePath, fallback = {}) {
  try {
    if (!fs.existsSync(filePath)) return fallback;
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return fallback;
  }
}

exports.renderEvaluationPage = (req, res) => {
  const metrics = readJsonSafe(metricsPath, null);
  const status = readJsonSafe(statusPath, {
    stage: "idle",
    message: "No training run yet.",
    done: false,
    error: null,
    updated_at: null
  });

  res.render("evaluation", {
    has_metrics: !!metrics,
    metrics,

    accuracy: metrics ? metrics.accuracy : "",
    precision: metrics ? metrics.precision : "",
    recall: metrics ? metrics.recall : "",
    f1: metrics ? metrics.f1 : "",
    roc_auc: metrics ? metrics.roc_auc : "",
    train_size: metrics ? metrics.train_size : "",
    eval_size: metrics ? metrics.eval_size : "",
    last_updated: metrics ? metrics.last_updated : "",

    confusion_matrix_path: metrics?.plots?.confusion_matrix || "",
    roc_curve_path: metrics?.plots?.roc_curve || "",

    status_stage: status.stage || "idle",
    status_message: status.message || "",
    status_updated: status.updated_at || "",
    status_error: status.error || ""
  });
};

exports.startRetrain = (req, res) => {
  // ✅ FIX: args was missing → this is why retraining never started
  const args = ["-m", "Backend.Evaluation.train_and_eval"];

  const py = spawn(pythonPath, args, {
    cwd: projectRoot,
    shell: false,
    detached: false,
    stdio: ["ignore", "pipe", "pipe"]
  });

  // ✅ Logging so you can actually see errors
  py.stdout.on("data", (data) => {
    console.log("[RETRAIN STDOUT]", data.toString());
  });

  py.stderr.on("data", (data) => {
    console.error("[RETRAIN STDERR]", data.toString());
  });

  py.on("error", (err) => {
    console.error("[RETRAIN PROCESS ERROR]", err);
  });

  py.on("close", (code) => {
    console.log(`[RETRAIN PROCESS CLOSED] Exit code: ${code}`);
  });

  res.json({
    ok: true,
    message: "Retraining started."
  });
};

exports.getStatus = (req, res) => {
  const status = readJsonSafe(statusPath, {
    stage: "idle",
    message: "No training run yet.",
    done: false,
    error: null,
    updated_at: null
  });

  const metrics = readJsonSafe(metricsPath, null);

  res.json({
    status,
    metrics
  });
};