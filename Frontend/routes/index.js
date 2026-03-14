const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");

const {
  analyzeWithPython,
  analyzeWithPythonDetailed
} = require("../controllers/analyzeController");

const {
  renderEvaluationPage,
  startRetrain,
  getStatus
} = require("../controllers/evaluationController");

const upload = multer({
  dest: path.join(__dirname, "..", "public", "uploads"),
  limits: { fileSize: 8 * 1024 * 1024 }
});

router.get("/", (req, res) => {
  res.render("home", {
    form_text: "",
    form_explain: true,
    has_result: false
  });
});

router.get("/home_explain", (req, res) => {
  res.render("home_explain", {
    form_text: "",
    form_explain: true,
    has_result: false
  });
});

router.get("/evaluation", renderEvaluationPage);

router.post("/analyze", upload.single("image"), analyzeWithPython);
router.post("/analyze-explain", upload.single("image"), analyzeWithPythonDetailed);

router.post("/evaluation/retrain", startRetrain);
router.get("/evaluation/status", getStatus);

module.exports = router;