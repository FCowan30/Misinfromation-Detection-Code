const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');

const { analyzeWithPython } = require("../controllers/analyzeController");

// save uploads in frontend/public/uploads
const upload = multer({
    dest: path.join(__dirname,"..", "public", "uploads"),
    limits: { fileSize: 8 * 1024 * 1024 }, // 8MB file size limit
});

router.get("/", (req,res) => {
    res.render("home", {
        form_text: "",
        form_explain: true,
        has_result: false
    });
});

router.post("/analyze", upload.single("image"), analyzeWithPython);

module.exports = router;