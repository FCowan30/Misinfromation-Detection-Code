const express = require("express");
const path = require("path");
const mustacheExpress = require("mustache-express");

const app = express();

app.engine(
  "mustache",
  mustacheExpress(path.join(__dirname, "views", "partials"))
);

app.set("view engine", "mustache");
app.set("views", path.join(__dirname, "views"));

app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, "public")));

app.use("/", require("./routes/index"));

const PORT = 3000;
app.listen(PORT, () => console.log(`Frontend is running: http://localhost:${PORT}`));