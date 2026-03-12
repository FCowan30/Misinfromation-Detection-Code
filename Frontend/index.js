const express = require("express");
const path = require("path");
const mustacheExpress = require("mustache-express");

const app = express();

//mustache
app.engine(
  "mustache",
  mustacheExpress(
    path.join(__dirname, "views", "partials")
  )
);
app.set("view engine","mustache");
app.set("views", path.join(__dirname, "views"));

//middleware
app.use(express.urlencoded({ extended: true}));
app.use(express.static(path.join(__dirname, "public")));

//routes
app.use("/", require("./routes/index"));

const PORT = 3000;
app.listen(PORT, () => console.log(`Frontend is running: http://localhost:${PORT}`));