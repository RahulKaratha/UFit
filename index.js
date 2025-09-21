const express = require("express");
const cors = require("cors");
const connectDB = require("./config/db");
const dotenv = require("dotenv");
const passport = require("passport");
const oauthRoutes = require("./routes/oauthRoutes")
require("./services/passport");

dotenv.config();
connectDB();

const app = express();

app.use(cors());
app.use(express.json());
app.use(passport.initialize());

app.get("/",(req,res)=>{
    res.send("UFit Application is Running")
});


app.get("/profile",(req,res)=>{
    res.send("Succesful Login")
});

app.use("/home/oauth",oauthRoutes);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

