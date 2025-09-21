const express = require("express");
const passport = require("passport");
const generateToken = require("../utils/generateToken")

const router = express.Router();

router.get("/google",passport.authenticate("google",{scope:["profile","email"]}));


router.get("/profile",
    passport.authenticate("google",{session:false,failureRedirect:"/"}),
    (req,res)=>{
        const token = generateToken(req.user._id);
        res.json({success:true,token});
    }
);

module.exports = router;