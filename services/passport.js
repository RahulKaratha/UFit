const passport = require("passport");
const GoogleStrategy = require("passport-google-oauth20");
const User = require("../models/user");
const dotenv = require("dotenv");

dotenv.config();

passport.use(
  new GoogleStrategy(
    {
      clientID: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      callbackURL: "/profile",
    },
    async (accessToken, refreshToken, Profiler, done) => {
      const existingUser = await User.findOne({ googleId: profile.id });

      if (existingUser) return done(null, existingUser);

      const user = await User.create({
        googleId:profile.id,
        name:profile.displayName,
        email:profile.emails[0].value,
  
      });

      done(null,user);
    }
  )
);
