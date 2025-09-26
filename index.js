// models/index.js
const mongoose = require('mongoose');
const { Schema } = mongoose;

/* -----------------------------
   User Schema (auth)
   ----------------------------- */
const UserSchema = new Schema({
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true,
    index: true
  },
  passwordHash: {
    type: String,
    required: true,
    select: false // do not return by default
  },
  // roles, flags, or provider info can be added here
}, {
  timestamps: true // createdAt & updatedAt
});

/* -----------------------------
   UserProfile Schema (fitness data)
   Kept in separate collection (referenced from User)
   ----------------------------- */
const UserProfileSchema = new Schema({
  userId: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    unique: true, // one-to-one mapping
    index: true
  },
  // core fitness fields
  goals: [{
    type: String,
    enum: ['Strength', 'Hypertrophy', 'Fat Loss', 'Endurance', 'Mobility', 'General Fitness', 'Rehab'],
    default: 'General Fitness'
  }],
  fitnessLevel: {
    type: String,
    enum: ['Beginner', 'Intermediate', 'Advanced'],
    default: 'Beginner',
    index: true
  },
  age: { type: Number, min: 8, max: 120 },
  heightCm: { type: Number, min: 30 },     // centimeters
  weightKg: { type: Number, min: 6 },      // kilograms
  sex: { type: String, enum: ['male','female','other'] },
  dominantHand: { type: String, enum: ['left','right','ambidextrous'], default: 'right' },

  // preferences / constraints used by AI generator
  equipment: [{ type: String }], // e.g., ['dumbbells','barbell','resistance_band','bodyweight']
  injuriesOrLimitations: [{ type: String }],
  preferredWorkoutDurationMinutes: { type: Number, default: 45 },
  timezone: { type: String },

  // usage metrics (denormalized counters) - optional but useful
  createdAt: { type: Date, default: Date.now }
}, {
  timestamps: true
});

/* -----------------------------
   Exercise Sub-document (embedded in Workout)
   ----------------------------- */
const ExerciseSchema = new Schema({
  exerciseName: { type: String, required: true }, // human-friendly name
  exerciseId: { type: Schema.Types.ObjectId, ref: 'ExerciseLibrary', required: false }, // optional ref to a canonical library
  sets: { type: Number, default: 3, min: 1 },
  reps: { type: Number },         // can be null for time-based exercises
  weightKg: { type: Number },     // optional
  rpe: { type: Number, min: 1, max: 10 }, // optional rate of perceived exertion for the set
  tempo: { type: String },        // e.g., "2-0-1"
  restPeriodSeconds: { type: Number, default: 60 },
  notes: { type: String }
}, { _id: false }); // don't create an id for every embedded exercise by default

/* -----------------------------
   Workout Schema (daily plan)
   Embedded exercises (array)
   Stores a small feedbackSummary (denormalized) for quick dashboard reads
   ----------------------------- */
const WorkoutSchema = new Schema({
  userId: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  date: { // date for which the workout is scheduled (date only)
    type: Date,
    required: true,
    index: true
  },
  status: {
    type: String,
    enum: ['pending', 'completed', 'skipped', 'cancelled'],
    default: 'pending',
    index: true
  },
  generatedBy: { type: String, enum: ['gemini', 'manual', 'template'], default: 'gemini' },

  // core plan: list of exercises (embedded)
  exercises: {
    type: [ExerciseSchema],
    default: []
  },

  // metadata for AI -> store prompt snapshot or generation metadata (optional)
  generationMeta: {
    promptHash: { type: String }, // small summary to correlate versions
    promptTextShort: { type: String }, // truncated prompt if needed
    aiModel: { type: String, default: 'gemini' },
    generationTimeMs: { type: Number }
  },

  // Denormalized feedback summary (very small) to avoid joins for common dashboard needs:
  feedbackSummary: {
    // created/updated when a WorkoutFeedback doc is created: updated by application logic
    hasFeedback: { type: Boolean, default: false },
    fuzzyScore: { type: Number, min: 1, max: 10 }, // 1-10, optional
    qualitativeRating: { type: String, enum: ['Too Easy','Moderate','Too Hard'] },
    feedbackAt: { type: Date }
  },

  notes: { type: String } // optional user/coach notes
}, {
  timestamps: true
});

// compound unique index to prevent multiple workouts for same user+date
WorkoutSchema.index({ userId: 1, date: 1 }, { unique: true });

/* -----------------------------
   WorkoutFeedback Schema (separate collection)
   - Primary storage for feedback (one doc per workout when user submits)
   - Keeps analytics-friendly structure and history (can grow with more fields)
   ----------------------------- */
const WorkoutFeedbackSchema = new Schema({
  userId: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  workoutId: {
    type: Schema.Types.ObjectId,
    ref: 'Workout',
    required: true,
    index: true,
    unique: true // one-to-one mapping to workout
  },
  createdAtClient: { type: Date }, // when user submitted (or local device time)
  qualitativeRating: {
    type: String,
    enum: ['Too Easy', 'Moderate', 'Too Hard'],
    required: true,
    index: true
  },
  notes: { type: String, maxlength: 2000 },

  // the fuzzy score is crucial for the AI - 1 to 10
  fuzzyScore: {
    type: Number,
    required: true,
    min: 1,
    max: 10
  },

  // optional structured telemetry: heart rate, perceived exertion, completion percentage
  completionPercent: { type: Number, min: 0, max: 100 },
  avgHeartRateBpm: { type: Number, min: 30, max: 240 },

  // store derived features used for the AI training/feedback loop
  derived: {
    type: Schema.Types.Mixed // flexible container for features (json blob)
  }
}, {
  timestamps: true
});

// ensure we can fetch user's recent feedback fast
WorkoutFeedbackSchema.index({ userId: 1, createdAt: -1 });
WorkoutFeedbackSchema.index({ workoutId: 1, userId: 1 });

/* -----------------------------
   Optional: ExerciseLibrary (canonical exercises referenced from Workout)
   Useful if you want an internal library of exercises; not required.
   ----------------------------- */
const ExerciseLibrarySchema = new Schema({
  name: { type: String, required: true, index: true },
  category: { type: String }, // e.g., 'Upper Body', 'Lower Body', 'Cardio'
  primaryMuscles: [{ type: String }],
  equipment: [{ type: String }],
  instructions: { type: String },
  defaultTempo: { type: String },
  defaultRestSeconds: { type: Number }
}, { timestamps: true });

/* -----------------------------
   Helpers / statics (example)
   ----------------------------- */
// Example: map qualitative rating to default fuzzyScore (app can override)
WorkoutFeedbackSchema.statics.mapQualitativeToFuzzy = function(qualitative) {
  if (qualitative === 'Too Easy') return 3;
  if (qualitative === 'Moderate') return 6;
  if (qualitative === 'Too Hard') return 8;
  return 5; // fallback
};

WorkoutSchema.methods.applyFeedbackSummary = function(feedbackDoc) {
  if (!feedbackDoc) return;
  this.feedbackSummary = {
    hasFeedback: true,
    fuzzyScore: feedbackDoc.fuzzyScore,
    qualitativeRating: feedbackDoc.qualitativeRating,
    feedbackAt: feedbackDoc.createdAt
  };
  return this;
};

/* -----------------------------
   Model exports
   ----------------------------- */
const User = mongoose.model('User', UserSchema);
const UserProfile = mongoose.model('UserProfile', UserProfileSchema);
const Workout = mongoose.model('Workout', WorkoutSchema);
const WorkoutFeedback = mongoose.model('WorkoutFeedback', WorkoutFeedbackSchema);
const ExerciseLibrary = mongoose.model('ExerciseLibrary', ExerciseLibrarySchema);

module.exports = {
  User,
  UserProfile,
  Workout,
  WorkoutFeedback,
  ExerciseLibrary
};
