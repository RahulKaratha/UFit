#!/usr/bin/env python3
"""
demo_fuzzy_gemini.py

Runnable demo:
  - has built-in sample data
  - tries the google-genai Python client first (if installed)
  - falls back to REST v1beta generateContent using x-goog-api-key
  - if calls fail or no key found, uses a mock Gemini response so the pipeline can be tested end-to-end

Usage:
  python demo_fuzzy_gemini.py

Environment:
  - Optionally create a .env with GEMINI_API_KEY and optional GEMINI_MODEL
    GEMINI_API_KEY=ya29....   (or an API key)
    GEMINI_MODEL=gemini-2.5-flash
"""
import os
import json
import time
from typing import List, Dict, Any

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# FUZZY LOGIC (recommend)
# -------------------------
def build_fuzzy_system():
    effort = ctrl.Antecedent(np.arange(0, 11, 1), 'effort')
    fatigue = ctrl.Antecedent(np.arange(0, 11, 1), 'fatigue')
    skipped = ctrl.Antecedent(np.arange(0, 11, 1), 'skipped')
    goal = ctrl.Antecedent(np.arange(0, 11, 1), 'goal')
    time_avail = ctrl.Antecedent(np.arange(0, 181, 1), 'time_avail')

    intensity = ctrl.Consequent(np.arange(0, 11, 1), 'intensity')
    duration = ctrl.Consequent(np.arange(0, 181, 1), 'duration')

    # membership functions
    effort['low'] = fuzz.trimf(effort.universe, [0, 0, 4])
    effort['medium'] = fuzz.trimf(effort.universe, [2, 5, 8])
    effort['high'] = fuzz.trimf(effort.universe, [6, 10, 10])

    fatigue['low'] = fuzz.trimf(fatigue.universe, [0, 0, 4])
    fatigue['medium'] = fuzz.trimf(fatigue.universe, [2, 5, 8])
    fatigue['high'] = fuzz.trimf(fatigue.universe, [6, 10, 10])

    skipped['none'] = fuzz.trimf(skipped.universe, [0, 0, 2])
    skipped['few'] = fuzz.trimf(skipped.universe, [1, 4, 7])
    skipped['many'] = fuzz.trimf(skipped.universe, [5, 10, 10])

    goal['fat_loss'] = fuzz.trimf(goal.universe, [0, 0, 3])
    goal['endurance'] = fuzz.trimf(goal.universe, [2, 4, 6])
    goal['strength'] = fuzz.trimf(goal.universe, [5, 6, 8])
    goal['muscle_gain'] = fuzz.trimf(goal.universe, [7, 10, 10])

    time_avail['short'] = fuzz.trimf(time_avail.universe, [0, 0, 30])
    time_avail['medium'] = fuzz.trimf(time_avail.universe, [20, 45, 75])
    time_avail['long'] = fuzz.trimf(time_avail.universe, [60, 180, 180])

    intensity['low'] = fuzz.trimf(intensity.universe, [0, 0, 4])
    intensity['medium'] = fuzz.trimf(intensity.universe, [3, 5, 7])
    intensity['high'] = fuzz.trimf(intensity.universe, [6, 10, 10])

    duration['short'] = fuzz.trimf(duration.universe, [0, 0, 25])
    duration['medium'] = fuzz.trimf(duration.universe, [20, 40, 70])
    duration['long'] = fuzz.trimf(duration.universe, [60, 120, 180])

    # rules
    rules = []
    rules.append(ctrl.Rule(fatigue['high'] | skipped['many'] | effort['high'],
                          (intensity['low'], duration['short'])))
    rules.append(ctrl.Rule(effort['low'] & fatigue['low'] & skipped['none'],
                          (intensity['high'], duration['medium'])))
    rules.append(ctrl.Rule(effort['medium'] & fatigue['medium'] & skipped['few'],
                          (intensity['medium'], duration['medium'])))

    # goal-based
    rules.append(ctrl.Rule(goal['endurance'] & effort['low'] & fatigue['low'],
                          (intensity['medium'], duration['long'])))
    rules.append(ctrl.Rule(goal['endurance'] & (effort['medium'] | fatigue['medium']),
                          (intensity['medium'], duration['medium'])))

    rules.append(ctrl.Rule(goal['strength'] & effort['low'] & fatigue['low'],
                          (intensity['high'], duration['medium'])))
    rules.append(ctrl.Rule(goal['strength'] & (effort['medium'] | fatigue['medium']),
                          (intensity['medium'], duration['short'])))

    rules.append(ctrl.Rule(goal['muscle_gain'] & effort['low'] & fatigue['low'],
                          (intensity['high'], duration['medium'])))
    rules.append(ctrl.Rule(goal['muscle_gain'] & (effort['medium'] | fatigue['medium']),
                          (intensity['medium'], duration['short'])))

    rules.append(ctrl.Rule(goal['fat_loss'] & effort['low'] & fatigue['low'],
                          (intensity['medium'], duration['long'])))
    rules.append(ctrl.Rule(goal['fat_loss'] & (effort['medium'] | fatigue['medium']),
                          (intensity['medium'], duration['medium'])))

    rules.append(ctrl.Rule(time_avail['short'], duration['short']))
    rules.append(ctrl.Rule(time_avail['long'] & goal['endurance'], duration['long']))

    sys = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(sys)
    return sim

def recommend(effort_val, fatigue_val, skipped_ratio_val, goal_val, time_available_min,
              prev_intensity=None, prev_duration=None):
    sim = build_fuzzy_system()

    effort_val = float(np.clip(effort_val, 0, 10))
    fatigue_val = float(np.clip(fatigue_val, 0, 10))
    skipped_ratio_val = float(np.clip(skipped_ratio_val, 0, 10))
    goal_val = float(np.clip(goal_val, 0, 10))
    time_available_min = float(np.clip(time_available_min, 0, 180))

    sim.input['effort'] = effort_val
    sim.input['fatigue'] = fatigue_val
    sim.input['skipped'] = skipped_ratio_val
    sim.input['goal'] = goal_val
    sim.input['time_avail'] = time_available_min

    sim.compute()

    raw_intensity = float(np.clip(sim.output['intensity'], 0, 10))
    raw_duration = float(np.clip(sim.output['duration'], 0, 180))

    # rest logic
    rest_recommendation = False
    rest_reason = None
    if fatigue_val >= 9 or effort_val >= 9:
        rest_recommendation = True
        rest_reason = "High fatigue or effort — recommend rest or light recovery."

    # clamp changes
    MAX_INTENSITY_DELTA = 2.0
    MAX_DURATION_PCT = 0.25

    if prev_intensity is not None:
        delta_i = np.clip(raw_intensity - prev_intensity, -MAX_INTENSITY_DELTA, MAX_INTENSITY_DELTA)
        final_intensity = np.clip(prev_intensity + delta_i, 0, 10)
    else:
        final_intensity = raw_intensity

    if prev_duration is not None:
        pct_change = (raw_duration - prev_duration) / max(1.0, prev_duration)
        pct_change = np.clip(pct_change, -MAX_DURATION_PCT, MAX_DURATION_PCT)
        final_duration = np.clip(prev_duration * (1 + pct_change), 0, time_available_min or 180)
    else:
        final_duration = np.clip(raw_duration, 0, time_available_min or 180)

    if time_available_min > 0:
        final_duration = min(final_duration, time_available_min)

    if rest_recommendation:
        return {
            "recommended_intensity": 0.0,
            "recommended_duration": 0.0,
            "intensity_delta": -(prev_intensity or raw_intensity),
            "duration_delta": -(prev_duration or raw_duration),
            "rest_recommendation": True,
            "rest_reason": rest_reason
        }

    return {
        "recommended_intensity": round(float(final_intensity), 2),
        "recommended_duration": round(float(final_duration), 1),
        "intensity_delta": round(float(final_intensity - (prev_intensity if prev_intensity is not None else raw_intensity)), 2),
        "duration_delta": round(float(final_duration - (prev_duration if prev_duration is not None else raw_duration)), 1),
        "rest_recommendation": False,
        "rest_reason": None
    }

# -------------------------
# Gemini integration
# -------------------------
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API") or os.getenv("GOOGLE_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def intensity_label(num: float) -> str:
    if num <= 3.33:
        return "low"
    if num <= 6.66:
        return "medium"
    return "high"

def build_prompt(user_goal: str, fuzzy_json: Dict[str, Any], last_3_muscles: List[str]) -> str:
    fuzzy_str = json.dumps(fuzzy_json, indent=2)
    prompt = f"""
You are a fitness planner. Produce a JSON-only workout plan (no explanation, no extra text).
Constraints:
- Output MUST be valid JSON and parseable by a machine.
- Top-level JSON shape:
  {{
    "exercises": [{{"name": string, "muscle_group": string, "sets": integer, "reps": integer, "intensity": string}}],
    "total_time_min": number,
    "notes": string_or_null
  }}
- Use intensity labels: "low", "medium", or "high".
- Prefer compound exercises for strength/muscle_gain goals and metabolic/circuit style for fat_loss/endurance when appropriate.
- Take into account the last 3 muscle groups trained (provided as a list). Avoid overloading them; you may include 1 light/rehab exercise for a recently trained muscle, but prioritize other groups.

Inputs:
1) user_goal: "{user_goal}"
2) fuzzy_output (machine JSON) -- use this directly to determine intensity and duration:
{fuzzy_str}
3) last_3_muscles: {json.dumps(last_3_muscles)}

Produce a JSON plan targeted to the recommended_duration (from fuzzy_output) and recommended_intensity.
If fuzzy_output.rest_recommendation is true, return exercises: [] and notes should explain the rest_reason string from fuzzy_output.
Return ONLY the JSON object (no surrounding text).
"""
    return prompt.strip()

def call_gemini_with_client(prompt: str, model: str = None, api_key: str = None) -> Dict[str, Any]:
    """
    Try the google-genai Python client first; if not available or fails, fall back to REST v1beta generateContent.
    Returns: {"text": "<text returned by model>"}
    """
    model_to_use = model or DEFAULT_MODEL
    # Try official client
    try:
        from google import genai
        # The client will use env var if api_key omitted
        client = genai.Client(api_key=api_key) if api_key else genai.Client()
        # SDK accepts contents either as string or list shape; use list-of-parts for predictability
        # Some SDK versions expose client.responses.create or client.models.generate_content - try both
        try:
            resp = client.models.generate_content(model=model_to_use, contents=[{"parts": [{"text": prompt}]}])
        except Exception:
            # try responses.create style (older/newer SDKs differ)
            resp = client.responses.create(model=model_to_use, input=prompt)
        # Attempt to extract text
        text = None
        # Common SDK attributes
        for attr in ("output_text", "text", "response", "content", "output"):
            if hasattr(resp, attr):
                try:
                    maybe = getattr(resp, attr)
                    if isinstance(maybe, str) and maybe.strip():
                        text = maybe
                        break
                    # if list/dict, try to stringify
                    if isinstance(maybe, (list, dict)):
                        text = json.dumps(maybe)
                        break
                except Exception:
                    pass
        if not text:
            # Try converting resp to string
            text = str(resp)
        return {"text": text}
    except Exception as e_client:
        # Fallback to REST v1beta generateContent
        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_to_use}:generateContent"
            headers = {
                "Content-Type": "application/json",
                # docs show x-goog-api-key is used for API-key-based auth in examples
                "x-goog-api-key": api_key or GEMINI_KEY or ""
            }
            body = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "maxOutputTokens": 800
            }
            r = requests.post(url, headers=headers, json=body, timeout=30)
            # Raise on HTTP error so caller can fallback if needed
            r.raise_for_status()
            data = r.json()
            # Extract model text from documented response shape:
            # data['candidates'][0]['content']['parts'][0]['text'], etc.
            text = None
            try:
                if isinstance(data, dict):
                    candidates = data.get("candidates") or data.get("candidate") or []
                    if candidates and isinstance(candidates, list):
                        c0 = candidates[0]
                        content = c0.get("content", {}) if isinstance(c0, dict) else {}
                        parts = content.get("parts", []) if isinstance(content, dict) else []
                        text_parts = []
                        for p in parts:
                            if isinstance(p, dict) and "text" in p:
                                text_parts.append(p["text"])
                        if text_parts:
                            text = "\n".join(text_parts)
                    # second fallback: some responses put output -> list of parts
                    if not text and "output" in data and isinstance(data["output"], list):
                        pieces = []
                        for item in data["output"]:
                            if isinstance(item, dict) and "content" in item:
                                for c in item["content"]:
                                    if c.get("type") == "output_text":
                                        pieces.append(c.get("text", ""))
                        if pieces:
                            text = "\n".join(pieces)
            except Exception:
                pass
            if not text:
                # Last fallback: return the raw JSON as string
                text = json.dumps(data)
            return {"text": text}
        except Exception as e_rest:
            # Propagate an informative exception to caller
            raise RuntimeError(f"Both SDK and REST call failed. SDK error: {e_client}; REST error: {e_rest}")

def parse_json_from_model(text: str) -> Dict[str, Any]:
    """
    Attempt to extract JSON from the model response robustly.
    Returns parsed dict or {"error": "...", "raw": "<raw text>"}
    """
    if not text:
        return {"error": "empty_response", "raw": text}
    text = text.strip()
    # Try to parse directly
    try:
        return json.loads(text)
    except Exception:
        # Find first { and last }, attempt to parse substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
    # Give up: return wrapper with raw text for debugging
    return {"error": "could_not_parse_model_output", "raw": text}

# -------------------------
# Main demo flow
# -------------------------
def main():
    # SAMPLE USER / previous state (change as you like)
    user_goal = "muscle_gain"  # one of: 'muscle_gain', 'strength', 'endurance', 'fat_loss'

    # Map goals to numeric domain the fuzzy system expects (0..10)
    goal_map = {
        "fat_loss": 0,
        "endurance": 3,
        "strength": 6,
        "muscle_gain": 9
    }
    goal_val = goal_map.get(user_goal, 5)

    # sample inputs (you can replace with dynamic inputs)
    effort_val = 6.0
    fatigue_val = 3.0
    skipped_ratio_val = 1.0
    time_available_min = 60.0
    prev_intensity = None
    prev_duration = None

    fuzzy_output = recommend(
        effort_val=effort_val,
        fatigue_val=fatigue_val,
        skipped_ratio_val=skipped_ratio_val,
        goal_val=goal_val,
        time_available_min=time_available_min,
        prev_intensity=prev_intensity,
        prev_duration=prev_duration
    )

    # example last 3 muscle groups trained
    last_3_muscles = ["chest", "triceps", "shoulders"]

    print("Fuzzy output (machine):")
    print(json.dumps(fuzzy_output, indent=2))

    # Build prompt for Gemini
    prompt = build_prompt(user_goal=user_goal, fuzzy_json=fuzzy_output, last_3_muscles=last_3_muscles)

    # If no API key, use mock
    if not GEMINI_KEY:
        print("\nNo GEMINI_API_KEY found in env — using mock Gemini response for demonstration.\n")
        if fuzzy_output.get("rest_recommendation"):
            mock = {
                "exercises": [],
                "total_time_min": 0,
                "notes": fuzzy_output.get("rest_reason")
            }
        else:
            intensity = intensity_label(fuzzy_output["recommended_intensity"])
            duration = float(fuzzy_output["recommended_duration"])
            if user_goal in ("muscle_gain", "strength"):
                exercises = [
                    {"name": "Barbell Squat", "muscle_group": "legs", "sets": 4, "reps": 6 if intensity=="high" else 8, "intensity": intensity},
                    {"name": "Bench Press", "muscle_group": "chest", "sets": 4, "reps": 6 if intensity=="high" else 8, "intensity": intensity},
                    {"name": "Bent-over Row", "muscle_group": "back", "sets": 3, "reps": 8, "intensity": intensity}
                ]
            else:
                exercises = [
                    {"name": "Circuit: Pushups", "muscle_group": "chest", "sets": 3, "reps": 12, "intensity": intensity},
                    {"name": "Jump Rope", "muscle_group": "full_body", "sets": 3, "reps": 60, "intensity": intensity}
                ]
            mock = {"exercises": exercises, "total_time_min": duration, "notes": None}
        print("Mock Gemini output:")
        print(json.dumps(mock, indent=2))
        with open("output.json", "w") as f:
            json.dump({"fuzzy": fuzzy_output, "plan": mock}, f, indent=2)
        print("\nSaved combined output to output.json")
        return

    # If we have a key, attempt real call
    print("\nGEMINI_API_KEY found — attempting to call Gemini. This may require the google-genai package or proper API access.\n")
    try:
        resp = call_gemini_with_client(prompt, model=os.getenv("GEMINI_MODEL"), api_key=GEMINI_KEY)
        text = resp.get("text", "")
        if text is None:
            text = ""
        print("Raw model text response (first 1000 chars):")
        print(text[:1000])
        parsed = parse_json_from_model(text)
        print("\nParsed JSON from model:")
        print(json.dumps(parsed, indent=2))
        with open("output.json", "w") as f:
            json.dump({"fuzzy": fuzzy_output, "plan": parsed}, f, indent=2)
        print("\nSaved combined output to output.json")
    except Exception as e:
        print("Error calling Gemini:", str(e))
        print("Falling back to mock output.\n")
        intensity = intensity_label(fuzzy_output["recommended_intensity"])
        duration = float(fuzzy_output["recommended_duration"])
        exercises = [
            {"name": "Goblet Squat", "muscle_group": "legs", "sets": 3, "reps": 10, "intensity": intensity},
            {"name": "Dumbbell Bench Press", "muscle_group": "chest", "sets": 3, "reps": 8, "intensity": intensity},
            {"name": "Single Arm Row", "muscle_group": "back", "sets": 3, "reps": 8, "intensity": intensity}
        ]
        mock = {"exercises": exercises, "total_time_min": duration, "notes": None}
        print(json.dumps(mock, indent=2))
        with open("output.json", "w") as f:
            json.dump({"fuzzy": fuzzy_output, "plan": mock}, f, indent=2)
        print("\nSaved combined output to output.json")

if __name__ == "__main__":
    main()
