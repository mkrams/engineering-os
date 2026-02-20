"""
EngineeringOS — Agentic NL → SysMLv2 Hardware Modeler
======================================================
An AI-powered tool that actively guides you through building
a formal SysMLv2 hardware model from natural language.
"""

import os
import json
import textwrap
from datetime import datetime

from flask import Flask, render_template, request, jsonify
import anthropic

from sysmlv2_validator import validate_sysmlv2

app = Flask(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

model_state = {
    "sysml_source": "",
    "history": [],
    "conversation": [],  # Full agent conversation for context
}


def reset_model():
    model_state["sysml_source"] = ""
    model_state["history"] = []
    model_state["conversation"] = []


# ---------------------------------------------------------------------------
# The agentic system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
You are EngineeringOS, an expert systems engineer AI that helps hardware
engineers build formal SysMLv2 models incrementally from natural language.

You are AGENTIC — you don't just translate, you actively HELP:

## Your Behavior

1. **TRANSLATE** the user's natural language into valid SysMLv2 textual notation.
2. **ANALYZE** the current model for gaps, missing definitions, incomplete
   specs, and logical conflicts.
3. **ASK** targeted follow-up questions about anything that's underspecified.
   Don't ask more than 3 questions at a time. Prioritize the most critical gaps.
4. **SUGGEST** what the user should define next based on good systems
   engineering practice (interfaces before internals, requirements before
   implementation, etc.).
5. **DETECT CONFLICTS** — if the user says something that contradicts the
   existing model, explain the conflict clearly and propose resolutions.
6. **GUIDE** the user through a logical progression:
   - System boundary & context
   - Key subsystems / part decomposition
   - Interfaces & ports between subsystems
   - Attributes & physical parameters
   - Constraints & requirements
   - Behaviors & states

## SysMLv2 Best Practices
- part def for component definitions
- attribute with ISQ types (MassValue, LengthValue, ForceValue, etc.) or
  primitives (Real, Integer, Boolean, String)
- port def / port for interfaces
- requirement def with require constraint for requirements
- connection for wiring
- Multiplicities: [4], [1..*], [0..1]
- Specialization :> for inheritance
- Wrap everything in a top-level package
- Add // comments tracing back to the NL input

## Output Format

Respond with EXACTLY this structure (three sections separated by markers):

---SYSML---
<the complete, updated SysMLv2 model source code — NOT a delta, the FULL model>
---ANALYSIS---
<a JSON object with your agentic analysis>
---END---

The JSON object MUST have these fields:
{
  "summary": "What changed in this update (1 sentence)",
  "completeness": 0-100,
  "questions": [
    {"priority": "high|medium|low", "question": "...", "context": "why this matters"}
  ],
  "suggestions": ["What to define next..."],
  "conflicts": [
    {"description": "...", "resolution": "..."}
  ],
  "missing_definitions": ["ObjectName1", ...],
  "model_health": {
    "parts_defined": 0,
    "requirements_defined": 0,
    "interfaces_defined": 0,
    "constraints_defined": 0
  }
}

If this is the FIRST input (empty model), also add a welcome-style summary
and suggest the logical next steps for the user's hardware product.

Be concise but thorough. You are a senior systems engineer, not a chatbot.
""")


def call_agent(nl_input: str, current_model: str, conversation_history: list) -> dict:
    """Call Claude as an agentic systems engineer."""
    client = anthropic.Anthropic()

    # Build messages with conversation history for context
    messages = []

    # Include last few conversation turns for context (keep it bounded)
    recent_history = conversation_history[-6:]
    for turn in recent_history:
        messages.append(turn)

    # Current request
    user_msg = f"""CURRENT MODEL STATE:
```sysml
{current_model if current_model else '(empty — no model yet)'}
```

USER INPUT:
{nl_input}

Analyze this input, update the model, identify gaps, and guide the user.
Remember: output the COMPLETE updated model, not just changes."""

    messages.append({"role": "user", "content": user_msg})

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    raw = response.content[0].text

    # Parse the three sections
    sysml_code = ""
    analysis = {}

    if "---SYSML---" in raw and "---ANALYSIS---" in raw:
        parts = raw.split("---SYSML---", 1)
        rest = parts[1] if len(parts) > 1 else ""

        if "---ANALYSIS---" in rest:
            sysml_part, analysis_part = rest.split("---ANALYSIS---", 1)
            sysml_code = sysml_part.strip()

            # Clean up analysis part
            analysis_str = analysis_part.replace("---END---", "").strip()
            # Remove markdown fences if present
            analysis_str = analysis_str.replace("```json", "").replace("```", "").strip()
            try:
                analysis = json.loads(analysis_str)
            except json.JSONDecodeError:
                analysis = {
                    "summary": "Model updated",
                    "completeness": 0,
                    "questions": [],
                    "suggestions": [],
                    "conflicts": [],
                    "missing_definitions": [],
                    "model_health": {}
                }
    else:
        # Fallback
        sysml_code = raw.strip()

    # Clean markdown fences from SysMLv2
    sysml_code = sysml_code.replace("```sysml", "").replace("```", "").strip()

    # Store conversation turn for context
    conversation_history.append({"role": "user", "content": user_msg})
    conversation_history.append({"role": "assistant", "content": raw})

    return {"sysml": sysml_code, "analysis": analysis}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/add", methods=["POST"])
def add_requirement():
    data = request.json
    nl_input = data.get("input", "").strip()
    if not nl_input:
        return jsonify({"error": "Empty input"}), 400

    try:
        result = call_agent(
            nl_input,
            model_state["sysml_source"],
            model_state["conversation"],
        )
    except anthropic.APIError as e:
        return jsonify({"error": f"API error: {str(e)}"}), 500

    new_source = result["sysml"]
    analysis = result["analysis"]

    # Validate
    validation = validate_sysmlv2(new_source)

    # Merge validation errors into analysis conflicts
    if validation["error_count"] > 0:
        for diag in validation["diagnostics"]:
            if diag["severity"] == "error":
                if "conflicts" not in analysis:
                    analysis["conflicts"] = []
                analysis["conflicts"].append({
                    "description": f"Syntax error at L{diag['line']}: {diag['message']}",
                    "resolution": "Fix the syntax issue in the generated code",
                })

    # Update state
    model_state["sysml_source"] = new_source
    model_state["history"].append({
        "nl": nl_input,
        "timestamp": datetime.now().isoformat(),
        "summary": analysis.get("summary", ""),
    })

    return jsonify({
        "sysml": new_source,
        "validation": validation,
        "analysis": analysis,
        "history_length": len(model_state["history"]),
    })


@app.route("/api/model", methods=["GET"])
def get_model():
    return jsonify({
        "sysml": model_state["sysml_source"],
        "history": model_state["history"],
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    reset_model()
    return jsonify({"status": "ok"})


@app.route("/api/validate", methods=["POST"])
def validate_endpoint():
    data = request.json
    source = data.get("source", "")
    validation = validate_sysmlv2(source)
    model_state["sysml_source"] = source
    return jsonify({"validation": validation})


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠️  Set ANTHROPIC_API_KEY environment variable first:")
        print("   export ANTHROPIC_API_KEY='sk-ant-...'")
        print()
    app.run(debug=True, port=8080)
