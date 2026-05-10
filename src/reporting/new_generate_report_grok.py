"""
ChestX-AI-Assistant - Grok/xAI Report Generator
===============================================
Reads outputs/reports/sample_evidence.json, calls the xAI Grok API to produce a
structured radiology-style narrative, then saves a formatted PDF.

Usage example:
    python src/reporting/new_generate_report_grok.py \
        --evidence outputs/reports/sample_evidence.json \
        --xray data/raw/patient_xray.png \
        --patient "John Doe" \
        --dob "1975-03-12" \
        --mrn "MRN-00421" \
        --out outputs/reports/report.pdf

Requirements:
    pip install openai reportlab Pillow

Environment:
    export XAI_API_KEY="your_xai_api_key_here"
    # optional: export XAI_MODEL="grok-4.3"
    # optional: export XAI_BASE_URL="https://api.x.ai/v1"
"""

import argparse
import json
import os
from datetime import datetime
from html import escape
from pathlib import Path

from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ------------------------------- helpers ------------------------------------

THRESHOLD = 0.50  # probability above which a finding is treated as positive
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

LABEL_DESCRIPTIONS = {
    "Atelectasis": "partial or complete collapse of lung tissue",
    "Cardiomegaly": "enlargement of the cardiac silhouette",
    "Consolidation": "airspace opacification consistent with consolidation",
    "Pleural Effusion": "fluid accumulation in the pleural space",
    "Pneumonia": "airspace disease consistent with pneumonia",
}

REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {
            "type": "string",
            "description": "A concise 3-6 sentence radiology findings paragraph.",
        },
        "impression": {
            "type": "array",
            "description": "Prioritized impression items, most important first.",
            "items": {"type": "string"},
        },
    },
    "required": ["findings", "impression"],
}


def confidence_word(probability: float) -> str:
    if probability >= 0.85:
        return "high"
    if probability >= 0.65:
        return "moderate"
    return "low"


def normalize_findings(evidence: dict) -> dict[str, float]:
    """Return validated findings as label -> probability."""
    if "findings" not in evidence or not isinstance(evidence["findings"], dict):
        raise ValueError("Evidence JSON must contain a 'findings' object.")

    findings: dict[str, float] = {}
    for label, value in evidence["findings"].items():
        try:
            probability = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Probability for {label!r} must be numeric.") from exc
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability for {label!r} must be between 0 and 1.")
        findings[label] = probability
    return findings


def parse_json_response(raw_text: str) -> dict:
    """Parse a model response that should be JSON, with light fence cleanup."""
    cleaned = raw_text.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Grok returned non-JSON text: {raw_text}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Grok response must be a JSON object.")
    if "findings" not in parsed or "impression" not in parsed:
        raise ValueError("Grok response must contain 'findings' and 'impression'.")
    if not isinstance(parsed["findings"], str):
        raise ValueError("Grok response field 'findings' must be a string.")
    if isinstance(parsed["impression"], str):
        parsed["impression"] = [
            line.strip() for line in parsed["impression"].splitlines() if line.strip()
        ]
    if not isinstance(parsed["impression"], list):
        raise ValueError("Grok response field 'impression' must be a list or string.")
    parsed["impression"] = [str(item).strip() for item in parsed["impression"] if str(item).strip()]
    return parsed


# ------------------------ LLM narrative generation --------------------------

def generate_narrative(evidence: dict, patient_name: str, model: str = DEFAULT_MODEL) -> dict:
    """Call xAI Grok to produce Findings + Impression text."""
    findings = normalize_findings(evidence)
    image_quality = evidence.get("image_quality", "adequate")

    positives = {
        label: probability
        for label, probability in findings.items()
        if probability >= THRESHOLD
    }
    negatives = {
        label: probability
        for label, probability in findings.items()
        if probability < THRESHOLD
    }

    system_prompt = (
        "You are a careful radiology-report drafting assistant. "
        "You do not diagnose. You convert AI model probabilities into a concise "
        "draft report for clinician review. Use only the supplied probabilities. "
        "Do not invent laterality, severity, devices, history, comparison exams, "
        "measurements, or clinical details that were not provided. Return JSON only."
    )

    user_prompt = f"""
Create a structured draft chest X-ray report for clinician review.

Patient display name: {patient_name}
Exam: Chest X-Ray PA View
Image quality: {image_quality}
Positive threshold: {THRESHOLD:.2f}

Disease label descriptions:
{json.dumps(LABEL_DESCRIPTIONS, indent=2)}

Model probabilities:
{json.dumps(findings, indent=2)}

Positive findings at or above threshold:
{json.dumps(positives, indent=2)}

Negative findings below threshold:
{json.dumps(negatives, indent=2)}

Writing rules:
- Findings must be 3-6 formal sentences.
- Mention positive findings first.
- For probabilities 0.85 or higher, use confident but still non-diagnostic language.
- For probabilities 0.65 to 0.84, use moderate-confidence language like "suggests".
- For probabilities 0.50 to 0.64, use hedged language like "may represent".
- For findings below threshold, state that the model does not identify them when useful.
- Impression must be a prioritized list of short strings.
- If no finding is above threshold, impression should say no AI-flagged acute cardiopulmonary abnormality among the modeled labels.
- Return only JSON with this shape: {{"findings": "...", "impression": ["..."]}}
""".strip()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Run: export GROQ_API_KEY='your_api_key_here'"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=DEFAULT_BASE_URL
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )

    raw_text = response.choices[0].message.content or ""
    return parse_json_response(raw_text)

# ------------------------------- PDF builder --------------------------------

NAVY = colors.HexColor("#1B3A6B")
BLUE = colors.HexColor("#2E6DA4")
LGRAY = colors.HexColor("#F4F6F9")
MGRAY = colors.HexColor("#D0D7E3")
RED = colors.HexColor("#C0392B")
GREEN = colors.HexColor("#1A7A4A")


def build_pdf(
    out_path: str,
    patient_name: str,
    dob: str,
    mrn: str,
    evidence: dict,
    narrative: dict,
    xray_path: str | None = None,
):
    doc = SimpleDocTemplate(
        out_path,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    width = letter[0] - 1.5 * inch

    def style(name, **kwargs):
        return ParagraphStyle(name, parent=styles["Normal"], **kwargs)

    h1 = style("H1", fontSize=18, textColor=NAVY, fontName="Helvetica-Bold", spaceAfter=4)
    h2 = style("H2", fontSize=11, textColor=BLUE, fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=4)
    sub = style("Sub", fontSize=9, textColor=colors.gray)
    body = style("Body", fontSize=10, leading=15, spaceAfter=6)
    disc = style(
        "Disc",
        fontSize=8,
        textColor=colors.HexColor("#8B0000"),
        fontName="Helvetica-Oblique",
        spaceBefore=6,
    )

    story = []

    header_data = [[
        Paragraph(
            "<b>ChestX-AI Assistant</b><br/><font size=9 color='#2E6DA4'>AI-Assisted Radiology Report</font>",
            h1,
        ),
        Paragraph(
            f"<font size=8 color='grey'>Report Date</font><br/><b>{datetime.now().strftime('%B %d, %Y')}</b>",
            sub,
        ),
    ]]
    header_table = Table(header_data, colWidths=[width * 0.7, width * 0.3])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width=width, thickness=2, color=NAVY, spaceAfter=10))

    info_data = [
        ["Patient Name", patient_name, "MRN", mrn],
        ["Date of Birth", dob, "Exam", "Chest X-Ray PA View"],
        ["Referring Physician", "-", "Image Quality", evidence.get("image_quality", "adequate").title()],
    ]
    info_table = Table(info_data, colWidths=[width * 0.2, width * 0.3, width * 0.18, width * 0.32])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LGRAY),
        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
        ("TEXTCOLOR", (2, 0), (2, -1), NAVY),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LGRAY, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, MGRAY),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 12))

    if xray_path and Path(xray_path).exists():
        img = Image(xray_path, width=2.4 * inch, height=2.4 * inch)
        img.hAlign = "LEFT"
        story.append(Paragraph("Submitted Image", h2))
        story.append(img)
        story.append(Spacer(1, 8))

    story.append(Paragraph("Model Findings - Probability Scores", h2))

    findings = normalize_findings(evidence)
    rows = [["Finding", "Probability", "Confidence", "Assessment"]]
    for label, probability in sorted(findings.items(), key=lambda item: -item[1]):
        confidence = confidence_word(probability)
        flag = "Positive" if probability >= THRESHOLD else "Negative"
        color_hex = "#C0392B" if probability >= THRESHOLD else "#1A7A4A"
        rows.append([
            label,
            f"{probability:.1%}",
            confidence.capitalize(),
            Paragraph(f"<font color='{color_hex}'><b>{flag}</b></font>", body),
        ])

    probability_table = Table(rows, colWidths=[width * 0.30, width * 0.18, width * 0.18, width * 0.34])
    probability_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LGRAY]),
        ("GRID", (0, 0), (-1, -1), 0.4, MGRAY),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("ALIGN", (1, 0), (2, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(probability_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Threshold for positive classification: {THRESHOLD:.0%}", sub))
    story.append(Spacer(1, 12))

    story.append(HRFlowable(width=width, thickness=0.5, color=MGRAY))
    story.append(Paragraph("Radiological Findings", h2))
    story.append(Paragraph(escape(narrative["findings"]), body))
    story.append(Spacer(1, 10))

    story.append(HRFlowable(width=width, thickness=0.5, color=MGRAY))
    story.append(Paragraph("Impression", h2))

    impression_items = narrative.get("impression", [])
    if isinstance(impression_items, str):
        impression_items = [line.strip() for line in impression_items.splitlines() if line.strip()]
    else:
        impression_items = [str(item).strip() for item in impression_items if str(item).strip()]

    for index, item in enumerate(impression_items, start=1):
        story.append(Paragraph(f"{index}. {escape(item)}", body))

    story.append(Spacer(1, 16))

    story.append(HRFlowable(width=width, thickness=1, color=colors.HexColor("#C0392B")))
    story.append(Spacer(1, 4))
    disclaimer = (
        "DISCLAIMER: This report is generated by an AI-assisted system and is intended "
        "for research and decision-support purposes only. It must be reviewed and "
        "validated by a licensed radiologist or qualified physician before any clinical "
        "decisions are made. This output does NOT constitute a final medical diagnosis."
    )
    story.append(Paragraph(disclaimer, disc))

    def add_footer(canvas_obj, doc_obj):
        canvas_obj.saveState()
        canvas_obj.setFont("Helvetica", 7)
        canvas_obj.setFillColor(colors.gray)
        canvas_obj.drawString(
            0.75 * inch,
            0.4 * inch,
            f"ChestX-AI Assistant | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | AI-Assisted - Not for standalone clinical use",
        )
        canvas_obj.drawRightString(letter[0] - 0.75 * inch, 0.4 * inch, f"Page {doc_obj.page}")
        canvas_obj.restoreState()

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    print(f"Report saved -> {out_path}")


# --------------------------------- CLI ---------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate a radiology-style PDF report with xAI Grok")
    parser.add_argument("--evidence", default="outputs/reports/sample_evidence.json")
    parser.add_argument("--xray", default=None, help="Path to X-ray image (optional)")
    parser.add_argument("--patient", default="Anonymous Patient")
    parser.add_argument("--dob", default="-")
    parser.add_argument("--mrn", default="N/A")
    parser.add_argument("--out", default="outputs/reports/report.pdf")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="xAI Grok model name")
    args = parser.parse_args()

    with open(args.evidence, "r", encoding="utf-8") as file:
        evidence = json.load(file)

    print(f"Generating narrative via xAI Grok API using model: {args.model}")
    narrative = generate_narrative(evidence, args.patient, model=args.model)

    print("Building PDF...")
    os.makedirs(Path(args.out).parent, exist_ok=True)
    build_pdf(
        out_path=args.out,
        patient_name=args.patient,
        dob=args.dob,
        mrn=args.mrn,
        evidence=evidence,
        narrative=narrative,
        xray_path=args.xray,
    )


if __name__ == "__main__":
    main()
