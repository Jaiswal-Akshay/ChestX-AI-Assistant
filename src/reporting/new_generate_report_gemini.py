import argparse
import json
import os
from pathlib import Path
from datetime import datetime

from google import genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

DEFAULT_MODEL = "gemini-2.0-flash"


# ---------------- Gemini Narrative ---------------- #

def generate_narrative(evidence: dict, patient_name: str, model: str):
    api_key = os.getenv("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a radiologist.

Given these disease probabilities:
{json.dumps(evidence["findings"], indent=2)}

Write a structured radiology report with:

Findings:
Impression:

Keep it concise and professional.
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    text = response.text

    return {
        "findings": text,
        "impression": "See findings above"
    }


# ---------------- PDF Builder ---------------- #

def build_pdf(out_path, patient_name, dob, mrn, evidence, narrative):
    doc = SimpleDocTemplate(out_path, pagesize=letter)
    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("<b>Chest X-Ray Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Patient: {patient_name}", styles["Normal"]))
    story.append(Paragraph(f"DOB: {dob}", styles["Normal"]))
    story.append(Paragraph(f"MRN: {mrn}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Findings</b>", styles["Heading2"]))
    story.append(Paragraph(narrative["findings"], styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Impression</b>", styles["Heading2"]))
    story.append(Paragraph(narrative["impression"], styles["Normal"]))

    doc.build(story)
    print(f"✅ Report saved → {out_path}")


# ---------------- MAIN ---------------- #

def main():
    parser = argparse.ArgumentParser(description="Generate report using Gemini")

    parser.add_argument("--evidence", default="outputs/reports/sample_evidence.json")
    parser.add_argument("--patient", default="Anonymous Patient")
    parser.add_argument("--dob", default="—")
    parser.add_argument("--mrn", default="N/A")
    parser.add_argument("--out", default="outputs/reports/report.pdf")
    parser.add_argument("--model", default=DEFAULT_MODEL)

    args = parser.parse_args()

    # ✅ API KEY CHECK
    api_key = os.getenv("GEMINI_API_KEY")
    print("ENV CHECK:", api_key is not None)

    if api_key is None or api_key.strip() == "":
        raise RuntimeError("GEMINI_API_KEY is not set")

    # ✅ Load evidence
    with open(args.evidence, "r", encoding="utf-8") as file:
        evidence = json.load(file)

    # ✅ Generate report text
    print("Generating narrative...")
    narrative = generate_narrative(evidence, args.patient, model=args.model)

    # ✅ Build PDF
    print("Building PDF...")
    os.makedirs(Path(args.out).parent, exist_ok=True)

    build_pdf(
        out_path=args.out,
        patient_name=args.patient,
        dob=args.dob,
        mrn=args.mrn,
        evidence=evidence,
        narrative=narrative
    )


if __name__ == "__main__":
    main()