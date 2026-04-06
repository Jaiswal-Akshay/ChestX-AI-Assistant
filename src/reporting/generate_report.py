import json
import os

INPUT_JSON = "outputs/reports/sample_evidence.json"
OUTPUT_TXT = "outputs/reports/generated_report.txt"


def interpret_probability(prob: float) -> str:
    if prob >= 0.70:
        return "likely"
    elif prob >= 0.40:
        return "possible"
    else:
        return "unlikely"


def build_findings_section(findings: dict) -> str:
    lines = []

    for label, prob in findings.items():
        level = interpret_probability(prob)

        if level == "likely":
            lines.append(f"{label} is likely present (probability {prob:.2f}).")
        elif level == "possible":
            lines.append(f"{label} is possibly present (probability {prob:.2f}).")
        else:
            lines.append(f"No strong model support for {label.lower()} (probability {prob:.2f}).")

    return " ".join(lines)


def build_impression_section(findings: dict) -> str:
    likely = []
    possible = []

    for label, prob in findings.items():
        level = interpret_probability(prob)
        if level == "likely":
            likely.append(label)
        elif level == "possible":
            possible.append(label)

    if likely:
        impression = "Likely findings: " + ", ".join(likely) + "."
        if possible:
            impression += " Additional possible findings: " + ", ".join(possible) + "."
        return impression

    if possible:
        return "No high-confidence finding detected. Possible findings include: " + ", ".join(possible) + "."

    return "No high-confidence abnormality detected from model outputs."


def build_limitations_section(image_quality: str) -> str:
    base = (
        "This is an AI decision-support output generated from CNN probabilities only. "
        "It is not a diagnosis and should not be used as a standalone clinical decision tool. "
        "Clinical correlation is recommended."
    )

    if image_quality != "adequate":
        base += f" Report reliability may be reduced because image quality was flagged as {image_quality}."

    return base


def build_patient_summary(findings: dict) -> str:
    top_label = max(findings, key=findings.get)
    top_prob = findings[top_label]
    level = interpret_probability(top_prob)

    if level == "likely":
        return (
            f"The AI system thinks the strongest finding is {top_label.lower()}. "
            "This is only a support result and should be reviewed by a clinician."
        )
    elif level == "possible":
        return (
            f"The AI system sees a possible sign of {top_label.lower()}, but the result is uncertain. "
            "A clinician should review the image."
        )
    else:
        return (
            "The AI system did not detect any strong abnormal finding, but this does not rule out disease. "
            "A clinician should still review the image."
        )


def main():
    os.makedirs("outputs/reports", exist_ok=True)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        evidence = json.load(f)

    findings = evidence["findings"]
    image_quality = evidence.get("image_quality", "adequate")

    findings_text = build_findings_section(findings)
    impression_text = build_impression_section(findings)
    limitations_text = build_limitations_section(image_quality)
    patient_summary_text = build_patient_summary(findings)

    full_report = f"""
FINDINGS
{findings_text}

IMPRESSION
{impression_text}

LIMITATIONS
{limitations_text}

PATIENT SUMMARY
{patient_summary_text}
""".strip()

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(full_report)

    print("\nGenerated Report:\n")
    print(full_report)
    print(f"\nSaved report to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()