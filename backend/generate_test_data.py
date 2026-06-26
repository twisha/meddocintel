"""Generate synthetic clinical documents for testing.

Produces 10 records across 4 formats:
  3 × PDF   (reportlab)
  3 × JPG   (Pillow — simulated scan with slight noise)
  3 × PNG   (Pillow — clean scan)
  1 × TXT   (plain text baseline)

NOTE: Audio (.mp3/.wav) is NOT supported by the current pipeline.
      Adding audio would require a speech-to-text layer (e.g. OpenAI Whisper)
      before the OCR stage. Skipped intentionally.

Usage (from meddocintel/backend/):
    pip install reportlab pillow
    python generate_test_data.py

Output: fixtures/clinical_notes/
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "fixtures" / "clinical_notes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Synthetic data pools
# ---------------------------------------------------------------------------

PATIENTS = [
    ("Alice Nguyen",      "02/14/1972", "F", "MRN-100291", "BlueCross-44821"),
    ("Marcus Thompson",   "09/03/1958", "M", "MRN-100292", "Aetna-99321"),
    ("Fatima Al-Hassan",  "11/28/1985", "F", "MRN-100293", "UHC-56701"),
    ("David Kim",         "06/17/1965", "M", "MRN-100294", "Cigna-88120"),
    ("Elena Vasquez",     "03/22/1990", "F", "MRN-100295", "Medicare-34509"),
    ("James O'Brien",     "07/05/1948", "M", "MRN-100296", "Humana-12934"),
    ("Priya Patel",       "12/01/1979", "F", "MRN-100297", "BlueCross-55610"),
    ("Samuel Washington", "04/19/1962", "M", "MRN-100298", "Medicaid-87432"),
    ("Lin Chen",          "08/30/1995", "F", "MRN-100299", "Aetna-20918"),
    ("Robert Martinez",   "01/11/1955", "M", "MRN-100300", "UHC-63401"),
]

PROVIDERS = [
    ("Dr. Sarah Patel, MD",    "Metro Internal Medicine"),
    ("Dr. James O'Brien, MD",  "Riverside Emergency Dept"),
    ("Dr. Emily Chen, MD",     "City Family Practice"),
    ("Dr. Michael Ross, MD",   "Downtown Cardiology"),
    ("Dr. Aisha Johnson, MD",  "Community Health Center"),
]

SCENARIOS = [
    {
        "type": "follow_up",
        "chief_complaint": "Follow-up for Type 2 Diabetes and hypertension",
        "vitals": {"bp": "148/92", "hr": 78, "rr": 16, "temp": 98.6, "wt": 214, "ht": 70, "bmi": 30.7, "o2": 97},
        "medications": [
            ("Metformin",    "1000mg", "PO", "twice daily",   "Type 2 Diabetes"),
            ("Lisinopril",   "20mg",   "PO", "once daily",    "Hypertension"),
            ("Atorvastatin", "40mg",   "PO", "at bedtime",    "Hyperlipidemia"),
            ("Aspirin",      "81mg",   "PO", "daily",         "Cardiovascular prophylaxis"),
        ],
        "diagnoses": [
            ("Type 2 Diabetes Mellitus", "E11.9", "ACTIVE"),
            ("Hypertension, essential",  "I10",   "ACTIVE"),
            ("Hyperlipidemia",           "E78.5", "ACTIVE"),
        ],
        "assessment": "HbA1c 8.4% above target. BP poorly controlled at 148/92.",
        "plan":       "Increase Metformin to 1000mg TID. Add HCTZ 12.5mg. Recheck in 4 weeks.",
    },
    {
        "type": "emergency",
        "chief_complaint": "Chest pain and shortness of breath x 2 hours",
        "vitals": {"bp": "122/78", "hr": 104, "rr": 22, "temp": 99.1, "wt": 137, "ht": 65, "bmi": 22.8, "o2": 94},
        "medications": [
            ("Loratadine", "10mg",  "PO", "daily", "Seasonal allergies"),
            ("Ibuprofen",  "400mg", "PO", "PRN",   "Pain management"),
        ],
        "diagnoses": [
            ("Pleuritic chest pain",         "R07.1",  "ACTIVE"),
            ("Suspected pulmonary embolism", "I26.99", "WORKING"),
            ("Sinus tachycardia",            "R00.0",  "ACTIVE"),
        ],
        "assessment": "Elevated D-dimer 1.4 ng/mL. O2 94% on room air. EKG: sinus tachycardia.",
        "plan":       "CT-PA ordered STAT. Anticoagulation on hold pending imaging. Admit for observation.",
    },
    {
        "type": "wellness",
        "chief_complaint": "Routine annual wellness exam",
        "vitals": {"bp": "118/76", "hr": 68, "rr": 14, "temp": 98.4, "wt": 155, "ht": 66, "bmi": 25.0, "o2": 99},
        "medications": [
            ("Levothyroxine", "75mcg", "PO", "once daily",   "Hypothyroidism"),
            ("Calcium",       "500mg", "PO", "twice daily",  "Bone health"),
        ],
        "diagnoses": [
            ("Hypothyroidism",       "E03.9", "ACTIVE, well-controlled"),
            ("Annual wellness exam", "Z00.00","COMPLETED"),
        ],
        "assessment": "TSH 2.1, within normal range. BMI 25.0 borderline. No acute concerns.",
        "plan":       "Continue Levothyroxine. Recheck TSH in 6 months. 30min daily exercise.",
    },
    {
        "type": "acute_visit",
        "chief_complaint": "Productive cough and fever for 3 days",
        "vitals": {"bp": "132/84", "hr": 96, "rr": 20, "temp": 101.8, "wt": 180, "ht": 72, "bmi": 24.4, "o2": 96},
        "medications": [
            ("Azithromycin", "500mg", "PO", "once daily x 5 days",    "Community-acquired pneumonia"),
            ("Guaifenesin",  "400mg", "PO", "every 4 hours PRN",      "Productive cough"),
            ("Acetaminophen","650mg", "PO", "every 6 hours PRN",      "Fever/pain"),
        ],
        "diagnoses": [
            ("Community-acquired pneumonia", "J18.9", "ACTIVE"),
            ("Fever",                        "R50.9", "ACTIVE"),
        ],
        "assessment": "CXR: right lower lobe infiltrate. WBC 13.2.",
        "plan":       "Azithromycin 500mg x 5 days. Return if no improvement in 48h.",
    },
    {
        "type": "cardiology",
        "chief_complaint": "Cardiac follow-up post-MI, 6 weeks",
        "vitals": {"bp": "126/80", "hr": 62, "rr": 14, "temp": 98.2, "wt": 198, "ht": 69, "bmi": 29.2, "o2": 98},
        "medications": [
            ("Metoprolol",   "50mg", "PO", "twice daily",  "Post-MI rate control"),
            ("Clopidogrel",  "75mg", "PO", "once daily",   "Antiplatelet therapy"),
            ("Atorvastatin", "80mg", "PO", "at bedtime",   "Post-MI statin therapy"),
            ("Lisinopril",   "10mg", "PO", "once daily",   "Post-MI cardioprotection"),
            ("Aspirin",      "81mg", "PO", "daily",        "Antiplatelet therapy"),
        ],
        "diagnoses": [
            ("ST-elevation MI, anterior, resolved", "I21.09", "RESOLVED"),
            ("Coronary artery disease",             "I25.10", "ACTIVE"),
            ("Hypertension",                        "I10",    "CONTROLLED"),
        ],
        "assessment": "6-week post-STEMI. EF 45% on echo. BP well-controlled. No chest pain.",
        "plan":       "Continue dual antiplatelet x 12 months. Cardiac rehab. Echo in 3 months.",
    },
    {
        "type": "neurology",
        "chief_complaint": "Severe migraine with visual aura, 6 hours",
        "vitals": {"bp": "138/88", "hr": 88, "rr": 16, "temp": 98.8, "wt": 142, "ht": 64, "bmi": 24.4, "o2": 99},
        "medications": [
            ("Sumatriptan", "100mg", "PO",  "at onset PRN",  "Migraine abort"),
            ("Ondansetron", "4mg",   "PO",  "every 6h PRN",  "Nausea"),
            ("Ketorolac",   "30mg",  "IV",  "x1 dose",       "Acute pain"),
        ],
        "diagnoses": [
            ("Migraine with aura",  "G43.109", "ACTIVE"),
            ("Nausea and vomiting", "R11.2",   "ACTIVE, secondary"),
        ],
        "assessment": "Classic migraine with visual aura. No meningismus. Neuro exam intact.",
        "plan":       "IV Ketorolac and antiemetic in ED. Sumatriptan for home. Neurology referral.",
    },
    {
        "type": "pulmonology",
        "chief_complaint": "COPD exacerbation follow-up, 2 weeks post-discharge",
        "vitals": {"bp": "140/88", "hr": 84, "rr": 18, "temp": 98.5, "wt": 168, "ht": 67, "bmi": 26.3, "o2": 93},
        "medications": [
            ("Tiotropium",  "18mcg",  "INH", "once daily",    "COPD maintenance"),
            ("Albuterol",   "90mcg",  "INH", "every 4-6h PRN","COPD rescue"),
            ("Fluticasone", "250mcg", "INH", "twice daily",   "COPD anti-inflammatory"),
            ("Prednisone",  "40mg",   "PO",  "daily x 3 days","COPD exacerbation taper"),
        ],
        "diagnoses": [
            ("COPD, moderate, exacerbation", "J44.1",  "ACTIVE"),
            ("Tobacco use disorder",         "F17.210","ACTIVE"),
            ("Hypoxemia",                    "J96.01", "IMPROVING"),
        ],
        "assessment": "O2 improved to 93% from 88% at discharge. SOB with exertion. Cough productive.",
        "plan":       "Complete prednisone taper. Pulmonary rehab. Smoking cessation counseling.",
    },
    {
        "type": "psychiatry",
        "chief_complaint": "Depression and anxiety medication management",
        "vitals": {"bp": "116/74", "hr": 72, "rr": 14, "temp": 98.3, "wt": 134, "ht": 65, "bmi": 22.3, "o2": 100},
        "medications": [
            ("Sertraline", "100mg", "PO", "once daily",   "Major depressive disorder"),
            ("Lorazepam",  "0.5mg", "PO", "PRN anxiety",  "Generalized anxiety disorder"),
            ("Melatonin",  "5mg",   "PO", "at bedtime",   "Insomnia"),
        ],
        "diagnoses": [
            ("Major depressive disorder", "F33.1", "ACTIVE, improving"),
            ("Generalized anxiety disorder","F41.1","ACTIVE"),
            ("Insomnia",                  "G47.00","ACTIVE"),
        ],
        "assessment": "PHQ-9 score 10 (moderate), down from 16. Improved sleep and mood.",
        "plan":       "Continue Sertraline 100mg. Taper Lorazepam. CBT referral. Follow-up 4 weeks.",
    },
    {
        "type": "orthopedics",
        "chief_complaint": "Acute lower back pain after lifting, onset today",
        "vitals": {"bp": "128/82", "hr": 80, "rr": 16, "temp": 98.6, "wt": 210, "ht": 71, "bmi": 29.3, "o2": 99},
        "medications": [
            ("Cyclobenzaprine","10mg",  "PO",  "three times daily PRN",  "Muscle spasm"),
            ("Naproxen",       "500mg", "PO",  "twice daily with food",   "Pain/inflammation"),
            ("Lidocaine 5%",   "patch", "TOP", "12h on / 12h off",       "Local pain relief"),
        ],
        "diagnoses": [
            ("Acute lumbar strain",  "M54.5",   "ACTIVE"),
            ("Muscle spasm, lumbar", "M62.830", "ACTIVE"),
        ],
        "assessment": "No radiculopathy. Negative SLR bilaterally. Paraspinal tenderness L3-L5.",
        "plan":       "NSAIDs + muscle relaxant + lidocaine patch. No heavy lifting x 2 weeks. PT referral.",
    },
    {
        "type": "nephrology",
        "chief_complaint": "Chronic kidney disease stage 3 management",
        "vitals": {"bp": "142/90", "hr": 74, "rr": 16, "temp": 98.4, "wt": 176, "ht": 68, "bmi": 26.8, "o2": 98},
        "medications": [
            ("Amlodipine",    "10mg",   "PO", "once daily",   "Hypertension/CKD renoprotection"),
            ("Furosemide",    "40mg",   "PO", "once daily",   "Volume management"),
            ("Sodium bicarb", "650mg",  "PO", "twice daily",  "Metabolic acidosis"),
            ("Calcitriol",    "0.25mcg","PO", "once daily",   "Renal osteodystrophy prevention"),
        ],
        "diagnoses": [
            ("Chronic kidney disease, stage 3", "N18.3", "ACTIVE"),
            ("Hypertension",                    "I10",   "ACTIVE, poorly controlled"),
            ("Metabolic acidosis",              "E87.2", "ACTIVE"),
            ("Anemia of CKD",                  "D63.1", "ACTIVE"),
        ],
        "assessment": "eGFR 42 (stable). BP 142/90, target <130/80. Hgb 10.2.",
        "plan":       "Nephrology referral. Increase Amlodipine to 10mg. Low-sodium, low-protein diet.",
    },
]

# Format distribution: (format, label)
FORMATS = [
    ("pdf", "reportlab PDF"),
    ("pdf", "reportlab PDF"),
    ("pdf", "reportlab PDF"),
    ("jpg", "Pillow JPG scan"),
    ("jpg", "Pillow JPG scan"),
    ("jpg", "Pillow JPG scan"),
    ("png", "Pillow PNG scan"),
    ("png", "Pillow PNG scan"),
    ("png", "Pillow PNG scan"),
    ("txt", "plain text"),
]


# ---------------------------------------------------------------------------
# Text builder
# ---------------------------------------------------------------------------

def build_text(patient, provider, visit_date, scenario) -> str:
    name, dob, gender, mrn, ins = patient
    doc_provider, facility = provider

    visit_type = {
        "emergency": "Emergency Visit",
        "follow_up": "Follow-up Visit",
        "wellness":  "Annual Wellness Exam",
    }.get(scenario["type"], "Acute Care Visit")

    v = scenario["vitals"]
    meds = "\n".join(
        f"  {i+1}. {m[0]} {m[1]} {m[2]} {m[3]}  —  {m[4]}"
        for i, m in enumerate(scenario["medications"])
    )
    dx = "\n".join(
        f"  - {d[0]} ({d[1]})  —  {d[2]}" for d in scenario["diagnoses"]
    )

    return f"""CLINICAL PROGRESS NOTE
{'='*62}
Patient Name:  {name:<30}  DOB: {dob}
MRN:           {mrn:<30}  Gender: {gender}
Insurance ID:  {ins}

Visit Date:  {visit_date}               Visit Type: {visit_type}
Provider:    {doc_provider}
Facility:    {facility}

CHIEF COMPLAINT:
  {scenario['chief_complaint']}

VITAL SIGNS:
  BP: {v['bp']} mmHg     HR: {v['hr']} bpm       RR: {v['rr']} breaths/min
  Temp: {v['temp']} F      Wt: {v['wt']} lbs       Ht: {v['ht']} in
  BMI: {v['bmi']}          O2 Sat: {v['o2']}%

CURRENT MEDICATIONS:
{meds}

DIAGNOSES / PROBLEM LIST:
{dx}

ASSESSMENT:
  {scenario['assessment']}

PLAN:
  {scenario['plan']}

Electronically signed by: {doc_provider}
Date/Time: {visit_date} {random.randint(8,17):02d}:{random.randint(0,59):02d} EST
{'='*62}
"""


# ---------------------------------------------------------------------------
# PDF writer
# ---------------------------------------------------------------------------

def write_pdf(path: Path, text: str):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    doc = SimpleDocTemplate(
        str(path), pagesize=letter,
        leftMargin=inch, rightMargin=inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    mono  = ParagraphStyle("mono",  fontName="Courier",      fontSize=9,  leading=13)
    bold  = ParagraphStyle("bold",  fontName="Courier-Bold",  fontSize=9,  leading=13,
                           textColor=colors.HexColor("#1a365d"))
    story = []
    for line in text.split("\n"):
        safe = line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        style = bold if (line.strip().isupper() and len(line.strip()) > 3 and not line.startswith("=")) else mono
        story.append(Paragraph(safe or "&nbsp;", style))
    doc.build(story)


# ---------------------------------------------------------------------------
# Image writer (JPG / PNG)
# ---------------------------------------------------------------------------

def write_image(path: Path, text: str, fmt: str):
    from PIL import Image, ImageDraw, ImageFilter
    import io

    # Page size at 150 dpi
    W, H = 1275, 1650
    img = Image.new("RGB", (W, H), color=(252, 252, 252))
    draw = ImageDraw.Draw(img)

    # Try to load a monospace system font; fall back to default
    font = None
    font_size = 22
    font_paths = [
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/System/Library/Fonts/Monaco.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    try:
        from PIL import ImageFont
        for fp in font_paths:
            if Path(fp).exists():
                font = ImageFont.truetype(fp, font_size)
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        pass

    margin_x, margin_y = 60, 60
    line_h = font_size + 6 if font else 18
    y = margin_y

    for line in text.split("\n"):
        if y + line_h > H - margin_y:
            break
        color = (20, 40, 100) if (line.strip().isupper() and len(line.strip()) > 3) else (30, 30, 30)
        draw.text((margin_x, y), line, fill=color, font=font)
        y += line_h

    # Simulate scan artifacts for JPG (slight noise + blur)
    if fmt == "jpg":
        import numpy as np
        arr = np.array(img).astype(np.int16)
        noise = np.random.randint(-6, 6, arr.shape, dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
        img.save(str(path), "JPEG", quality=88)
    else:
        img.save(str(path), "PNG")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base_date = datetime(2025, 1, 15)
    generated = []

    print(f"Output directory: {OUTPUT_DIR}\n")

    for i, (patient, scenario, (fmt, label)) in enumerate(zip(PATIENTS, SCENARIOS, FORMATS)):
        visit_date = (base_date + timedelta(days=i * 7)).strftime("%m/%d/%Y")
        provider   = PROVIDERS[i % len(PROVIDERS)]
        text       = build_text(patient, provider, visit_date, scenario)
        fname      = f"synthetic_{i+1:03d}_{scenario['type']}.{fmt}"
        fpath      = OUTPUT_DIR / fname

        try:
            if fmt == "pdf":
                write_pdf(fpath, text)
            elif fmt in ("jpg", "png"):
                write_image(fpath, text, fmt)
            else:
                fpath.write_text(text, encoding="utf-8")

            size_kb = fpath.stat().st_size // 1024
            print(f"  [{fmt.upper():3s}] {fname}  ({size_kb} KB)")
            generated.append(fname)

        except Exception as e:
            # Fallback to txt if a dependency is missing
            fallback = fpath.with_suffix(".txt")
            fallback.write_text(text, encoding="utf-8")
            print(f"  [TXT] {fallback.name}  (fallback — {e})")
            generated.append(fallback.name)

    print(f"\n✓ {len(generated)} files written to {OUTPUT_DIR}")
    print("\nNOTE: Audio (.mp3/.wav) is not supported by this pipeline.")
    print("      To add audio support, a speech-to-text layer (e.g. Whisper) is needed")
    print("      before the OCR stage.")


if __name__ == "__main__":
    main()
