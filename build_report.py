"""Build the lab report PDF from the included project report content/design."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "project_report.pdf"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
if (FONT_DIR / "DejaVuSerif.ttf").exists():
    pdfmetrics.registerFont(TTFont("ReportSerif", str(FONT_DIR / "DejaVuSerif.ttf")))
    pdfmetrics.registerFont(TTFont("ReportSerif-Bold", str(FONT_DIR / "DejaVuSerif-Bold.ttf")))
    FONT, BOLD = "ReportSerif", "ReportSerif-Bold"
else:
    FONT, BOLD = "Times-Roman", "Times-Bold"

BLUE = colors.HexColor("#dbeaf5")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitleX", fontName=BOLD, fontSize=17, leading=23, alignment=TA_CENTER, spaceAfter=8))
styles.add(ParagraphStyle(name="CoverTextX", fontName=BOLD, fontSize=12, leading=18, alignment=TA_CENTER, spaceAfter=4))
styles.add(ParagraphStyle(name="SectionX", fontName=BOLD, fontSize=12.4, leading=16, spaceBefore=7, spaceAfter=5, keepWithNext=True))
styles.add(ParagraphStyle(name="SubsectionX", fontName=BOLD, fontSize=10.3, leading=13, spaceBefore=5, spaceAfter=3, keepWithNext=True))
styles.add(ParagraphStyle(name="BodyX", fontName=FONT, fontSize=8.5, leading=11.2, alignment=TA_JUSTIFY, spaceAfter=5))
styles.add(ParagraphStyle(name="BodyLeftX", fontName=FONT, fontSize=8.5, leading=11.2, alignment=TA_LEFT, spaceAfter=3))
styles.add(ParagraphStyle(name="SmallX", fontName=FONT, fontSize=7.5, leading=9.3, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="TableHeadX", fontName=BOLD, fontSize=8.2, leading=9.5, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="QuoteX", fontName=BOLD, fontSize=8.2, leading=10.2, alignment=TA_LEFT, textColor=colors.HexColor("#6b1d1d"), borderColor=colors.HexColor("#ddbaba"), borderWidth=0.5, borderPadding=5, spaceBefore=4, spaceAfter=7))


def P(text, style="BodyX"):
    return Paragraph(text, styles[style])


def bullet(text):
    return P("•&nbsp;&nbsp;" + text, "BodyLeftX")


def table(data, widths, header=True, font_size=8):
    formatted = []
    for r, row in enumerate(data):
        formatted.append([cell if isinstance(cell, Paragraph) else P(str(cell), "TableHeadX" if header and r == 0 else "SmallX") for cell in row])
    t = Table(formatted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.55, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), BLUE))
    t.setStyle(TableStyle(commands))
    return t


def footer(canvas, doc):
    if doc.page > 1:
        canvas.saveState()
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(20 * mm, 12 * mm, "Digital Image Processing Lab | Breast Histopathology Classifier")
        canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, str(doc.page))
        canvas.restoreState()


def build():
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        rightMargin=20 * mm, leftMargin=20 * mm,
        topMargin=17 * mm, bottomMargin=19 * mm,
        title="Breast Cancer Detection from Histopathology Images Using DIP Features and SVM",
        author="Noushin Ulfat Shukannya",
    )
    story = []
    # Cover page, styled after the sample report (without reproducing its institutional seal).
    story.extend([Spacer(1, 15 * mm), P("Project Report", "CoverTitleX"),
                  P("Course Title: Digital Image Processing Lab", "CoverTextX"),
                  P("Course Code: ICT-4202", "CoverTextX"), Spacer(1, 12 * mm),
                  P("Submitted to,", "CoverTextX"), P("Dr. Fahima Tabassum", "CoverTextX"),
                  P("Professor", "CoverTextX"),
                  P("Institute of Information Technology, Jahangirnagar University", "CoverTextX"),
                  Spacer(1, 17 * mm), P("Submitted by,", "CoverTextX"), Spacer(1, 3 * mm)])
    cover_rows = [
        [P("Name:", "TableHeadX"), P("Noushin Ulfat Shukannya", "TableHeadX")],
        [P("Class Roll:", "TableHeadX"), P("2025", "TableHeadX")],
        [P("Exam Roll:", "TableHeadX"), P("201746", "TableHeadX")],
    ]
    cover = Table(cover_rows, colWidths=[78 * mm, 92 * mm], rowHeights=[13 * mm] * 3)
    cover.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    story.extend([cover, PageBreak()])

    # Page 2: overview, objectives, and algorithm.
    story.extend([
        P("Project Name: Breast Cancer Detection from Histopathology Images Using DIP Features and SVM", "SectionX"),
        P("1. Project Overview", "SectionX"),
        P("This project implements an educational binary image-classification system for H&amp;E-stained breast-tissue histopathology microscopy images. It preprocesses an input tissue image, enhances local contrast, extracts color, texture, and edge descriptors, and uses a Support Vector Machine (SVM) trained on benign and malignant examples to estimate a class. The system provides a CLI for training, evaluation, and inference and an optional local Streamlit user interface. It classifies histopathology images, not mammograms. The BreaKHis dataset is recommended but is not included in the ZIP; it must be obtained separately under the dataset terms.", "BodyX"),
        P("This is an educational prototype, not a medical device. It has not been clinically validated and must not be used for diagnosis, screening, or treatment decisions.", "QuoteX"),
        P("2. Project Objectives", "SectionX"),
        bullet("Apply digital image processing to prepare histopathology images for classification."),
        bullet("Extract color, texture, and edge features from tissue microscopy images."),
        bullet("Train a benign/malignant classifier with balanced class weights."),
        bullet("Evaluate a held-out split using accuracy, malignant precision/recall/F1, and a confusion matrix."),
        bullet("Save a reusable model and support inference with JSON and a visual panel."),
        P("3. Algorithm Used: CLAHE, HOG, LBP, Histograms, and SVM", "SectionX"),
        P("Images are orientation-corrected, converted to RGB, and resized to 128 × 128. Grayscale conversion followed by Contrast Limited Adaptive Histogram Equalization (CLAHE) improves local contrast. The feature vector combines Histogram of Oriented Gradients (HOG) for edge structure, uniform Local Binary Patterns (LBP) for local texture, RGB histograms for stain/color distribution, and an enhanced grayscale histogram for global intensity. The vector is standardized and classified with an RBF-kernel SVM (C = 10, balanced class weights).", "BodyX"),
        P("Why these methods fit the project", "SubsectionX"),
        bullet("The preprocessing and descriptors demonstrate classical DIP methods directly."),
        bullet("Complementary color, texture, and edge cues represent tissue appearance."),
        bullet("SVM training and inference run on a CPU; no GPU is required."),
        PageBreak(),
    ])

    # Page 3: workflow and training configuration.
    story.extend([P("4. System Workflow and Methodology", "SectionX"),
                  P("Workflow Diagram", "SubsectionX")])
    diagram = ROOT / "workflow.png"
    if diagram.exists():
        from PIL import Image as PILImage
        with PILImage.open(diagram) as im:
            ratio = im.height / im.width
        story.append(Image(str(diagram), width=170 * mm, height=170 * mm * ratio))
    story.extend([
        P("Workflow Explanation", "SubsectionX"),
        P("<b>Dataset preparation:</b> Images are found recursively from class directories named benign/malignant, or the standard BreaKHis layout. <b>Preprocessing:</b> Images are resized to 128 × 128, converted to grayscale, and locally contrast-enhanced with CLAHE. <b>Features and model:</b> HOG, uniform LBP, RGB/intensity histograms are standardized and passed to a balanced RBF SVM. <b>Evaluation:</b> Original BreaKHis patient IDs are grouped where the file naming pattern is recognized; otherwise the program uses an image-level stratified split, which may not prevent patient leakage. <b>Inference:</b> The saved checkpoint is applied to new tissue images and outputs predictions and visual panels.", "BodyX"),
        P("5. Main Training Configuration", "SectionX"),
        table([
            ["Parameter", "Value"],
            ["Task / input", "Binary classification; RGB H&E histopathology image"],
            ["Image size / enhancement", "128 × 128; grayscale CLAHE (clip limit 0.025)"],
            ["Features", "HOG + uniform LBP + RGB and intensity histograms"],
            ["Classifier", "StandardScaler + RBF SVM (C=10, class_weight=balanced)"],
            ["Holdout / seed", "80/20 default; seed 42; patient grouping for standard BreaKHis names"],
            ["Dataset / hardware", "BreaKHis (download separately); CPU"],
            ["Outputs", "Joblib checkpoint and JSON evaluation metrics"],
        ], [43 * mm, 127 * mm]),
        PageBreak(),
    ])

    # Page 4: honest results, inference, tools, conclusion.
    story.extend([
        P("6. Evaluation and Results", "SectionX"),
        P("No real-data experiment was run for this delivered package because the BreaKHis image dataset was not present in the report-generation environment. Therefore, no scientific accuracy or clinical-performance value is claimed. A software smoke test using generated fixture images checks loading, feature extraction, model serialization, and prediction paths only; these fixtures are not tissue images and are not scientific results.", "BodyX"),
        table([["Metric", "Status"],
               ["Accuracy / malignant precision / recall / F1", "Not measured on real histopathology data"],
               ["Confusion matrix", "Generated after training on the user's dataset"],
               ["Smoke test", "Passed (software functionality only)"],
              ], [90 * mm, 80 * mm]),
        P("Run the training command in the README to create real dataset-specific metrics. A single holdout is exploratory; report the exact split method and use independent data before making broader claims.", "BodyX"),
        P("7. Inference Procedure", "SectionX"),
        bullet("Train with `python -m breast_cancer_detector train --data /path/to/BreaKHis_v1`."),
        bullet("Predict with `python -m breast_cancer_detector predict --model models/breast_cancer_svm.joblib --image /path/to/image.png --output outputs`."),
        bullet("Review the output label, estimated probabilities, JSON result, and input/enhanced image panel cautiously."),
        bullet("Optionally start the local UI using `streamlit run app.py`."),
        P("8. Tools and Technologies", "SectionX"),
        table([["Component", "Role"],
               ["Python, Pillow, NumPy", "Project language, image I/O and arrays"],
               ["scikit-image", "CLAHE, HOG and LBP feature extraction"],
               ["scikit-learn, joblib", "SVM training, metrics, model persistence"],
               ["Matplotlib, Streamlit, pytest", "Visualization, optional UI, software smoke tests"],
               ["BreaKHis", "External licensed histopathology dataset; not bundled"],
              ], [55 * mm, 115 * mm]),
        P("9. Conclusion and Scope for Improvement", "SectionX"),
        P("The project supplies an end-to-end, runnable DIP workflow for binary breast histopathology image classification, including preprocessing, feature extraction, SVM training, evaluation, saved checkpoints, command-line inference, visualization, and an optional UI. Real-data performance must be measured after obtaining BreaKHis. Future work should add repeated patient-wise cross-validation, independent external testing, stain normalization, magnification-aware evaluation, comparison with other methods, and expert pathology review. The system is not for medical use.", "BodyX"),
        P("Dataset reference: F. Spanhol et al., “A Dataset for Breast Cancer Histopathological Image Classification,” IEEE Transactions on Biomedical Engineering, 63(7), 1455–1462, 2016. DOI: 10.1109/TBME.2015.2496264. Official dataset: web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/.", "SmallX"),
    ])
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
