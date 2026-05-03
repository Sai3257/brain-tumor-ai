from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from datetime import datetime
import os


def generate_pdf_report(file_path, data):
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # =========================
    # HEADER
    # =========================
    elements.append(Paragraph("<b>APOLLO AI DIAGNOSTIC CENTER</b>", styles["Title"]))
    elements.append(Paragraph("Brain Tumor Analysis Report", styles["Heading2"]))
    elements.append(Spacer(1, 0.3 * inch))

    # =========================
    # REPORT INFO
    # =========================
    elements.append(Paragraph(f"Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')}", styles["Normal"]))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # =========================
    # TABLE DATA
    # =========================
    table_data = [
        ["Tumor Type", data["tumor"]],
        ["Confidence", f'{data["confidence"]}%'],
        ["Severity", data["severity"]],
        ["Risk Level", data["risk"]],
    ]

    table = Table(table_data, colWidths=[2.5 * inch, 3 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.5 * inch))

    # =========================
    # HEATMAP IMAGE (OPTIONAL)
    # =========================
    if "heatmap_path" in data and os.path.exists(data["heatmap_path"]):
        elements.append(Paragraph("Grad-CAM Tumor Localization", styles["Heading3"]))
        elements.append(Spacer(1, 0.2 * inch))

        img = Image(data["heatmap_path"], width=4 * inch, height=4 * inch)
        elements.append(img)
        elements.append(Spacer(1, 0.5 * inch))

    # =========================
    # FOOTER
    # =========================
    elements.append(Paragraph("AI-Assisted Diagnosis. Please consult a neurologist for confirmation.", styles["Normal"]))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("Authorized Medical Officer Signature: ____________________", styles["Normal"]))

    doc.build(elements)
