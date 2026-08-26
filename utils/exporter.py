"""
Export utilities — Excel and PDF report generation.
"""

import io
import pandas as pd
from datetime import datetime


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "SIF_Reports") -> bytes:
    """Convert a DataFrame to Excel bytes for Streamlit download."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # Header format
        header_fmt = workbook.add_format({
            "bold": True, "bg_color": "#1a3a5c", "font_color": "#FFFFFF",
            "border": 1, "align": "center",
        })
        sif_fmt = workbook.add_format({"bg_color": "#f8d7da"})
        ok_fmt  = workbook.add_format({"bg_color": "#d4edda"})

        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_fmt)
            worksheet.set_column(col_num, col_num, max(15, len(str(col_name)) + 4))

        # Colour rows by SIF potential
        if "sif_potential" in df.columns:
            for row_num, val in enumerate(df["sif_potential"], start=1):
                fmt = sif_fmt if val else ok_fmt
                worksheet.set_row(row_num, None, fmt)

    return output.getvalue()


def to_pdf_bytes(df: pd.DataFrame, title: str = "SIF Precursor Report") -> bytes:
    """Generate a simple PDF summary using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=landscape(A4), topMargin=1*cm, bottomMargin=1*cm)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph(title, styles["Title"]))
        elements.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Total Records: {len(df)}",
            styles["Normal"],
        ))
        elements.append(Spacer(1, 0.5*cm))

        # Limit columns for PDF
        cols = ["report_id", "date", "site", "report_type", "sif_potential", "sif_score", "confidence", "life_saving_rules"]
        cols = [c for c in cols if c in df.columns]
        sub = df[cols].head(100)

        data = [cols] + sub.values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTSIZE",   (0, 0), (-1, -1), 7),
            ("GRID",       (0, 0), (-1, -1), 0.25, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        elements.append(table)
        doc.build(elements)
        return output.getvalue()
    except ImportError:
        # Fallback: return empty bytes if reportlab not installed
        return b""
