import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header rule & title
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 750, 558, 750)
        self.drawString(54, 755, "CrediPredict ML — Comprehensive Project Documentation")
        
        # Footer rule & page number
        self.line(54, 45, 558, 45)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_text)
        self.drawString(54, 32, "Confidential — Educational & Technical Reference Manual")
        self.restoreState()

def markdown_to_pdf(md_file_path, output_pdf_path):
    print(f"Converting {md_file_path} -> {output_pdf_path}...")
    
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0284c7"),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2563eb"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=body_style,
        fontName='Helvetica-Oblique',
        textColor=colors.HexColor("#1e40af"),
        leftIndent=12,
        spaceBefore=4,
        spaceAfter=4
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )

    story = []
    lines = md_text.split('\n')
    i = 0
    in_table = False
    table_lines = []

    def clean_text(t):
        # Escape XML entities for ReportLab Paragraph
        t = t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # Bold
        t = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', t)
        # Italic
        t = re.sub(r'\*(.*?)\*', r'<i>\1</i>', t)
        # Code
        t = re.sub(r'`(.*?)`', r'<font fontName="Courier" color="#b91c1c">\1</font>', t)
        # Latex math approximation
        t = re.sub(r'\$(.*?)\$', r'<i>\1</i>', t)
        return t

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        # Handle Markdown Tables
        if stripped.startswith('|') and stripped.endswith('|'):
            table_lines.append(stripped)
            i += 1
            continue
        elif table_lines:
            # Process accumulated table
            table_data = []
            for r_idx, t_line in enumerate(table_lines):
                if '---' in t_line:
                    continue  # skip separator
                parts = [c.strip() for c in t_line.strip('|').split('|')]
                row_cells = []
                for c_idx, p in enumerate(parts):
                    p_clean = clean_text(p)
                    if r_idx == 0:
                        row_cells.append(Paragraph(f"<b>{p_clean}</b>", table_header_style))
                    else:
                        row_cells.append(Paragraph(p_clean, table_cell_style))
                table_data.append(row_cells)

            if table_data:
                t = Table(table_data, hAlign='LEFT')
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(t)
                story.append(Spacer(1, 8))
            table_lines = []

        if not stripped:
            story.append(Spacer(1, 4))
            i += 1
            continue

        if stripped.startswith('# '):
            story.append(Spacer(1, 10))
            story.append(Paragraph(clean_text(stripped[2:]), title_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=10))
        elif stripped.startswith('## '):
            story.append(Paragraph(clean_text(stripped[3:]), subtitle_style))
        elif stripped.startswith('### '):
            story.append(Paragraph(clean_text(stripped[4:]), h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=6))
        elif stripped.startswith('#### '):
            story.append(Paragraph(clean_text(stripped[5:]), h2_style))
        elif stripped.startswith('##### '):
            story.append(Paragraph(clean_text(stripped[6:]), h3_style))
        elif stripped.startswith('- ') or stripped.startswith('* '):
            item_text = clean_text(stripped[2:])
            story.append(Paragraph(f"&bull; {item_text}", bullet_style))
        elif re.match(r'^\d+\.\s', stripped):
            item_text = clean_text(re.sub(r'^\d+\.\s', '', stripped))
            num = stripped.split('.')[0]
            story.append(Paragraph(f"<b>{num}.</b> {item_text}", bullet_style))
        elif stripped.startswith('> '):
            story.append(Paragraph(clean_text(stripped[2:]), callout_style))
        elif stripped.startswith('---'):
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceBefore=6, spaceAfter=6))
        else:
            story.append(Paragraph(clean_text(line), body_style))

        i += 1

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {output_pdf_path}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    docs_to_convert = [
        ('ml_topics_explanation.md', 'ml_topics_explanation.pdf'),
        ('project_files_explanation.md', 'project_files_explanation.pdf'),
        ('ml_models_explanation.md', 'ml_models_explanation.pdf')
    ]
    
    for md_name, pdf_name in docs_to_convert:
        md_path = os.path.join(base_dir, md_name)
        pdf_path = os.path.join(base_dir, pdf_name)
        if os.path.exists(md_path):
            markdown_to_pdf(md_path, pdf_path)
        else:
            print(f"File not found: {md_path}")
