"""
pdf_generator.py

Server-side PDF generation for SourceSkillsMiner profiles.
Rebuilds the full report directly from the raw JSON data using ReportLab,
completely bypassing html2canvas / html2pdf.js.

Usage (called from api.py):
    from pdf_generator import build_pdf
    pdf_bytes = build_pdf(username, profile_data, classification_data)
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, Image
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import datetime
import urllib.request

# ── Colour palette (mirrors CSS tokens) ───────────────────────────────────────
NAVY      = colors.HexColor('#1a2744')
BURGUNDY  = colors.HexColor('#7c1d2d')
TEAL      = colors.HexColor('#1a5c52')
AMBER_DK  = colors.HexColor('#7a4f00')
PAPER     = colors.HexColor('#faf8f3')
PAPER2    = colors.HexColor('#f3f0e8')
PAPER3    = colors.HexColor('#ede9df')
INK       = colors.HexColor('#1a1a2e')
INK2      = colors.HexColor('#2c2c4a')
INK3      = colors.HexColor('#4a4a6a')
MUTED     = colors.HexColor('#7a7a9a')
RULE      = colors.HexColor('#d6d3cc')
WHITE     = colors.white


# ── Styles ─────────────────────────────────────────────────────────────────────

def _styles():
    base = getSampleStyleSheet()

    def ps(name, **kw):
        defaults = dict(fontName='Helvetica', fontSize=10, leading=14,
                        textColor=INK, spaceAfter=0, spaceBefore=0)
        defaults.update(kw)
        return ParagraphStyle(name, **defaults)

    return {
        'journal_name': ps('jn', fontName='Helvetica-Bold', fontSize=8,
                           textColor=BURGUNDY, letterSpacing=2),
        'journal_title': ps('jt', fontName='Helvetica-Bold', fontSize=18,
                            textColor=NAVY, leading=22),
        'journal_sub': ps('js', fontSize=10, textColor=MUTED),

        'kicker': ps('kick', fontName='Helvetica-Bold', fontSize=7,
                     textColor=BURGUNDY, letterSpacing=2),
        'h1': ps('h1', fontName='Helvetica-Bold', fontSize=22,
                 textColor=NAVY, leading=26, spaceBefore=4),
        'h2': ps('h2', fontName='Helvetica-Bold', fontSize=11,
                 textColor=NAVY, letterSpacing=1.5, spaceBefore=14, spaceAfter=4),
        'body': ps('body', fontSize=9, textColor=INK, leading=13),
        'mono': ps('mono', fontName='Courier', fontSize=8,
                   textColor=INK3, leading=12),
        'mono_small': ps('mono_sm', fontName='Courier', fontSize=7,
                         textColor=MUTED, leading=10),
        'byline': ps('byl', fontSize=9, textColor=INK3),
        'score_big': ps('sb', fontName='Helvetica-Bold', fontSize=16,
                        textColor=NAVY),
        'label': ps('lbl', fontName='Helvetica-Bold', fontSize=7,
                    textColor=MUTED, letterSpacing=1.5),
        'cls_name': ps('cls', fontName='Helvetica-Bold', fontSize=18,
                       textColor=NAVY, leading=22),
        'col_head': ps('ch', fontName='Helvetica-Bold', fontSize=7,
                       textColor=INK3, letterSpacing=1.2),
        'tag': ps('tag', fontSize=7, textColor=MUTED),
        'colophon': ps('col', fontName='Courier', fontSize=7,
                       textColor=MUTED, leading=10),
    }


# ── Mini bar flowable ───────────────────────────────────────────────────────────

class HBar(Flowable):
    """A simple filled-rectangle progress bar."""
    def __init__(self, width, height, fraction, fg=NAVY, bg=PAPER3):
        super().__init__()
        self.width = width
        self.height = height
        self.fraction = max(0.0, min(1.0, fraction or 0.0))
        self.fg = fg
        self.bg = bg

    def draw(self):
        c = self.canv
        c.setFillColor(self.bg)
        c.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        if self.fraction > 0:
            c.setFillColor(self.fg)
            c.rect(0, 0, self.width * self.fraction, self.height, fill=1, stroke=0)

    def wrap(self, *_):
        return self.width, self.height


# ── Helpers ────────────────────────────────────────────────────────────────────

def _score_color(s):
    if s is None:
        return MUTED
    if s >= 0.5:
        return TEAL
    if s >= 0.0:
        return AMBER_DK
    return BURGUNDY


def _fmt_score(s):
    if s is None:
        return '—'
    return ('+' if s >= 0 else '') + f'{s:.3f}'


def _section_heading(text, styles):
    return [
        Spacer(1, 10),
        HRFlowable(width='100%', thickness=0.5, color=NAVY, spaceAfter=4),
        Paragraph(text, styles['h2']),
        Spacer(1, 4),
    ]


def _fetch_avatar(url: str, size: int = 80):
    """
    Download avatar from URL and return a ReportLab Image flowable,
    or None if the fetch fails for any reason.
    """
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'SourceSkillsMiner/1.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
        img_buf = io.BytesIO(data)
        return Image(img_buf, width=size, height=size)
    except Exception:
        return None


# ── Main builder ───────────────────────────────────────────────────────────────

def build_pdf(username: str, profile: dict, classification: dict) -> bytes:
    """
    Construct a full A4 PDF report from profile/classification dicts.
    Returns raw PDF bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title=f'SourceSkillsMiner — {username}',
        author='SourceSkillsMiner',
    )

    W = A4[0] - 4.4 * cm   # usable width
    S = _styles()
    story = []

    today = datetime.date.today().strftime('%d %B %Y').lstrip('0')

    # ── Masthead ───────────────────────────────────────────────────────────────
    mast_data = [[
        [Paragraph('RESEARCH INSTRUMENT', S['journal_name']),
         Paragraph('SourceSkillsMiner', S['journal_title']),
         Paragraph('GitHub Developer Profile Analysis System', S['journal_sub'])],
        [Paragraph(f'Vol. I · {today}<br/>ISSN 0000-0000', S['mono_small'])],
    ]]
    mast_tbl = Table(mast_data, colWidths=[W * 0.65, W * 0.35])
    mast_tbl.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'BOTTOM'),
        ('ALIGN',  (1, 0), (1, 0),  'RIGHT'),
    ]))
    story.append(mast_tbl)
    story.append(HRFlowable(width='100%', thickness=1.5, color=INK, spaceBefore=6, spaceAfter=16))

    # ── Article header ────────────────────────────────────────────────────────
    fs = profile.get('final_score', 0) or 0
    score_color = _score_color(fs)

    story.append(Paragraph('PROFILE REPORT', S['kicker']))
    story.append(Spacer(1, 6))

    # Fetch avatar (non-blocking fallback to None)
    avatar_url = profile.get('avatar_url')
    avatar_img = _fetch_avatar(avatar_url, size=56) if avatar_url else None

    # Build the name + meta block
    name_block = [
        Paragraph(username, S['h1']),
        Spacer(1, 4),
        Paragraph(f'Analysis date: {today}', S['byline']),
    ]

    score_block = [
        Paragraph('FINAL SCORE', S['label']),
        Paragraph(_fmt_score(fs), ParagraphStyle(
            'score_col', fontName='Helvetica-Bold', fontSize=14,
            textColor=score_color, leading=18)),
    ]

    if avatar_img:
        # Three-column: avatar | name+meta | score badge
        header_data = [[avatar_img, name_block, score_block]]
        col_widths   = [64, W * 0.58, W * 0.28]
    else:
        # Two-column fallback: name+meta | score badge
        header_data = [[name_block, score_block]]
        col_widths   = [W * 0.65, W * 0.35]

    header_tbl = Table(header_data, colWidths=col_widths)
    score_col_idx = 2 if avatar_img else 1
    header_tbl.setStyle(TableStyle([
        ('VALIGN',  (0, 0), (-1, -1), 'TOP'),
        ('ALIGN',   (score_col_idx, 0), (score_col_idx, 0), 'RIGHT'),
        ('BOX',     (score_col_idx, 0), (score_col_idx, 0), 0.5, RULE),
        ('PADDING', (score_col_idx, 0), (score_col_idx, 0), 6),
        ('LEFTPADDING',  (1, 0), (1, 0), 10),
    ]))
    story.append(header_tbl)
    story.append(HRFlowable(width='100%', thickness=0.5, color=RULE,
                             spaceBefore=10, spaceAfter=6))

    # ── §1 Classification ─────────────────────────────────────────────────────
    story += _section_heading('§1   ROLE CLASSIFICATION', S)

    if classification and not classification.get('error'):
        pred       = classification.get('prediction', '—')
        conf_pct   = classification.get('confidence_pct', 0) or 0
        all_probs  = (classification.get('all_probabilities') or [])[:9]

        # Left: big prediction + confidence
        left_items = [
            Paragraph('PREDICTED CATEGORY', S['label']),
            Spacer(1, 4),
            Paragraph(pred, S['cls_name']),
            Spacer(1, 6),
            Paragraph(f'◆  {conf_pct:.1f}% confidence', ParagraphStyle(
                'conf', fontName='Courier', fontSize=8,
                textColor=TEAL, leading=12)),
        ]

        # Right: probability bars
        right_items = [Paragraph('POSTERIOR PROBABILITY DISTRIBUTION', S['label']),
                       Spacer(1, 4)]
        bar_w = W * 0.45 - 1.2 * cm
        for i, p in enumerate(all_probs):
            is_top = (i == 0)
            bar_color = BURGUNDY if is_top else NAVY
            txt_color = BURGUNDY if is_top else INK3
            cat   = p.get('category', '—')
            prob  = (p.get('probability_pct') or 0) / 100
            pct_s = f"{p.get('probability_pct', 0):.1f}%"

            row_data = [[
                Paragraph(cat, ParagraphStyle('pn', fontName='Courier', fontSize=7,
                                              textColor=txt_color, leading=9)),
                HBar(bar_w * 0.55, 3, prob, fg=bar_color),
                Paragraph(pct_s, ParagraphStyle('pp', fontName='Courier', fontSize=7,
                                                textColor=bar_color if is_top else MUTED,
                                                leading=9, alignment=TA_RIGHT)),
            ]]
            rt = Table(row_data, colWidths=[bar_w * 0.38, bar_w * 0.42, bar_w * 0.20])
            rt.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                                    ('TOPPADDING',    (0, 0), (-1, -1), 2)]))
            right_items.append(rt)

        cls_data = [[left_items, right_items]]
        cls_tbl = Table(cls_data, colWidths=[W * 0.45, W * 0.55])
        cls_tbl.setStyle(TableStyle([
            ('VALIGN',  (0, 0), (-1, -1), 'TOP'),
            ('BOX',     (0, 0), (-1, -1), 0.5, RULE),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('LINEAFTER', (0, 0), (0, -1), 0.5, RULE),
            ('BACKGROUND', (0, 0), (-1, -1), WHITE),
        ]))
        story.append(cls_tbl)
    else:
        story.append(Paragraph('Classification unavailable.', S['tag']))

    # ── §2 Area Scores ────────────────────────────────────────────────────────
    story += _section_heading('§2   SCORING DIMENSIONS', S)

    AREA_META = {
        'OSS':          ('Open-Source Engagement',
                         ['issue_resolution_rate', 'pr_merge_rate', 'commits_activity']),
        'Status':       ('Development Status',
                         ['lines_frequency', 'commits_frequency', 'week_streak']),
        'Adaptability': ('Adaptability',
                         ['language_diversity_score', 'technology_adoption_score',
                          'domain_flexibility_score', 'resilience_score']),
        'Sentiment':    ('Communication Sentiment', []),
        'Commitment':   ('Long-term Commitment',    []),
    }

    # Table header
    col_w = [W * 0.22, W * 0.42, W * 0.22, W * 0.14]
    hdr   = [Paragraph(h, S['col_head']) for h in
             ['DIMENSION', 'SUB-INDICATORS', 'BAR', 'SCORE']]
    rows  = [hdr]
    areas = profile.get('areas', {}) or {}

    for key, (label, detail_keys) in AREA_META.items():
        area    = areas.get(key) or {}
        score   = area.get('score', 0) or 0
        details = area.get('details') or {}
        color   = _score_color(score)
        # bar: map [-1,1] → [0,1]
        fraction = (score + 1) / 2

        detail_parts = []
        for k in detail_keys:
            v = details.get(k)
            if v is not None:
                vstr = f'{v:.3f}' if isinstance(v, float) else str(v)
                detail_parts.append(f'{k.replace("_"," ")}: {vstr}')
        detail_str = '   '.join(detail_parts) if detail_parts else '—'

        rows.append([
            Paragraph(f'<b>{label}</b>', ParagraphStyle(
                'atn', fontName='Helvetica-Bold', fontSize=8,
                textColor=NAVY, leading=11)),
            Paragraph(detail_str, ParagraphStyle(
                'atd', fontName='Courier', fontSize=7,
                textColor=MUTED, leading=10)),
            HBar(col_w[2] - 4, 4, fraction, fg=color),
            Paragraph(_fmt_score(score), ParagraphStyle(
                'ats', fontName='Courier', fontSize=9, textColor=color,
                leading=12, alignment=TA_RIGHT)),
        ])

    areas_tbl = Table(rows, colWidths=col_w, repeatRows=1)
    areas_tbl.setStyle(TableStyle([
        ('FONTNAME',    (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, 0),  7),
        ('TEXTCOLOR',   (0, 0), (-1, 0),  INK3),
        ('LINEBELOW',   (0, 0), (-1, 0),  1.0, NAVY),
        ('LINEBELOW',   (0, 1), (-1, -1), 0.3, RULE),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',  (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, PAPER]),
    ]))
    story.append(areas_tbl)

    # ── §3 Technology Stack ───────────────────────────────────────────────────
    story += _section_heading('§3   TECHNOLOGY STACK', S)

    lu          = profile.get('language_usage') or {}
    total_lines = lu.get('total_lines') or 1
    lang_list   = lu.get('top_5_languages') or [
        {'language': l, 'lines': s[0] if isinstance(s, (list, tuple)) else s}
        for l, s in (lu.get('languages') or {}).items()
    ]
    lang_list = lang_list[:7]

    imp       = profile.get('import_scan') or {}
    pkgs      = []
    if not imp.get('error') and imp.get('languages'):
        for ld in imp['languages'].values():
            for pkg, cnt in (ld.get('packages') or {}).items():
                pkgs.append((pkg, cnt))
        pkgs.sort(key=lambda x: -x[1])
        pkgs = [p for p, _ in pkgs[:24]]

    bar_col_w = [W * 0.47 - 8, W * 0.47 - 8]

    # Language bars column
    lang_items = [Paragraph('LANGUAGE DISTRIBUTION', S['label']), Spacer(1, 6)]
    lang_bar_w = (W * 0.47 - 8) - 1.2 * cm
    for entry in lang_list:
        lang = entry.get('language', '?')
        lines = entry.get('lines', 0) or 0
        pct   = (lines / total_lines)
        pct_s = f'{pct * 100:.1f}%'
        row_data = [[
            Paragraph(lang, ParagraphStyle('ln', fontName='Courier', fontSize=8,
                                           textColor=INK, leading=10)),
            HBar(lang_bar_w * 0.55, 4, pct, fg=NAVY),
            Paragraph(pct_s, ParagraphStyle('lp', fontName='Courier', fontSize=7,
                                            textColor=MUTED, leading=10, alignment=TA_RIGHT)),
        ]]
        rt = Table(row_data, colWidths=[lang_bar_w * 0.35, lang_bar_w * 0.45, lang_bar_w * 0.20])
        rt.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                                ('TOPPADDING',    (0, 0), (-1, -1), 3)]))
        lang_items.append(rt)

    # Libraries column
    lib_items = [Paragraph('DETECTED LIBRARIES & PACKAGES', S['label']), Spacer(1, 6)]
    if pkgs:
        # Render as wrapped comma-separated mono text for compactness
        lib_items.append(Paragraph(
            '   '.join(pkgs),
            ParagraphStyle('libs', fontName='Courier', fontSize=7.5,
                           textColor=INK3, leading=12)
        ))
    else:
        lib_items.append(Paragraph('No import data available', S['tag']))

    tech_data = [[lang_items, lib_items]]
    tech_tbl  = Table(tech_data, colWidths=bar_col_w)
    tech_tbl.setStyle(TableStyle([
        ('VALIGN',  (0, 0), (-1, -1), 'TOP'),
        ('BOX',     (0, 0), (-1, -1), 0.5, RULE),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('LINEAFTER', (0, 0), (0, -1), 0.5, RULE),
        ('BACKGROUND', (0, 0), (-1, -1), WHITE),
    ]))
    story.append(tech_tbl)

    # ── §4 Commitment & Sentiment ─────────────────────────────────────────────
    story += _section_heading('§4   COMMITMENT & SENTIMENT', S)

    crit_labels = {
        'has_12_month_streak':                  '12-month contribution streak',
        'has_6_month_streak':                   '6-month contribution streak',
        'has_substantial_commits_per_repo':     'Median commits per repo at or above 50',
        'at_75th_percentile_followers':          'Followers at/above 75th percentile',
    }
    criteria  = (areas.get('Commitment') or {}).get('details', {}).get('criteria_met', {}) or {}
    crit_items = [Paragraph('COMMITMENT CRITERIA', S['label']), Spacer(1, 6)]
    for key, label in crit_labels.items():
        passed = criteria.get(key) is True
        mark   = '✓' if passed else '–'
        m_col  = TEAL if passed else MUTED
        t_col  = INK  if passed else MUTED
        row = [[
            Paragraph(mark, ParagraphStyle('cm', fontName='Helvetica-Bold',
                                           fontSize=9, textColor=m_col, leading=11)),
            Paragraph(label, ParagraphStyle('ct', fontSize=8, textColor=t_col, leading=11)),
        ]]
        rt = Table(row, colWidths=[14, (W * 0.47 - 8) - 14 - 10])
        rt.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                                ('LINEBELOW', (0, 0), (-1, -1), 0.3, RULE)]))
        crit_items.append(rt)

    # Sentiment
    sent_details  = (areas.get('Sentiment') or {}).get('details') or {}
    sent_entries  = sorted(sent_details.items(), key=lambda x: -abs(x[1]))[:12]
    sent_items    = [Paragraph('REPOSITORY SENTIMENT (VADER)', S['label']), Spacer(1, 6)]
    for repo, val in sent_entries:
        cls = 'pos' if val > 0.05 else ('neg' if val < -0.05 else 'neu')
        color_map = {'pos': TEAL, 'neg': BURGUNDY, 'neu': AMBER_DK}
        vstr  = ('+' if val >= 0 else '') + f'{val:.4f}'
        short = repo.split('/')[-1] if '/' in repo else repo
        row = [[
            Paragraph(short, ParagraphStyle('sr', fontName='Courier', fontSize=7.5,
                                            textColor=INK3, leading=10)),
            Paragraph(vstr, ParagraphStyle('sv', fontName='Courier', fontSize=7.5,
                                           textColor=color_map[cls], leading=10,
                                           alignment=TA_RIGHT)),
        ]]
        rt = Table(row, colWidths=[(W * 0.47 - 8) * 0.72, (W * 0.47 - 8) * 0.28])
        rt.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                                ('LINEBELOW', (0, 0), (-1, -1), 0.3, RULE)]))
        sent_items.append(rt)
    if not sent_entries:
        sent_items.append(Paragraph('No sentiment data available', S['tag']))

    commit_data = [[crit_items, sent_items]]
    commit_tbl  = Table(commit_data, colWidths=bar_col_w)
    commit_tbl.setStyle(TableStyle([
        ('VALIGN',  (0, 0), (-1, -1), 'TOP'),
        ('BOX',     (0, 0), (-1, -1), 0.5, RULE),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('LINEAFTER', (0, 0), (0, -1), 0.5, RULE),
        ('BACKGROUND', (0, 0), (-1, -1), WHITE),
    ]))
    story.append(commit_tbl)

    # ── Colophon ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width='100%', thickness=0.5, color=RULE, spaceAfter=8))
    col_data = [[
        Paragraph(
            'SourceSkillsMiner · Automated GitHub Profile Intelligence<br/>'
            'Scoring model: Multinomial Naïve Bayes · Features: language distribution, import graph',
            S['colophon']),
        Paragraph(f'Generated: {today}', ParagraphStyle(
            'cr', fontName='Courier', fontSize=7, textColor=MUTED,
            leading=10, alignment=TA_RIGHT)),
    ]]
    col_tbl = Table(col_data, colWidths=[W * 0.65, W * 0.35])
    col_tbl.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    story.append(col_tbl)

    # ── Build ──────────────────────────────────────────────────────────────────
    doc.build(story)
    return buf.getvalue()