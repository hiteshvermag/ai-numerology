"""
AI Numerology: Gradio UI + CrewAI agents
=========================================
Features
  * Numerology chart, year-wise predictions, name check + suggestions,
    relationship / life guidance
  * Partner compatibility mode (two people)
  * Output language: English, Hinglish (Roman Hindi) or Hindi (Devanagari)
  * PDF export of every report

Python computes ALL numbers (deterministic, accurate).
CrewAI agents only INTERPRET those numbers.

Run:
    pip install -r requirements.txt
    export OPENAI_API_KEY=...              # or ANTHROPIC_API_KEY=...
    export CREWAI_MODEL=gpt-4o-mini        # or anthropic/claude-sonnet-4-5
    python numerology_app.py

Hindi PDF needs a Devanagari font (see find_devanagari_font()).
On Windows it works out of the box (Nirmala UI). On Linux/Mac download
"NotoSansDevanagari-Regular.ttf" and put it in ./fonts/ or set
NUMEROLOGY_FONT=/path/to/font.ttf
"""

import os
import re
import string
import tempfile
import datetime as dt

import gradio as gr

# ======================================================================
# 1. NUMEROLOGY ENGINE (Pythagorean system)
# ======================================================================
LETTER_VALUES = {ch: (i % 9) + 1 for i, ch in enumerate(string.ascii_uppercase)}
VOWELS = set("AEIOU")
MASTER_NUMBERS = {11, 22, 33}

NUM_KEYWORD = {
    1: "Leader", 2: "Diplomat", 3: "Creator", 4: "Builder", 5: "Adventurer",
    6: "Nurturer", 7: "Seeker", 8: "Achiever", 9: "Humanitarian",
    11: "Intuitive (master)", 22: "Master Builder", 33: "Master Teacher",
}


def reduce_number(n: int, keep_master: bool = True) -> int:
    while n > 9:
        if keep_master and n in MASTER_NUMBERS:
            return n
        n = sum(int(d) for d in str(n))
    return n


def clean(name: str) -> str:
    return "".join(c for c in name.upper() if c in LETTER_VALUES)


def name_total(name: str, only=None) -> int:
    total = 0
    for c in clean(name):
        if only == "vowels" and c not in VOWELS:
            continue
        if only == "consonants" and c in VOWELS:
            continue
        total += LETTER_VALUES[c]
    return total


def expression_number(name: str) -> int:
    return reduce_number(name_total(name))


def soul_urge_number(name: str) -> int:
    return reduce_number(name_total(name, "vowels"))


def personality_number(name: str) -> int:
    return reduce_number(name_total(name, "consonants"))


def life_path_number(d: dt.date) -> int:
    m = reduce_number(d.month)
    day = reduce_number(d.day)
    y = reduce_number(sum(int(x) for x in str(d.year)))
    return reduce_number(m + day + y)


def personal_year(d: dt.date, year: int) -> int:
    return reduce_number(
        reduce_number(d.day)
        + reduce_number(d.month)
        + reduce_number(sum(int(x) for x in str(year)))
    )


def build_chart(first: str, last: str, dob: dt.date) -> dict:
    full = f"{first} {last}"
    lp = life_path_number(dob)
    exp = expression_number(full)
    return {
        "Full Name": full,
        "Date of Birth": dob.isoformat(),
        "Life Path": lp,
        "Expression (Destiny)": exp,
        "Soul Urge (Heart's Desire)": soul_urge_number(full),
        "Personality": personality_number(full),
        "Birthday Number": reduce_number(dob.day),
        "Attitude Number": reduce_number(reduce_number(dob.day) + reduce_number(dob.month)),
        "Maturity Number": reduce_number(lp + exp),
        "First Name Number": expression_number(first),
        "Last Name Number": expression_number(last),
        "Current Personal Year": personal_year(dob, dt.date.today().year),
    }


def chart_rows(chart: dict) -> list:
    return [[k, v, NUM_KEYWORD.get(v, "") if isinstance(v, int) else ""]
            for k, v in chart.items()]


def yearly_numbers(dob: dt.date, start_year: int, count: int) -> dict:
    return {y: personal_year(dob, y) for y in range(start_year, start_year + count)}


def name_candidates(first: str, last: str, life_path: int, limit: int = 12):
    """Spelling variants whose Expression number matches the Life Path
    (or its next harmonic). Verified by code so the AI never does maths."""
    first_c, last_c = first.strip().title(), last.strip().title()
    first_variants = {first_c, first_c + first_c[-1].lower()}
    if len(first_c) > 3:
        first_variants.add(first_c[:-1])
    for suffix in ("a", "h", "i", "y", "e"):
        first_variants.add(first_c + suffix)
    last_variants = {last_c, last_c + last_c[-1].lower(), last_c + "h", last_c + "a"}

    pool = set()
    for fv in first_variants:
        for lv in last_variants:
            pool.add(f"{fv} {lv}")
            for ch in string.ascii_uppercase:
                pool.add(f"{fv} {ch}. {lv}")

    original = f"{first_c} {last_c}"
    targets = {life_path, reduce_number(life_path + 1)}
    out = [(v, expression_number(v)) for v in pool
           if v != original and expression_number(v) in targets]
    out.sort(key=lambda t: (abs(len(t[0]) - len(original)), t[0]))
    return out[:limit]


# ----- compatibility (deterministic) ----------------------------------
_BASE = {11: 2, 22: 4, 33: 6}
_GROUP = {1: "A", 5: "A", 7: "A", 2: "B", 4: "B", 8: "B", 3: "C", 6: "C", 9: "C"}
_CROSS = {frozenset("AB"): 55, frozenset("BC"): 65, frozenset("AC"): 70}


def pair_score(a: int, b: int) -> int:
    a, b = _BASE.get(a, a), _BASE.get(b, b)
    if a == b:
        return 80
    ga, gb = _GROUP[a], _GROUP[b]
    return 90 if ga == gb else _CROSS[frozenset((ga, gb))]


def build_compat(c1: dict, c2: dict):
    weights = [("Life Path", 0.4), ("Soul Urge (Heart's Desire)", 0.3),
               ("Expression (Destiny)", 0.2), ("Birthday Number", 0.1)]
    rows, total = [], 0.0
    for key, w in weights:
        s = pair_score(c1[key], c2[key])
        total += s * w
        rows.append([key, c1[key], c2[key], f"{s}%"])
    couple = reduce_number(c1["Life Path"] + c2["Life Path"])
    return rows, round(total), couple


# ======================================================================
# 2. LANGUAGE SETTINGS
# ======================================================================
LANG_RULE = {
    "English": "Write in clear, simple English.",
    "Hinglish": "Write in Hinglish: Hindi written in Roman (Latin) script, the casual way "
                "Indians type on WhatsApp. Keep headings simple.",
    "Hindi": "Write ENTIRELY in Hindi using Devanagari script. Keep numerals as digits and "
             "keep personal names and social-media handles in Latin script.",
}

LABELS = {
    "English": {
        "report": "Numerology Report", "chart": "Numerology Chart",
        "years": "Year-wise Predictions", "name": "Name Check & Suggestions",
        "rel": "Relationships & Life", "compat": "Compatibility Report",
        "score": "Compatibility score", "couple": "Couple number",
        "disc": "For entertainment and self-reflection only.",
    },
    "Hinglish": {
        "report": "Numerology Report", "chart": "Numerology Chart",
        "years": "Saal-dar Predictions", "name": "Naam Check aur Suggestions",
        "rel": "Rishte aur Zindagi", "compat": "Compatibility Report",
        "score": "Compatibility score", "couple": "Couple number",
        "disc": "Sirf manoranjan aur self-reflection ke liye.",
    },
    "Hindi": {
        "report": "अंक ज्योतिष रिपोर्ट", "chart": "अंक ज्योतिष चार्ट",
        "years": "वर्षवार भविष्यफल", "name": "नाम जाँच और सुझाव",
        "rel": "रिश्ते और जीवन", "compat": "अनुकूलता रिपोर्ट",
        "score": "अनुकूलता स्कोर", "couple": "जोड़ी अंक",
        "disc": "केवल मनोरंजन और आत्म-चिंतन के लिए।",
    },
}

# ======================================================================
# 3. CREWAI
# ======================================================================
def _llm():
    from crewai import LLM  # lazy import
    return LLM(model=os.getenv("CREWAI_MODEL", "gpt-4o-mini"), temperature=0.7)


def _agent(**kw):
    from crewai import Agent
    return Agent(verbose=False, allow_delegation=False, **kw)


def run_report_crew(chart: dict, years: dict, candidates: list, language: str) -> list:
    from crewai import Task, Crew, Process
    llm = _llm()

    chart_txt = "\n".join(f"- {k}: {v}" for k, v in chart.items())
    years_txt = "\n".join(f"- {y}: Personal Year {n}" for y, n in years.items())
    cand_txt = "\n".join(f"- {n} (Expression = {e})" for n, e in candidates) or "None found"

    analyst = _agent(
        role="Senior Numerologist",
        goal="Explain a person's numerology chart clearly and warmly.",
        backstory="Experienced Pythagorean numerologist who explains each number in plain "
                  "language: strengths, challenges, life purpose.",
        llm=llm)
    forecaster = _agent(
        role="Numerology Forecaster",
        goal="Give year-wise predictions based on Personal Year numbers.",
        backstory="Specialist in 9-year numerology cycles (career, money, love, health, "
                  "growth) who gives practical, balanced guidance.",
        llm=llm)
    advisor = _agent(
        role="Name Numerology Advisor",
        goal="Judge whether a name aligns with the Life Path and suggest better spellings.",
        backstory="You suggest small, realistic spelling changes and social media handles. "
                  "You NEVER calculate numbers; you only use numbers provided to you.",
        llm=llm)
    coach = _agent(
        role="Relationship & Life Coach (Numerology)",
        goal="Describe love, relationships, career, money and health tendencies.",
        backstory="You translate numerology into compassionate, practical advice.",
        llm=llm)

    common = (
        "Use ONLY the numbers given below. Do not recalculate. Write in Markdown. "
        "Be positive but honest. This is for entertainment/self-reflection.\n"
        f"LANGUAGE: {LANG_RULE[language]}\n\nCHART:\n{chart_txt}\n"
    )

    t1 = Task(
        description=common + "\nExplain every number in the chart: meaning, strengths, "
                    "challenges. Start with a short table (Number | Value | Keyword), then a "
                    "short paragraph per number, then a 3-line overall personality summary.",
        expected_output="Markdown numerology chart reading.", agent=analyst)
    t2 = Task(
        description=common + f"\nPERSONAL YEARS:\n{years_txt}\n\nFor each year give: theme, "
                    "career/money, love/relationships, health, one 'do' and one 'avoid'. "
                    "Mention the best and the most challenging year.",
        expected_output="Markdown year-wise prediction.", agent=forecaster)
    t3 = Task(
        description=common + f"\nCANDIDATE SPELLINGS (pre-verified by code):\n{cand_txt}\n\n"
                    "1) Say whether the current name aligns with the Life Path and Expression "
                    "numbers (Correct / Partly / Not aligned) and why.\n"
                    "2) If not fully aligned, recommend the best 3 names ONLY from the candidate "
                    "list and explain what each improves.\n"
                    "3) Give 8 example social-media usernames/handles (Latin script) using the "
                    "best names, mentioning which platform suits which "
                    "(Instagram / X / LinkedIn / YouTube).\n"
                    "4) Add a short note that changing a name is optional.",
        expected_output="Markdown name analysis with suggestions and handles.", agent=advisor)
    t4 = Task(
        description=common + "\nCover: love style, ideal partner Life Path numbers "
                    "(best / ok / challenging), friendship, family, career fields, money habits, "
                    "health tendencies, lucky colours/days/gemstone (traditional associations), "
                    "and 5 practical tips.",
        expected_output="Markdown relationship and life guidance.", agent=coach)

    crew = Crew(agents=[analyst, forecaster, advisor, coach], tasks=[t1, t2, t3, t4],
                process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return [t.raw for t in result.tasks_output]


def run_compat_crew(c1: dict, c2: dict, rows: list, score: int, couple: int,
                    language: str) -> str:
    from crewai import Task, Crew, Process
    llm = _llm()

    def block(c):
        return "\n".join(f"  - {k}: {v}" for k, v in c.items())

    table = "\n".join(f"- {r[0]}: {r[1]} vs {r[2]} -> {r[3]}" for r in rows)
    analyst = _agent(
        role="Numerology Compatibility Analyst",
        goal="Explain how two people's numerology charts interact in a relationship.",
        backstory="You give kind, realistic compatibility readings and practical advice. "
                  "You never recalculate numbers; scores are supplied to you.",
        llm=llm)
    task = Task(
        description=(
            "Use ONLY the numbers below. Write in Markdown.\n"
            f"LANGUAGE: {LANG_RULE[language]}\n\n"
            f"PERSON 1:\n{block(c1)}\n\nPERSON 2:\n{block(c2)}\n\n"
            f"PAIR SCORES:\n{table}\n\nOVERALL COMPATIBILITY: {score}%\n"
            f"COUPLE NUMBER (sum of Life Paths): {couple}\n\n"
            "Write: 1) overall verdict, 2) emotional bond, 3) communication style, "
            "4) romance & intimacy, 5) money & career together, 6) family & long-term, "
            "7) likely friction points and how to handle each, 8) 5 practical tips. "
            "Be encouraging, never fatalistic: a low score is not a doom prediction."),
        expected_output="Markdown compatibility reading.", agent=analyst)
    result = Crew(agents=[analyst], tasks=[task], process=Process.sequential,
                  verbose=False).kickoff()
    return result.tasks_output[0].raw


# ======================================================================
# 4. PDF EXPORT (fpdf2)
# ======================================================================
def find_devanagari_font():
    candidates = [
        os.getenv("NUMEROLOGY_FONT", ""),
        "./fonts/NotoSansDevanagari-Regular.ttf",
        "C:/Windows/Fonts/Nirmala.ttf",
        "C:/Windows/Fonts/mangal.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    ]
    return next((p for p in candidates if p and os.path.exists(p)), None)


def find_latin_font():
    """Fallback font for Latin letters (names, handles) inside Hindi PDFs."""
    candidates = [
        os.getenv("NUMEROLOGY_LATIN_FONT", ""),
        "./fonts/NotoSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    return next((p for p in candidates if p and os.path.exists(p)), None)


_EMOJI = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F\u200d]")


def _latin(text: str) -> str:
    for a, b in {"–": "-", "—": "-", "’": "'", "‘": "'", "“": '"', "”": '"',
                 "•": "-", "…": "...", "→": "->"}.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "ignore").decode("latin-1")


def _inline_clean(s: str) -> str:
    s = re.sub(r"\*\*|__|`", "", s)
    return re.sub(r"(?<!\w)\*(?!\s)|(?<=\S)\*(?!\w)", "", s).strip()


def _md_lines(md: str):
    for raw in md.splitlines():
        line = raw.rstrip()
        s = line.strip()
        if not s:
            yield "gap", ""
        elif re.fullmatch(r"[|\s:\-]+", s) and "-" in s:
            continue                                   # table separator row
        elif s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            yield "p", _inline_clean("  |  ".join(cells))
        elif s.startswith("#"):
            level = min(len(s) - len(s.lstrip("#")), 3)
            yield f"h{level}", _inline_clean(s.lstrip("#"))
        elif re.match(r"^[-*+]\s+", s):
            yield "li", "- " + _inline_clean(re.sub(r"^[-*+]\s+", "", s))
        else:
            yield "p", _inline_clean(s)


def make_pdf(title: str, sections: list, language: str, footer: str):
    """sections = [(heading, markdown_text), ...]. Returns file path."""
    from fpdf import FPDF

    font_path = find_devanagari_font() if language == "Hindi" else None
    if language == "Hindi" and not font_path:
        raise RuntimeError(
            "Hindi PDF needs a Devanagari font. Download NotoSansDevanagari-Regular.ttf "
            "into ./fonts/ or set NUMEROLOGY_FONT=/path/to/font.ttf")

    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    if font_path:
        pdf.add_font("U", "", font_path)
        pdf.set_text_shaping(True)       # needs uharfbuzz for Devanagari
        latin_path = find_latin_font()
        if latin_path:                   # Devanagari fonts often lack A-Z
            pdf.add_font("L", "", latin_path)
            pdf.set_fallback_fonts(["L"])
        family, bold = "U", ""
        fix = lambda t: _EMOJI.sub("", t)
    else:
        family, bold = "Helvetica", "B"
        fix = lambda t: _latin(_EMOJI.sub("", t))

    def write(text, size, style="", h=None, gap_after=0):
        pdf.set_font(family, style, size)
        pdf.multi_cell(0, h or size * 0.55, fix(text), new_x="LMARGIN", new_y="NEXT")
        if gap_after:
            pdf.ln(gap_after)

    write(title, 20, bold, h=10, gap_after=4)
    for heading, md in sections:
        if not md:
            continue
        pdf.ln(3)
        write(heading, 15, bold, h=8, gap_after=2)
        for kind, text in _md_lines(md):
            if kind == "gap":
                pdf.ln(2)
            elif kind in ("h1", "h2"):
                write(text, 13, bold, h=7, gap_after=1)
            elif kind == "h3":
                write(text, 11.5, bold, h=6.5)
            else:
                write(text, 10.5, h=5.6)
    pdf.ln(6)
    write(footer, 9, "", h=5)

    out = os.path.join(tempfile.mkdtemp(), "report.pdf")
    pdf.output(out)
    return out


def pdf_name(prefix: str, *names: str) -> str:
    safe = re.sub(r"\W+", "_", "_".join(names)).strip("_") or "report"
    return f"{prefix}_{safe}.pdf"


def _safe_pdf(*args, **kw):
    """Never let a PDF problem break the report. Returns (path|None)."""
    try:
        path = make_pdf(*args, **kw)
        return path
    except Exception as e:  # noqa: BLE001
        gr.Warning(f"PDF could not be created: {e}")
        return None


# ======================================================================
# 5. GRADIO HANDLERS
# ======================================================================
def parse_inputs(first, last, dob_str):
    if not (first or "").strip() or not (last or "").strip():
        raise ValueError("Please enter both first name and last name.")
    try:
        dob = dt.datetime.strptime((dob_str or "").strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Date of birth must be in YYYY-MM-DD format (e.g. 1995-08-24).")
    if dob > dt.date.today():
        raise ValueError("Date of birth cannot be in the future.")
    return first.strip(), last.strip(), dob


def generate(first, last, dob_str, start_year, n_years, language):
    empty = ([], "", "", "", "", None)
    try:
        first, last, dob = parse_inputs(first, last, dob_str)
    except ValueError as e:
        return ([], f"⚠️ {e}", "", "", "", None)

    chart = build_chart(first, last, dob)
    years = yearly_numbers(dob, int(start_year or dt.date.today().year), int(n_years))
    cands = name_candidates(first, last, chart["Life Path"])
    table = chart_rows(chart)

    try:
        chart_md, years_md, name_md, rel_md = run_report_crew(chart, years, cands, language)
    except Exception as e:  # noqa: BLE001
        return (table, f"❌ CrewAI error: {e}\n\nCheck your API key and CREWAI_MODEL.",
                "", "", "", None)

    L = LABELS[language]
    summary = "\n".join(f"- {k}: {v}" + (f"  ({NUM_KEYWORD[v]})" if v in NUM_KEYWORD else "")
                        if isinstance(v, int) else f"- {k}: {v}"
                        for k, v in chart.items())
    pdf_path = _safe_pdf(
        f"{L['report']}: {first} {last}",
        [(L["chart"], summary + "\n\n" + chart_md), (L["years"], years_md),
         (L["name"], name_md), (L["rel"], rel_md)],
        language, L["disc"])
    if pdf_path:
        final = os.path.join(os.path.dirname(pdf_path), pdf_name("numerology", first, last))
        os.replace(pdf_path, final)
        pdf_path = final
    return table, chart_md, years_md, name_md, rel_md, pdf_path


def generate_compat(first1, last1, dob1, first2, last2, dob2, language):
    try:
        f1, l1, d1 = parse_inputs(first1, last1, dob1)
        f2, l2, d2 = parse_inputs(first2, last2, dob2)
    except ValueError as e:
        return f"⚠️ {e}", [], "", None

    c1, c2 = build_chart(f1, l1, d1), build_chart(f2, l2, d2)
    rows, score, couple = build_compat(c1, c2)
    L = LABELS[language]
    headline = f"## 💞 {L['score']}: **{score}%**   |   {L['couple']}: **{couple}**"

    try:
        md = run_compat_crew(c1, c2, rows, score, couple, language)
    except Exception as e:  # noqa: BLE001
        return headline, rows, f"❌ CrewAI error: {e}", None

    table_txt = "\n".join(f"- {r[0]}: {r[1]} / {r[2]} = {r[3]}" for r in rows)
    pdf_path = _safe_pdf(
        f"{L['compat']}: {f1} & {f2}",
        [(f"{L['score']}: {score}%  |  {L['couple']}: {couple}", table_txt),
         (L["compat"], md)],
        language, L["disc"])
    if pdf_path:
        final = os.path.join(os.path.dirname(pdf_path), pdf_name("compatibility", f1, f2))
        os.replace(pdf_path, final)
        pdf_path = final
    return headline, rows, md, pdf_path


# ======================================================================
# 6. UI
# ======================================================================
LANG_CHOICES = [("English", "English"),
                ("Hinglish (Roman Hindi)", "Hinglish"),
                ("हिन्दी (Hindi)", "Hindi")]

with gr.Blocks(title="AI Numerology") as demo:
    gr.Markdown("# 🔢 AI Numerology\nChart · yearly predictions · name check · "
                "relationships · partner compatibility · PDF export")

    with gr.Row():
        first = gr.Textbox(label="First Name", placeholder="Rahul")
        last = gr.Textbox(label="Last Name", placeholder="Sharma")
        dob = gr.Textbox(label="Date of Birth (YYYY-MM-DD)", placeholder="1995-08-24")
    with gr.Row():
        language = gr.Dropdown(LANG_CHOICES, value="English", label="Report language")
        start_year = gr.Number(label="Predictions start year",
                               value=dt.date.today().year, precision=0)
        n_years = gr.Slider(1, 10, value=5, step=1, label="Years to predict")

    with gr.Tabs():
        with gr.Tab("🧿 My Report"):
            btn = gr.Button("✨ Generate Numerology Report", variant="primary")
            with gr.Tabs():
                with gr.Tab("📊 Chart"):
                    chart_table = gr.Dataframe(headers=["Number", "Value", "Keyword"],
                                               interactive=False)
                    chart_md = gr.Markdown()
                with gr.Tab("📅 Year-wise"):
                    years_md = gr.Markdown()
                with gr.Tab("🔤 Name Check"):
                    name_md = gr.Markdown()
                with gr.Tab("❤️ Relationships & Life"):
                    rel_md = gr.Markdown()
            pdf_out = gr.File(label="📄 Download PDF report")

        with gr.Tab("💑 Compatibility"):
            gr.Markdown("**Person 1** is the person entered at the top. "
                        "Enter the partner's details below.")
            with gr.Row():
                p_first = gr.Textbox(label="Partner First Name")
                p_last = gr.Textbox(label="Partner Last Name")
                p_dob = gr.Textbox(label="Partner DOB (YYYY-MM-DD)")
            cbtn = gr.Button("💞 Check Compatibility", variant="primary")
            c_head = gr.Markdown()
            c_table = gr.Dataframe(headers=["Factor", "Person 1", "Person 2", "Match"],
                                   interactive=False)
            c_md = gr.Markdown()
            c_pdf = gr.File(label="📄 Download compatibility PDF")

    btn.click(generate, [first, last, dob, start_year, n_years, language],
              [chart_table, chart_md, years_md, name_md, rel_md, pdf_out])
    cbtn.click(generate_compat, [first, last, dob, p_first, p_last, p_dob, language],
               [c_head, c_table, c_md, c_pdf])

    gr.Markdown("*For entertainment and self-reflection only, not professional advice.*")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")) , share=False)
