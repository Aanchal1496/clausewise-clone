import traceback
import re
import fitz  # PyMuPDF

HEADING_PATTERNS = [
    r'^\d+\.\d*\s+[A-Z]',        # matches: 1. PAYMENT  or  1.1 Terms
    r'^\d+\)\s+[A-Z]',            # matches: 1) PAYMENT
    r'^(ARTICLE|SECTION|CLAUSE)\s+[\dIVX]+',  # matches: ARTICLE 1, SECTION IV
    r'^WHEREAS',                   # matches: WHEREAS the parties...
    r'^NOW,?\s*THEREFORE',         # matches: NOW THEREFORE
]

# Separate pattern for all-caps headings — stricter to avoid false splits
ALL_CAPS_HEADING = r'^[A-Z][A-Z\s]{7,50}$'

def is_heading(line):
    """Check if a line matches any common contract heading patterns."""
    for pattern in HEADING_PATTERNS:
        if re.match(pattern, line):
            return True
    return False

def is_likely_title(line):
    """Stricter check: all-caps line that looks like a section title, not random text."""
    if not re.match(ALL_CAPS_HEADING, line):
        return False
    # Avoid matching lines with punctuation or common words in all-caps context
    if ',' in line or '.' in line or 'AND' in line:
        return False
    return True

def extract_clauses(pdf_input):
    try:
        import fitz
        import re
    except Exception as e:
        print(f"[extractor] Import error: {e}")
        traceback.print_exc()
        return []

    try:
        # 1. Get raw text from input
        if isinstance(pdf_input, bytes):
            try:
                pdf = fitz.open(stream=pdf_input, filetype="pdf")
                text = ""
                for page in pdf:
                    text += page.get_text()
                pdf.close()
            except Exception as e:
                print(f"[extractor] Error opening PDF: {e}")
                traceback.print_exc()
                return []
        elif isinstance(pdf_input, str):
            text = pdf_input
        else:
            return []

    # 2. Pre-process text
    text = text.replace('\xa0', ' ')
    lines = text.split('\n')

    clauses = []
    current_clause = None

    # 3. Split text into clauses based on headings
    for line in lines:
        line = line.strip()
        if not line:
            continue

        is_new_heading = is_heading(line) or is_likely_title(line)

        if is_new_heading:
            # Skip duplicate consecutive headings (empty title-only clause)
            if current_clause is not None and current_clause['body']:
                clauses.append(current_clause)

            current_clause = {
                'id': len(clauses) + 1,
                'title': line,
                'body': [],
                'full_text': line
            }
        elif current_clause is not None:
            current_clause['body'].append(line)
            current_clause['full_text'] += ' ' + line
        else:
            # If no heading found yet, create an initial generic clause
            current_clause = {
                'id': 1,
                'title': "Introduction / General Terms",
                'body': [line],
                'full_text': line
            }

    try:
        if current_clause is not None and current_clause['body']:
            clauses.append(current_clause)
        elif current_clause is not None and not current_clause['body']:
            if clauses:
                clauses[-1]['full_text'] += ' ' + current_clause['full_text']
                clauses[-1]['title'] += ' - ' + current_clause['title']

        # 4. Merge very short clauses
        merged = []
        for c in clauses:
            word_count = len(c['full_text'].split())
            if word_count < 10 and merged:
                merged[-1]['full_text'] += ' ' + c['full_text']
                merged[-1]['title'] += ' / ' + c['title']
                merged[-1]['body'].extend(c['body'])
            else:
                merged.append(c)
        clauses = merged

        # 5. If still no clauses found, return the whole text as one clause
        if not clauses and text.strip():
            clauses.append({
                'id': 1,
                'title': "Full Document",
                'body': [text.strip()],
                'full_text': text.strip()
            })

        # Re-number IDs sequentially
        for i, c in enumerate(clauses):
            c['id'] = i + 1
    except Exception as e:
        print(f"[extractor] Post-processing error: {e}")
        traceback.print_exc()
        return []

    return clauses