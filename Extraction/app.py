import os
import sys
from dotenv import load_dotenv
load_dotenv()

# ── Startup validation ───────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    print("=" * 60)
    print("  WARNING: GROQ_API_KEY is not set!")
    print("  AI explanations and chat will use fallback text.")
    print("  Set it in Railway dashboard → Variables → GROQ_API_KEY")
    print("=" * 60)

import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
from extractor import extract_clauses
from classifier import classify_all
from translator import translate_all
from scorer import score_contract
from rag_store import store_all_clauses, get_store_stats
from rag_qa import answer_question
import uuid

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'endpoints': {
            'POST /extract':      'Step 1 only',
            'POST /analyze':      'Steps 1+2',
            'POST /full-analyze': 'Steps 1+2+3',
            'POST /report':       'Steps 1+2+3+4 via text',
            'POST /analyze-pdf':  'Steps 1+2+3+4 via PDF upload',
        }
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/extract', methods=['POST'])
def extract():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Please send JSON with a "text" field'}), 400
    clauses = extract_clauses(data['text'].strip())
    return jsonify({'total_clauses': len(clauses), 'clauses': clauses})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Please send JSON with a "text" field'}), 400
    clauses    = extract_clauses(data['text'].strip())
    classified = classify_all(clauses)
    return jsonify({
        'total_clauses': len(classified),
        'risky_count':   len([c for c in classified if c['is_risky']]),
        'clauses':       classified
    })

@app.route('/full-analyze', methods=['POST'])
def full_analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Please send JSON with a "text" field'}), 400
    clauses    = extract_clauses(data['text'].strip())
    classified = classify_all(clauses)
    translated = translate_all(classified)
    return jsonify({
        'total_clauses': len(translated),
        'risky_count':   len([c for c in translated if c['is_risky']]),
        'clauses':       translated
    })

@app.route('/report', methods=['POST'])
def report():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Please send JSON with a "text" field'}), 400
    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Text field is empty'}), 400
    
    clauses    = extract_clauses(text)
    classified = classify_all(clauses)
    translated = translate_all(classified)
    final      = score_contract(translated)
    
    # Store clauses for RAG
    contract_id = str(uuid.uuid4())
    store_all_clauses(translated, contract_id=contract_id)
    final['contract_id'] = contract_id
    
    return jsonify(final)

@app.route('/analyze-pdf', methods=['POST'])
def analyze_pdf():
    try:
        # ── Stage 0: Validate upload ──────────────────────────────────────────
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded. Send a PDF as form-data with key "file"'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400

        pdf_bytes = file.read()
        print("[analyze-pdf] Stage 0: Upload validated")
    except Exception as e:
        print(f"[analyze-pdf] Stage 0 error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Upload validation failed', 'detail': str(e)}), 500

    # ── Stage 1: Extract clauses from PDF ────────────────────────────────────
    try:
        print("[analyze-pdf] Stage 1: Extracting clauses...")
        clauses = extract_clauses(pdf_bytes)
        print(f"[analyze-pdf] Stage 1: Extracted {len(clauses)} clauses")
        if not clauses:
            return jsonify({'error': 'Could not extract text from PDF. It may be a scanned image or protected.'}), 400
    except Exception as e:
        print(f"[analyze-pdf] Stage 1 error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'PDF text extraction failed', 'detail': str(e), 'stage': 'extract'}), 500

    # ── Stage 2: Classify clauses ────────────────────────────────────────────
    try:
        print("[analyze-pdf] Stage 2: Classifying clauses...")
        classified = classify_all(clauses)
        print(f"[analyze-pdf] Stage 2: Classified {len(classified)} clauses")
    except Exception as e:
        print(f"[analyze-pdf] Stage 2 error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Clause classification failed', 'detail': str(e), 'stage': 'classify'}), 500

    # ── Stage 3: Translate / explain clauses ─────────────────────────────────
    try:
        print("[analyze-pdf] Stage 3: Generating explanations...")
        translated = translate_all(classified)
        print(f"[analyze-pdf] Stage 3: Translated {len(translated)} clauses")
    except Exception as e:
        print(f"[analyze-pdf] Stage 3 error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'AI explanation generation failed', 'detail': str(e), 'stage': 'translate'}), 500

    # ── Stage 4: Score contract ──────────────────────────────────────────────
    try:
        print("[analyze-pdf] Stage 4: Scoring contract...")
        final = score_contract(translated)
        print(f"[analyze-pdf] Stage 4: Score = {final.get('score')}")
    except Exception as e:
        print(f"[analyze-pdf] Stage 4 error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Contract scoring failed', 'detail': str(e), 'stage': 'score'}), 500

    # ── Stage 5: Store in RAG database ───────────────────────────────────────
    try:
        print("[analyze-pdf] Stage 5: Storing clauses for Q&A...")
        contract_id = str(uuid.uuid4())
        store_all_clauses(translated, contract_id=contract_id)
        final['contract_id'] = contract_id
        print(f"[analyze-pdf] Stage 5: Stored with contract_id={contract_id}")
    except Exception as e:
        print(f"[analyze-pdf] Stage 5 error (non-fatal): {e}")
        traceback.print_exc()
        final['contract_id'] = None
        final['rag_warning'] = 'Clause storage failed; chat Q&A may be limited'

    # ── Stage 6: Attach extracted text ───────────────────────────────────────
    try:
        final['extracted_text'] = " ".join([c['full_text'] for c in clauses])
    except Exception:
        pass

    print("[analyze-pdf] Complete — returning result")
    return jsonify(final)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Send JSON with "question" and "clauses"'}), 400
    try:
        answer = answer_question(data['question'], data.get('clauses', []))
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': 'Chat processing failed', 'detail': str(e)}), 500

@app.route('/rag-stats')
def rag_stats():
    return jsonify(get_store_stats())

if __name__ == '__main__':
    print("\n* Contract Analyzer - full pipeline ready")
    print("* POST /analyze-pdf - upload a PDF contract")
    print("* POST /report      - send raw text")
    print("* POST /chat        - ask questions about the contract\n")
    app.run(debug=True, port=5000, host='0.0.0.0')