import json
import traceback
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Load trained model once at startup ───────────────────────────────────────
# Get absolute path to the model directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load the trained model from Hugging Face
MODEL_DIR = "Aanchal14/clausewise-bert"
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")

device = torch.device("cpu")

print("* Downloading/loading trained BERT model from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=False
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    trust_remote_code=False
).to(device)
model.eval()  # inference mode, not training mode

with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)
    # convert string keys back to integers
    label_map = {int(k): v for k, v in label_map.items()}

print(f"* BERT classifier ready - {len(label_map)} clause types")

# ── Risk levels ──────────────────────────────────────────────
RISK_LEVELS = {
    'auto_renewal':       {'level': 'high',   'color': 'red'},
    'liability_waiver':   {'level': 'high',   'color': 'red'},
    'arbitration':        {'level': 'high',   'color': 'red'},
    'data_selling':       {'level': 'high',   'color': 'red'},
    'unilateral_changes': {'level': 'high',   'color': 'red'},
    'exit_penalty':       {'level': 'medium', 'color': 'amber'},
    'price_escalation':   {'level': 'medium', 'color': 'amber'},
    'jurisdiction':       {'level': 'medium', 'color': 'amber'},
    'ip_ownership':       {'level': 'medium', 'color': 'amber'},
    'notice_period':      {'level': 'low',    'color': 'green'},
    'general':            {'level': 'low',    'color': 'green'},
}

PLAIN_ENGLISH = {
    'auto_renewal':       'Contract renews automatically unless cancelled by the user within the specified notice period.',
    'liability_waiver':   'Company disclaims responsibility for damages or losses incurred by the user.',
    'arbitration':        'User waives the right to pursue claims in court and agrees to binding arbitration.',
    'data_selling':       'User data may be shared with or sold to third parties.',
    'unilateral_changes': 'Company reserves the right to modify terms without prior notice to the user.',
    'exit_penalty':       'Early termination or cancellation may incur financial penalties.',
    'price_escalation':   'Pricing is subject to increase at specified intervals or under certain conditions.',
    'jurisdiction':       'Legal disputes must be resolved in a specified jurisdiction, which may differ from the user\'s location.',
    'ip_ownership':       'Company retains ownership of intellectual property created using its platform.',
    'notice_period':      'A minimum notice period is required to terminate or cancel the agreement.',
    'general':            'Standard clause. Review in context of the full agreement.',
}

# ── Predict clause type using BERT ────────────────────────────────────────────
def detect_type(text):
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits      = outputs.logits
        predicted_id = torch.argmax(logits, dim=1).item()
        clause_type  = label_map.get(predicted_id, "general")

        probabilities = torch.softmax(logits, dim=1)
        confidence    = probabilities[0][predicted_id].item()

        return clause_type, round(confidence * 100, 1)
    except Exception as e:
        print(f"[classifier] detect_type error: {e}")
        traceback.print_exc()
        return "general", 0.0

def classify_all(clauses):
    results = []
    for clause in clauses:
        try:
            results.append(classify_clause(clause))
        except Exception as e:
            print(f"[classifier] classify_clause error on clause {clause.get('id', '?')}: {e}")
            traceback.print_exc()
            # Return a safe fallback for this clause
            results.append({
                **clause,
                'type':         'general',
                'confidence':   0.0,
                'self_healed':  False,
                'risk_level':   'low',
                'flag_color':   'green',
                'plain_english': PLAIN_ENGLISH.get('general', ''),
                'is_risky':     False,
            })
    return results

def classify_clause(clause):
    full_text         = clause.get('full_text', '')
    clause_type, conf = detect_type(full_text)

    # ── Self-healing ──────────────────────────────────────────────────────────
    was_healed = False
    try:
        from rag_healer import heal_classification
        clause_type, conf, was_healed = heal_classification(full_text, clause_type, conf)
    except Exception as e:
        print(f"  Healer skipped: {e}")

    risk = RISK_LEVELS.get(clause_type, RISK_LEVELS['general'])
    return {
        **clause,
        'type':        clause_type,
        'confidence':  conf,
        'self_healed': was_healed,
        'risk_level':  risk['level'],
        'flag_color':  risk['color'],
        'plain_english': PLAIN_ENGLISH.get(clause_type, ''),
        'is_risky':    risk['level'] in ('high', 'medium'),
    }