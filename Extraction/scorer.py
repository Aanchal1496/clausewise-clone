import traceback

def get_verdict(safety_score):
    if safety_score >= 80:
        return {
            'verdict':      'safe',
            'color':        'green',
            'summary':      'This contract appears to be relatively standard. A full review is still recommended before signing.',
        }
    elif safety_score >= 50:
        return {
            'verdict':      'medium',
            'color':        'amber',
            'summary':      'This contract contains several clauses that warrant careful review before signing.',
        }
    else:
        return {
            'verdict':      'unsafe',
            'color':        'red',
            'summary':      'This contract contains serious red flags. Legal advice is strongly recommended before signing.',
        }


def score_contract(clauses):
    try:
        if not clauses:
            return {
                'score': 0,
                'verdict': 'safe',
                'color': 'green',
                'summary': 'No clauses found to analyze.',
                'clause_counts': {'high': 0, 'medium': 0, 'low': 0},
                'red_flags': [],
                'clauses': []
            }

        counts = {'high': 0, 'medium': 0, 'low': 0}
        for clause in clauses:
            level = clause.get('risk_level', 'low')
            counts[level] = counts.get(level, 0) + 1

        total_clauses = len(clauses)
        weighted_sum = (counts['low'] * 1.0) + (counts['medium'] * 0.5) + (counts['high'] * 0.0)
        safety_score = round((weighted_sum / total_clauses) * 100)
        verdict_info = get_verdict(safety_score)

        red_flags = [
            {
                'title':          c.get('title', ''),
                'type':           c.get('type', ''),
                'ai_explanation': c.get('ai_explanation', c.get('plain_english', '')),
            }
            for c in clauses if c.get('risk_level') == 'high'
        ]

        risk_score = 100 - safety_score

        return {
            'score':         risk_score,
            'verdict':       verdict_info['verdict'],
            'color':         verdict_info['color'],
            'summary':       verdict_info['summary'],
            'clause_counts': counts,
            'red_flags':     red_flags,
            'clauses':       clauses,
        }
    except Exception as e:
        print(f"[scorer] score_contract error: {e}")
        traceback.print_exc()
        return {
            'score': 50,
            'verdict': 'medium',
            'color': 'amber',
            'summary': 'Error calculating score. Partial results shown.',
            'clause_counts': {'high': 0, 'medium': 0, 'low': 0},
            'red_flags': [],
            'clauses': clauses or [],
        }