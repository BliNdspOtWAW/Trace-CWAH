"""Agreement report; never interprets JSD as prediction accuracy."""
import json
import math
from pathlib import Path
from .inference import read_rows
from .dataset import digest
from .storage import data_directory


def report(directory):
    output = Path(directory)
    directory = data_directory(output)
    intervals = read_rows(directory / 'intervals.jsonl')
    latest = {r['interval_id']: r for r in read_rows(directory / 'results.jsonl')} if (directory / 'results.jsonl').exists() else {}
    rows = []
    for index, interval in enumerate(intervals):
        if digest({k: v for k, v in interval.items() if k != 'input_hash'}) != interval['input_hash']:
            raise ValueError('frozen interval hash mismatch')
        result = latest.get(interval['interval_id'])
        if result and result['input_hash'] != interval['input_hash']:
            raise ValueError('result hash mismatch')
        rows.append(dict(index=index, interval_id=interval['interval_id'],
                         start_decision=interval['common']['start_decision_id'],
                         end_decision=interval['common']['end_decision_id'],
                         start_step=interval['start_step'], end_step=interval['end_step'],
                         new_event_count=interval['new_event_count'],
                         status=result['status'] if result else 'pending',
                         jsd_bits=result.get('jsd_bits') if result else None))
    values = sorted(r['jsd_bits'] for r in rows if r['status'] == 'ok')
    def quantile(p):
        if not values:
            return None
        position = (len(values) - 1) * p
        a, b = math.floor(position), math.ceil(position)
        return values[a] + (values[b] - values[a]) * (position - a)
    valid = len(values)
    failed = sum(r['status'] == 'failed' for r in rows)
    summary = {'interpretation': 'JSD is inference-mechanism agreement, NOT accuracy. No ground-truth evaluation.',
               'total_intervals': len(rows), 'valid': valid, 'failed': failed,
               'pending': len(rows) - valid - failed,
               'failure_rate_among_attempted': failed / (valid + failed) if valid + failed else None,
               'mean_jsd_bits': math.fsum(values) / valid if valid else None,
               'median_jsd_bits': quantile(0.5), 'p90_jsd_bits': quantile(0.9),
               'max_jsd_bits': values[-1] if values else None,
               'largest_disagreements': [r['interval_id'] for r in sorted(
                   (r for r in rows if r['status'] == 'ok'), key=lambda r: r['jsd_bits'], reverse=True)[:10]]}
    (directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot([r['index'] for r in rows], [r['jsd_bits'] if r['status'] == 'ok' else math.nan for r in rows], marker='.', label='A vs B')
    for status, color in [('failed', 'red'), ('pending', 'gray')]:
        indices = [r['index'] for r in rows if r['status'] == status]
        if indices:
            ax.scatter(indices, [-0.04] * len(indices), marker='x', color=color, label=status)
    ax.set(xlabel='Decision interval index (ordered, not environment step)', ylabel='JSD (bits)',
           ylim=(-0.07, 1.03), title='Retrospective start-Intent agreement; not accuracy')
    ax.legend()
    mean_label = f"{summary['mean_jsd_bits']:.5f}" if valid else 'N/A'
    median_label = f"{summary['median_jsd_bits']:.5f}" if valid else 'N/A'
    fig.text(0.5, 0.02,
             f"Valid {valid}/{len(rows)} | Failed {failed} | Pending {summary['pending']} | "
             f"Mean JSD {mean_label} | Median {median_label}",
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output / 'posterior_agreement.png', dpi=160)
    # Superseded generated exports only; keep all raw records and source data.
    for name in ('jsd_curve.png', 'jsd_curve.pdf', 'metrics.csv'):
        old = output / name
        if old.is_file() and not old.is_symlink():
            old.unlink()
    plt.close(fig)
    return summary
