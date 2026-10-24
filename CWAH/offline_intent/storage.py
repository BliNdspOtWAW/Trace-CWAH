"""One visible figure; reproducibility and resume files stay under details/."""
from pathlib import Path

DATA_FILES = ('manifest.json', 'intervals.jsonl', 'inference_config.json',
              'calls.jsonl', 'branch_cache.jsonl', 'results.jsonl', 'summary.json')

def data_directory(directory):
    directory = Path(directory)
    details = directory / 'details'
    # Validate every destination before moving legacy files. No overwrites.
    sources = [directory / name for name in DATA_FILES if (directory / name).exists()]
    if any((details / source.name).exists() for source in sources):
        raise ValueError('conflicting old/new analysis files; refusing to overwrite')
    details.mkdir(parents=True, exist_ok=True)
    for source in sources:
        source.rename(details / source.name)
    return details
