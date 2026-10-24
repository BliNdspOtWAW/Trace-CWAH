"""Download only the official Qwen3-8B files, with per-file SHA256 provenance."""
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    base = 'https://modelscope.cn/api/v1/models/Qwen/Qwen3-8B/repo'
    args.destination.mkdir(parents=True, exist_ok=True)
    manifest = args.destination / 'trace_download_manifest.json'
    if manifest.exists():
        provenance = json.loads(manifest.read_text())
        if provenance.get('repository') != 'Qwen/Qwen3-8B':
            raise ValueError('Unexpected existing download manifest')
        files = provenance['files']
    else:
        response = requests.get(base + '/files', params={'Revision': 'master', 'Recursive': 'true'}, timeout=(10, 30))
        response.raise_for_status()
        files = [f for f in response.json()['Data']['Files'] if f['Type'] == 'blob'
                 and (f['Path'].endswith(('.json', '.safetensors', '.jinja', '.txt', '.md'))
                      or f['Path'] == 'LICENSE')]
        provenance = {'repository': 'Qwen/Qwen3-8B', 'source': 'https://modelscope.cn/models/Qwen/Qwen3-8B',
                      'files': files}
        manifest.write_text(json.dumps(provenance, indent=2) + '\n')

    def download(item):
        path = args.destination / item['Path']
        if Path(item['Path']).name != item['Path']:
            raise ValueError('Only root-level model files are allowed')
        if path.exists():
            if digest(path) != item['Sha256']:
                raise ValueError(f'Existing model file checksum mismatch: {path.name}')
            print('VERIFIED_EXISTING', path.name, flush=True)
            return
        partial = path.with_name(path.name + '.part')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == item['Size'] and digest(partial) == item['Sha256']:
            partial.rename(path)
            print('VERIFIED', path.name, flush=True)
            return
        params = {'Revision': item['Revision'], 'FilePath': item['Path']}
        print('DOWNLOADING', path.name, 'bytes', item['Size'], flush=True)
        with requests.get(base, params=params, headers={'Range': f'bytes={offset}-'} if offset else {},
                          stream=True, timeout=(15, 120)) as result:
            result.raise_for_status()
            mode = 'ab' if offset and result.status_code == 206 else 'wb'
            with partial.open(mode) as stream:
                for chunk in result.iter_content(8 * 1024 * 1024):
                    stream.write(chunk)
        if partial.stat().st_size != item['Size'] or digest(partial) != item['Sha256']:
            raise ValueError(f'Download checksum mismatch: {path.name}')
        partial.rename(path)
        print('VERIFIED', path.name, flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, files))
    print('MODEL_READY', args.destination, flush=True)


if __name__ == '__main__':
    main()
