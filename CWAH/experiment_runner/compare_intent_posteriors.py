"""Independent offline prepare / infer / report commands. No online TRACE imports."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from offline_intent.dataset import prepare
from offline_intent.inference import infer
from offline_intent.report import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='freeze legal inputs from a trusted CWAH trajectory')
    p.add_argument('--trajectory', required=True)
    p.add_argument('--output', required=True)
    p = commands.add_parser('infer', help='two independent Chat calls per interval; automatic resume')
    p.add_argument('--analysis', required=True)
    p.add_argument('--model', default=os.environ.get('CWAH_OFFLINE_MODEL') or os.environ.get('CWAH_AGENT_LM_ID'))
    p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--retries', type=int, default=2)
    p.add_argument('--limit', type=int, help='maximum unfinished intervals for this invocation')
    p = commands.add_parser('report', help='derive metrics and plots without model calls')
    p.add_argument('--analysis', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.trajectory, args.output)
    elif args.command == 'infer':
        if not args.model:
            parser.error('supply --model or CWAH_OFFLINE_MODEL/CWAH_AGENT_LM_ID')
        if (not math.isfinite(args.temperature) or args.temperature < 0 or args.max_tokens < 1
                or args.retries < 0 or (args.limit is not None and args.limit < 1)):
            parser.error('invalid inference parameters')
        result = infer(args.analysis, args.model, args.temperature, args.max_tokens, args.retries, args.limit)
    else:
        result = report(args.analysis)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
