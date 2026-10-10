"""Reproduce native-sample, browser interaction, and large-file regression tests.

Example (from the repository root):
  python -m pip install playwright numpy soundfile
  python -m playwright install chromium
  python tests/audio_annotator_v2/run.py --large --audio /path/to/recording.wav

Repeat --audio to test every original recording, including local multi-GB files.
Use --inject-html only when an environment policy blocks file:// navigation.
Nothing is uploaded. Synthetic fixtures and reports stay in the selected directory.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
from fixtures import generate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--fixture-dir', type=Path, default=Path.home()/'.cache/audio-annotator-v2-tests')
    parser.add_argument('--audio', action='append', type=Path, default=[])
    parser.add_argument('--large', action='store_true', help='Create the 484 MB FLAC, 2.15 GB WAV and 4.8 GB RF64 fixtures; run the extra regressions.')
    parser.add_argument('--inject-html', action='store_true', help='Inject HTML into a real Chromium page instead of navigating file:// (restricted CI only).')
    parser.add_argument('--chromium', type=Path)
    parser.add_argument('--suite', choices=['all','numeric','interaction','frequency_scale','regression'], default='all')
    args = parser.parse_args()
    root, fixtures = args.root.resolve(), args.fixture_dir.expanduser().resolve()
    for p in [root/'audio_annotatorv2.html', root/'sample/251006_001_0002.WAV', *args.audio]:
        if not p.expanduser().is_file():
            parser.error(f'File does not exist: {p}')
    if args.suite == 'regression' and not args.large:
        parser.error('--suite regression requires --large')
    generate(fixtures, large=args.large)
    env = dict(os.environ, AUDIO_V2_ROOT=str(root), AUDIO_V2_FIXTURES=str(fixtures), AUDIO_V2_REPORTS=str(fixtures/'reports'), AUDIO_V2_FILES=json.dumps([str(p.expanduser().resolve()) for p in args.audio]), AUDIO_V2_INJECT='1' if args.inject_html else '0')
    if args.chromium:
        env['AUDIO_V2_CHROMIUM'] = str(args.chromium)
    suites = ['numeric', 'interaction', 'frequency_scale'] + (['regression'] if args.large else []) if args.suite == 'all' else [args.suite]
    codes = []
    for suite in suites:
        result = subprocess.run([sys.executable, str(Path(__file__).with_name(suite+'_test.py'))], env=env, check=False)
        codes.append(result.returncode)
    print(f'Reports: {fixtures / "reports"}')
    return 1 if any(codes) else 0


if __name__ == '__main__':
    raise SystemExit(main())
