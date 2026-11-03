"""Save a teacher API key locally without printing it or making API requests."""
import argparse
import getpass
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def key_path(provider):
    if provider not in ('openai', 'anthropic'):
        raise ValueError('Choose openai or anthropic')
    return ROOT / 'data/processed/bam_api_credentials' / (provider + '.key')


def save_key(provider, key):
    if not isinstance(key, str) or len(key) < 10 or any(c.isspace() for c in key):
        raise ValueError('Expected a nonempty API key with no whitespace')
    path = key_path(provider)
    ignored = subprocess.run(['git', 'check-ignore', '--quiet', str(path.relative_to(ROOT))],
                             cwd=ROOT, capture_output=True)
    if ignored.returncode != 0:
        raise ValueError('Credential path must be git-ignored before saving')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    fd, temporary = tempfile.mkstemp(prefix='.key-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(key + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('openai', 'anthropic'), required=True)
    parser.add_argument('--check', action='store_true', help='Report presence only; never display the key')
    args = parser.parse_args()
    path = key_path(args.provider)
    if args.check:
        print(args.provider + ': ' + ('key file present' if path.is_file() else 'not configured'))
        return
    key = getpass.getpass(args.provider + ' API key (hidden): ').strip()
    save_key(args.provider, key)
    print('Saved private, git-ignored ' + args.provider + ' key. No API request or charge was made.')


if __name__ == '__main__':
    main()
