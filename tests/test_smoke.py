from subprocess import run, CalledProcessError
import sys

def test_cli_smoke():
    # Verifică dacă scriptul pornește și rulează un epoch rapid
    try:
        result = run([sys.executable, "-m", "fmri2image.cli"], check=True, capture_output=True)
    except CalledProcessError as e:
        print(e.stdout.decode(), e.stderr.decode())
        raise
