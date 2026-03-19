#!/usr/bin/env python3
import sys
import subprocess

def ensure_pkg(pkg):
    try:
        __import__(pkg)
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', pkg])

def extract(path):
    ensure_pkg('PyPDF2')
    from PyPDF2 import PdfReader
    text_parts = []
    with open(path, 'rb') as f:
        reader = PdfReader(f)
        for p in reader.pages:
            try:
                t = p.extract_text() or ''
            except Exception:
                t = ''
            text_parts.append(t)
    return '\n\n'.join(text_parts)

def main():
    if len(sys.argv) < 2:
        print('Usage: extract_pdf_text.py <file.pdf>')
        sys.exit(2)
    path = sys.argv[1]
    txt = extract(path)
    sys.stdout.write(txt)

if __name__ == '__main__':
    main()
