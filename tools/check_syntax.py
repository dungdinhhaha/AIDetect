import py_compile
import glob
import sys
import os

files = [f for f in glob.glob('**/*.py', recursive=True) if '__pycache__' not in f]
errors = []
count = 0
print('Checking', len(files), 'Python files for syntax errors...')
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        count += 1
    except Exception as e:
        errors.append((f, str(e)))

print('Done. compiled:', count, 'errors:', len(errors))
for f, e in errors:
    print('ERROR:', f)
    print('  ', e)

if errors:
    sys.exit(2)
else:
    sys.exit(0)
