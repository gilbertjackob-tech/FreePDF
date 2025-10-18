from pathlib import Path
p=Path(r'P:/Hasnat/pdf/utils/pdf_utils.py')
lines=p.read_text(encoding='utf-8').splitlines()
for i in range(720, 810):
    print(f"{i+1:4}: {lines[i]}")
