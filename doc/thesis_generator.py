import re
from pylatex import Document, Package
from pylatex.utils import NoEscape

# Paths
TEX_PATH = r'WAT-WCY-1.0.tex'
PDFLATEX = r'C:\Users\radek\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe'

# Read the .tex
with open(TEX_PATH, 'r', encoding='utf-8') as f:
    tex_text = f.read()

# Split preamble/body
preamble_text, rest = re.split(r'\\begin\{document\}', tex_text, maxsplit=1)
body_text, _ = re.split(r'\\end\{document\}', rest, maxsplit=1)

# Parse documentclass
m = re.search(r'\\documentclass(?:\[(.*?)\])?\{(.*?)\}', preamble_text, re.DOTALL)
opts = m.group(1).split(',') if m.group(1) else None
doccls = m.group(2)
doc = Document(documentclass=doccls, document_options=opts)

# Replay usepackage lines with two-capture regex
usepkg_pattern = r'\\usepackage(?:\[(.*?)\])?\{(.*?)\}'
for pkg_opts, pkg_name in re.findall(usepkg_pattern, preamble_text):
    options = pkg_opts.split(',') if pkg_opts else None
    doc.packages.append(Package(pkg_name, options=options))

# Inject body and compile
doc.append(NoEscape(body_text))
tex_file, pdf_file = doc.generate_pdf(
    filepath='WAT-WCY-1.0',
    compiler=PDFLATEX,
    compiler_args=['-interaction=nonstopmode'],
    clean_tex=False
)
print(f"✔ Written PDF to {pdf_file}")
