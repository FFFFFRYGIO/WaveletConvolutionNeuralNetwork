import re
import shutil
import sys
from subprocess import CalledProcessError
from pylatex import Document, Package
from pylatex.utils import NoEscape

# Paths
TEX_PATH = r'WAT-WCY-1.0.tex'

# Locate pdflatex
PDFLATEX = shutil.which('pdflatex')
if PDFLATEX is None:
    sys.exit("❌ pdflatex not found in PATH. Please install TeX Live or add it to your PATH.")

# Read the .tex file
try:
    with open(TEX_PATH, 'r', encoding='utf-8') as f:
        tex_text = f.read()
except FileNotFoundError:
    sys.exit(f"❌ .tex file not found at {TEX_PATH}")

# Split into preamble and body
parts = re.split(r'\\begin\{document\}', tex_text, maxsplit=1)
if len(parts) != 2:
    sys.exit("❌ Could not find '\\begin{document}' in the .tex file.")

preamble_text, rest = parts
body_parts = re.split(r'\\end\{document\}', rest, maxsplit=1)
if len(body_parts) < 1:
    sys.exit("❌ Could not find '\\end{document}' in the .tex file.")

body_text = body_parts[0]

# Parse \documentclass
m = re.search(r'\\documentclass(?:\[(.*?)\])?\{(.*?)\}', preamble_text, re.DOTALL)
if not m:
    sys.exit("❌ Could not parse '\\documentclass' in preamble.")
opts = m.group(1).split(',') if m.group(1) else None
doccls = m.group(2)

# Create a PyLaTeX Document
doc = Document(documentclass=doccls, document_options=opts)

# Replay \usepackage lines
usepkg_pattern = r'\\usepackage(?:\[(.*?)\])?\{(.*?)\}'
for pkg_opts, pkg_name in re.findall(usepkg_pattern, preamble_text):
    options = pkg_opts.split(',') if pkg_opts else None
    doc.packages.append(Package(pkg_name, options=options))

# Insert the document body
doc.append(NoEscape(body_text))

# Determine output basename
tex_basename = TEX_PATH.rsplit('.', 1)[0]

# Try compiling; capture errors
try:
    tex_file, pdf_file = doc.generate_pdf(
        filepath=tex_basename,
        compiler=PDFLATEX,
        compiler_args=['-halt-on-error'],
        clean_tex=False
    )
    print(f"✔ Written PDF to {pdf_file} using compiler at {PDFLATEX}")
except CalledProcessError as e:
    # Retrieve raw output
    output = e.output.decode('utf-8', errors='ignore') if hasattr(e, 'output') else str(e)
    print("❌ pdflatex compilation failed with the following output:")
    print(output)

    # Show last lines of log for context
    log_file = f"{tex_basename}.log"
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as logf:
            lines = logf.readlines()
    except FileNotFoundError:
        lines = []

    if lines:
        print("\nLast lines of log file:")
        for line in lines[-20:]:
            print(line.rstrip())

        # Detect common unit errors
        if any("Illegal unit of measure" in ln for ln in lines):
            print("\n⚠ Detected an 'Illegal unit of measure' error. This typically means there's a stray character or missing unit in a length specification (e.g., an extra '*' or missing 'pt'). Check the indicated line in your .tex source for such issues.")

        # Detect missing number errors
        if any("Missing number, treated as zero" in ln for ln in lines) or any("Missing number" in ln for ln in lines):
            print("\n⚠ Detected a 'Missing number' error. TeX expected a numeric value (e.g., a dimension or counter). Look for stray braces, an asterisk, or a misplaced command around the reported line in your .tex source.")
    else:
        print("(No log file found to display further details.)")

    sys.exit(e.returncode)
