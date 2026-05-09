"""Render REPORT.md → HTML (with images inlined) → PDF + long PNG."""
import os, re, base64, markdown
from weasyprint import HTML, CSS

ROOT = os.path.expanduser("~/xunji-data/analysis")
IMG = f"{ROOT}/img"
OUT = f"{ROOT}/out"

with open(f"{ROOT}/REPORT.md") as f:
    md = f.read()

# Replace the "MEDIA:.../NN_xxx.png" trailing list and the附图清单 block:
# We'll instead inject images right after each section that mentions them.
# Strategy: at the end of section that mentions an image filename like `01_frequency.png`,
# insert the image. Simpler: scan for backticked filenames and inline image after the
# paragraph containing them.

# Build image map
imgs = sorted([f for f in os.listdir(IMG) if f.endswith(".png")])

# Strip the trailing MEDIA: list block & the 附图清单 list — we'll embed inline instead
md = re.sub(r"\n附图清单（按本报告引用顺序，下方逐张推送）：[\s\S]*$", "", md)

# Insert image after the FIRST paragraph that references its filename
def inject_image_after_first_mention(md_text, filename):
    pattern = re.compile(rf"(`{re.escape(filename)}`[^\n]*\n)")
    img_path = f"{IMG}/{filename}"
    img_md = f"\n![{filename}]({img_path})\n\n"
    new, n = pattern.subn(lambda m: m.group(1) + img_md, md_text, count=1)
    if n == 0:
        # fallback: append at end
        new = md_text + f"\n## {filename}\n{img_md}\n"
    return new

for fn in imgs:
    md = inject_image_after_first_mention(md, fn)

# Markdown → HTML
html_body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc"])

CSS_STYLE = """
@page { size: A4; margin: 18mm 14mm; @bottom-center { content: counter(page) " / " counter(pages); font-size: 9pt; color: #888; } }
body { font-family: "WenQuanYi Zen Hei", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; line-height: 1.55; color: #222; font-size: 10.5pt; }
h1 { font-size: 22pt; color: #1a1a1a; border-bottom: 3px solid #2563eb; padding-bottom: 8px; margin-top: 0; }
h2 { font-size: 15pt; color: #1e40af; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px; margin-top: 22px; page-break-after: avoid; }
h3 { font-size: 12pt; color: #334155; margin-top: 16px; page-break-after: avoid; }
h4 { font-size: 11pt; color: #475569; }
p { margin: 6px 0; }
blockquote { border-left: 4px solid #f59e0b; background: #fff7ed; padding: 8px 12px; margin: 10px 0; font-size: 10pt; color: #44403c; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; page-break-inside: avoid; }
th { background: #1e40af; color: white; padding: 6px 8px; text-align: left; font-weight: 600; }
td { padding: 5px 8px; border-bottom: 1px solid #e5e7eb; }
tr:nth-child(even) td { background: #f8fafc; }
code { background: #f1f5f9; padding: 1px 5px; border-radius: 3px; font-size: 9pt; font-family: "Courier New", monospace; }
pre { background: #f8fafc; border-left: 3px solid #94a3b8; padding: 8px 12px; font-size: 9pt; overflow-x: auto; page-break-inside: avoid; }
img { max-width: 100%; height: auto; display: block; margin: 12px auto; border: 1px solid #e2e8f0; border-radius: 4px; page-break-inside: avoid; }
ul, ol { margin: 6px 0 6px 22px; }
li { margin: 3px 0; }
hr { border: none; border-top: 2px dashed #cbd5e1; margin: 24px 0; }
strong { color: #1e40af; }
"""

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>训练数据深度分析报告</title>
<style>{CSS_STYLE}</style></head>
<body>{html_body}</body></html>"""

with open(f"{OUT}/REPORT.html", "w") as f:
    f.write(html)

# PDF
HTML(string=html, base_url=ROOT).write_pdf(f"{OUT}/REPORT.pdf")
print("PDF →", f"{OUT}/REPORT.pdf", os.path.getsize(f"{OUT}/REPORT.pdf")//1024, "KB")

# Long PNG (full report as one tall image): render via weasyprint to PNG sequence then stitch
# Simpler: use weasyprint write_png is removed in newer versions. Use chromium-headless? Not installed.
# Alternative: render each PDF page → PNG via pdftoppm, then stitch vertically via PIL.
import subprocess
subprocess.run(["pdftoppm", "-r", "300", "-png", f"{OUT}/REPORT.pdf", f"{OUT}/page"], check=True)
from PIL import Image
pages = sorted([f for f in os.listdir(OUT) if f.startswith("page-") and f.endswith(".png")])
print("pages:", len(pages))
ims = [Image.open(f"{OUT}/{p}") for p in pages]
W = max(im.width for im in ims)
H = sum(im.height for im in ims)
big = Image.new("RGB", (W, H), "white")
y = 0
for im in ims:
    big.paste(im, (0, y)); y += im.height
big.save(f"{OUT}/REPORT_full.png", optimize=True)
print("Long PNG →", f"{OUT}/REPORT_full.png", os.path.getsize(f"{OUT}/REPORT_full.png")//1024, "KB",
      f"{W}x{H}")
# Cleanup page-* intermediates
for p in pages:
    os.remove(f"{OUT}/{p}")
