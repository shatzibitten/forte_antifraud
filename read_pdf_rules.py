import pypdf
import os
import glob

# Find the PDF file
pdf_files = glob.glob('knowledges/*.pdf')
target_pdf = None
for f in pdf_files:
    if 'AI Hackathon ForteBank' in f:
        target_pdf = f
        break

if not target_pdf:
    print("PDF not found!")
    exit(1)

print(f"Reading {target_pdf}...")

try:
    reader = pypdf.PdfReader(target_pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    print(text)

except Exception as e:
    print(f"Error reading PDF: {e}")
