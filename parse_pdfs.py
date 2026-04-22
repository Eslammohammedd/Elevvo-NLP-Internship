import pypdf
import sys

def extract_pdf_to_txt(pdf_path, txt_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Successfully extracted {pdf_path} to {txt_path}")
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")

if __name__ == "__main__":
    extract_pdf_to_txt('Natural Language Processing Tasks.pdf', 'Natural_Language_Processing_Tasks.txt')
    extract_pdf_to_txt('Internship Instructions.pdf', 'Internship_Instructions.txt')
