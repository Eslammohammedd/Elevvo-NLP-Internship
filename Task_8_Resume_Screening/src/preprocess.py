import re
import io
import PyPDF2

def clean_text(text: str) -> str:
    """
    Cleans the input text by lowercasing, removing special characters,
    and reducing multiple whitespaces to a single space.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r'[^a-z0-9\s\.,;:-]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_text_from_pdf(file_obj) -> str:
    """
    Extracts text from a PDF file object.
    """
    text = ""
    try:
        reader = PyPDF2.PdfReader(file_obj)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def process_file(file_obj, file_type: str) -> str:
    """
    Processes an uploaded file based on its type (.txt or .pdf)
    and returns the cleaned text.
    """
    if file_type == "pdf":
        raw_text = extract_text_from_pdf(file_obj)
    else:
        # Assume text file
        try:
            raw_text = file_obj.read().decode('utf-8')
        except AttributeError:
            raw_text = file_obj.read()
            
    return clean_text(raw_text)
