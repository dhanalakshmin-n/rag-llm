import os
from pypdf import PdfReader

def load_document(file_path: str) -> str:
    """
    Load text from a document (PDF or TXT)
    """

    _, ext = os.path.splitext(file_path)

    # 📄 Handle PDF
    if ext.lower() == ".pdf":
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text

    # 📄 Handle TXT
    elif ext.lower() == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError("Unsupported file format")
