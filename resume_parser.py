
import fitz
from docx import Document
from pathlib import Path


def extract_text_from_pdf(file_path):
    """Extract text from a PDF resume."""
    text = ""

    pdf = fitz.open(file_path)

    for page in pdf:
        text += page.get_text() + "\n"

    pdf.close()
    return text.strip()


def extract_text_from_docx(file_path):
    """Extract text from a Word resume."""
    document = Document(file_path)

    text = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text.append(paragraph.text)

    return "\n".join(text)


def extract_text_from_txt(file_path):
    """Extract text from a text resume."""
    return Path(file_path).read_text(
        encoding="utf-8",
        errors="ignore"
    )


def extract_resume_text(file_path):
    """Identify the file format and extract resume text."""
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)

    elif extension == ".docx":
        return extract_text_from_docx(file_path)

    elif extension == ".txt":
        return extract_text_from_txt(file_path)

    else:
        raise ValueError("Unsupported file format")


if __name__ == "__main__":
    print("Resume Parser Module Ready!")
