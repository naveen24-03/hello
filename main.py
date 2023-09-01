import PyPDF2

# Open the PDF file

with open('.pdf', 'rb') as pdf_file:
    # Create a PDF object
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # Get the number of pages in the PDF
    num_pages = len(pdf_reader.pages)

    # Extract text from each page
    text = ''
    for page_num in range(num_pages):
        page = pdf_reader.pages[]
        text += page.extract_text()

# Print the extracted text
print(text)
