# utils/pdf_edit_utils.py
from pypdf import PdfReader, PdfWriter

def edit_pdf(input_path, output_path, rotate=0, delete_list=None, reorder_list=None):
    """
    Real edit helper:
    - rotate pages by rotate degrees
    - delete pages in delete_list
    - reorder pages as per reorder_list
    """
    reader = PdfReader(input_path)
    writer = PdfWriter()

    pages = list(range(len(reader.pages)))

    # Delete pages if requested
    if delete_list:
        pages = [p for p in pages if p not in delete_list]

    # Reorder pages if provided
    if reorder_list:
        pages = [pages[i] for i in reorder_list if i < len(pages)]

    for i in pages:
        page = reader.pages[i]
        if rotate:
            page.rotate(rotate)
        writer.add_page(page)

    with open(output_path, 'wb') as f:
        writer.write(f)
