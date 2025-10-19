"""Utils package shims for FreePDF project.

This package provides minimal implementations of pdf utilities and a
lightweight SQL indexer so the Flask app can import and run during
development. The implementations try to use optional libraries when
available (Pillow, PyPDF2, python-docx) and fall back to safe no-op
behaviour otherwise. Replace these with production-grade implementations
when ready.
"""

__all__ = ["pdf_utils", "sql_indexer", "cv_utils"]
