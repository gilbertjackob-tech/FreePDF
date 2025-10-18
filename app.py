"""Clean Flask app for FreePDF (development-only).

This file provides a single, unambiguous definition of the app and its
routes. It relies on simple shims in `utils/` so the server can start in
development without heavyweight native dependencies. Replace utilities
with production-ready implementations before deploying.
"""
import os
import uuid
import logging
from flask import Flask, render_template, request, send_file, redirect, flash, url_for, jsonify
from werkzeug.utils import secure_filename
from utils.pdf_utils import (
    split_pdf_custom,
    merge_pdfs,
    merge_pdfs_bytes,
    images_to_pdf,
    pdf_to_docx,
    edit_pdf,
    reorder_pages,
    generate_thumbnails,
)
from utils.sql_indexer import SQLIndexer

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
THUMB_FOLDER = os.path.join(BASE_DIR, 'thumbs')
DATA_DIR = os.path.join(BASE_DIR, 'data')
for d in (UPLOAD_FOLDER, OUTPUT_FOLDER, THUMB_FOLDER, DATA_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
address = '127.0.0.1' # default bind address

indexer = SQLIndexer(os.path.join(DATA_DIR, 'index.json'))


def allowed_ext(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'png', 'jpg', 'jpeg'}


def detect_mime(path: str) -> str:
    # lightweight mime detection
    if path.lower().endswith('.pdf'):
        return 'application/pdf'
    if path.lower().endswith(('.jpg', '.jpeg')):
        return 'image/jpeg'
    if path.lower().endswith('.png'):
        return 'image/png'
    return 'application/octet-stream'


@app.context_processor
def inject_csrf_token():
    # templates expect a csrf_token callable; we return None for dev simplicity
    return dict(csrf_token=None)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/split', methods=['GET', 'POST'])
def split():
    if request.method == 'POST':
        f = request.files.get('pdf')
        ranges = request.form.get('ranges')
        if not f or f.filename == '':
            return {'error': 'no file'}, 400
        filename = secure_filename(f.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        # index the uploaded source so clients can reference it for reruns
        try:
            indexer.set(unique, unique)
        except Exception:
            logger.exception('failed to index uploaded file')
        # ensure output filename has a PDF extension so browsers handle it correctly
        out_name = f'split_{unique}.pdf'
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)
        split_pdf_custom(in_path, ranges, out_path)
        # return with explicit download filename header
        resp = send_file(out_path, as_attachment=True)
        try:
            resp.headers['X-Download-Name'] = out_name
            resp.headers['Content-Disposition'] = f'attachment; filename="{out_name}"'
        except Exception:
            pass
        return resp
    return render_template('split.html')


@app.route('/merge', methods=['GET', 'POST'])
def merge():
    if request.method == 'POST':
        files = request.files.getlist('files') or request.files.getlist('pdfs')
        saved = []
        # Save uploaded files to disk (index them) but merge in-memory where possible
        for f in files:
            if not f or f.filename == '':
                continue
            filename = secure_filename(f.filename)
            unique = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            f.save(path)
            saved.append(path)
            indexer.set(unique, unique)
        try:
            merged_bytes, backend, elapsed = merge_pdfs_bytes(saved)
            if merged_bytes:
                from io import BytesIO
                resp = send_file(BytesIO(merged_bytes), as_attachment=True, download_name='merged.pdf', mimetype='application/pdf')
                try:
                    resp.headers['X-Merge-Backend'] = backend
                    resp.headers['X-Merge-Time'] = f"{elapsed:.3f}"
                except Exception:
                    pass
                return resp
        except Exception:
            logger.exception('in-memory merge failed, falling back to disk')
        # fallback to disk-based merge
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'merged_{uuid.uuid4().hex}.pdf')
        merge_pdfs(saved, out_path)
        return send_file(out_path, as_attachment=True)
    return render_template('merge.html')


@app.route('/merge-by-ids', methods=['POST'])
def merge_by_ids():
    # Accept ordered file ids
    file_ids = request.form.getlist('file_id[]') or request.form.getlist('file_id')
    if not file_ids:
        try:
            data = request.get_json(force=True)
            file_ids = data.get('file_ids') or data.get('ids')
        except Exception:
            file_ids = None
    if not file_ids:
        return {'error': 'no file_ids provided'}, 400
    saved_paths = []
    for fid in file_ids:
        mapped = indexer.get(fid)
        if not mapped:
            return {'error': f'file_id not found: {fid}'}, 404
        p = os.path.join(app.config['UPLOAD_FOLDER'], mapped)
        if not os.path.exists(p):
            return {'error': f'file not found on disk: {fid}'}, 404
        saved_paths.append(p)
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'merged_{uuid.uuid4().hex}.pdf')
    try:
        merged_bytes, backend, elapsed = merge_pdfs_bytes(saved_paths)
        if merged_bytes:
            from io import BytesIO
            resp = send_file(BytesIO(merged_bytes), as_attachment=True, download_name='merged.pdf', mimetype='application/pdf')
            try:
                resp.headers['X-Merge-Backend'] = backend
                resp.headers['X-Merge-Time'] = f"{elapsed:.3f}"
            except Exception:
                pass
            return resp
    except Exception:
        logger.exception('in-memory merge failed, falling back to disk')
    merge_pdfs(saved_paths, out_path)
    return send_file(out_path, as_attachment=True)


@app.route('/images-to-pdf', methods=['GET', 'POST'], endpoint='images_to_pdf')
def images_to_pdf_route():
    if request.method == 'POST':
        images = request.files.getlist('images')
        saved = []
        for img in images:
            if not img or img.filename == '':
                continue
            filename = secure_filename(img.filename)
            unique = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            img.save(path)
            saved.append(path)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'images_{uuid.uuid4().hex}.pdf')
        images_to_pdf(saved, out_path)
        return send_file(out_path, as_attachment=True)
    return render_template('images_to_pdf.html')


@app.route('/pdf-to-docx', methods=['GET', 'POST'], endpoint='pdf_to_docx')
def pdf_to_docx_route():
    if request.method == 'POST':
        f = request.files.get('pdf')
        if not f or f.filename == '':
            return {'error': 'no file'}, 400
        filename = secure_filename(f.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        # read optional parameters from the form (thresholds controlled by the UI)
        ocr_lang = request.form.get('ocr_lang') or request.args.get('ocr_lang') or 'eng'
        try:
            fragment_threshold = int(request.form.get('fragment_threshold') or request.args.get('fragment_threshold') or 8)
        except Exception:
            fragment_threshold = 8
        try:
            non_black_pct = float(request.form.get('non_black_pct') or request.args.get('non_black_pct') or 0.02)
        except Exception:
            non_black_pct = 0.02
        try:
            stdev_thresh = float(request.form.get('stdev_thresh') or request.args.get('stdev_thresh') or 2.5)
        except Exception:
            stdev_thresh = 2.5

        out_path = os.path.join(app.config['OUTPUT_FOLDER'], filename.rsplit('.', 1)[0] + '.docx')
        try:
            diagnostics = pdf_to_docx(
                in_path,
                out_path,
                ocr_lang=ocr_lang,
                fragment_threshold=fragment_threshold,
                non_black_pct=non_black_pct,
                stdev_thresh=stdev_thresh,
            )
        except TypeError:
            diagnostics = None
            try:
                pdf_to_docx(in_path, out_path)
            except Exception:
                pass

        resp = send_file(out_path, as_attachment=True)
        # Attach diagnostics in headers (JSON-safe short summary)
        try:
            import json
            if diagnostics and isinstance(diagnostics, dict):
                resp.headers['X-PDF2DOCX-Backend'] = diagnostics.get('backend', 'unknown')
                resp.headers['X-PDF2DOCX-PerPage'] = json.dumps(diagnostics.get('per_page', []))
            else:
                resp.headers['X-PDF2DOCX-Backend'] = 'unknown'
        except Exception:
            pass
        return resp
    return render_template('pdf_to_docx.html')


@app.route('/upload-for-edit', methods=['POST'])
def upload_for_edit():
    f = request.files.get('pdf')
    if not f or f.filename == '':
        return {'error': 'no file'}, 400
    filename = secure_filename(f.filename)
    unique = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
    f.save(path)
    indexer.set(unique, unique)
    return {'file_id': unique}, 200


# Rerun/force-OCR endpoint removed per user request (simplified no-OCR workflow)


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        # Accept either an uploaded file or a previously uploaded file_id
        f = request.files.get('pdf')
        file_id = request.form.get('file_id')
        rotate = int(request.form.get('rotate', '0'))
        pages = request.form.get('pages')
        delete = request.form.get('delete')

        if not f and not file_id:
            return {'error': 'no input'}, 400

        if f and f.filename:
            filename = secure_filename(f.filename)
            unique = f"{uuid.uuid4().hex}_{filename}"
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            f.save(in_path)
            indexer.set(unique, unique)
        else:
            mapped = indexer.get(file_id)
            if not mapped:
                return {'error': 'file_id not found'}, 404
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], mapped)

        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'edited_{uuid.uuid4().hex}.pdf')

        if pages:
            try:
                order = [int(x) for x in pages.split(',') if x.strip()]
            except Exception:
                return {'error': 'invalid pages'}, 400
            delete_list = None
            if delete:
                try:
                    delete_list = [int(x) for x in delete.split(',') if x.strip()]
                except Exception:
                    return {'error': 'invalid delete list'}, 400
            reorder_pages(in_path, order, out_path, rotate=rotate, delete_list=delete_list)
        else:
            edit_pdf(in_path, out_path, rotate=rotate)

        return send_file(out_path, as_attachment=True)
    return render_template('edit.html')


@app.route('/thumbnails', methods=['POST'])
def thumbnails():
    file_id = request.form.get('file_id')
    f = request.files.get('pdf')
    if f and f.filename:
        filename = secure_filename(f.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        indexer.set(unique, unique)
    elif file_id:
        mapped = indexer.get(file_id)
        if not mapped:
            return {'error': 'file not found'}, 404
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], mapped)
        if not os.path.exists(in_path):
            return {'error': 'file not found on disk'}, 404
    else:
        return {'error': 'no input'}, 400
    thumb_dir = os.path.join(THUMB_FOLDER, os.path.splitext(os.path.basename(in_path))[0])
    thumbs = generate_thumbnails(in_path, thumb_dir)
    urls = [url_for('serve_thumb', file=os.path.basename(thumb_dir), name=os.path.basename(p)) for p in thumbs]
    return {'file_id': os.path.basename(in_path), 'thumbs': urls}


@app.route('/thumbs/<file>/<name>')
def serve_thumb(file, name):
    p = os.path.join(THUMB_FOLDER, file, name)
    if not os.path.exists(p):
        return ('Not found', 404)
    return send_file(p, mimetype='image/jpeg')


@app.route('/camscanner')
def camscanner_index():
    return render_template('camscanner.html')


@app.route('/camscanner/scan')
def camscanner_scan():
    return render_template('camscanner/scan.html')


@app.route('/camscanner/id')
def camscanner_id():
    return render_template('camscanner/id.html')


@app.route('/scan-perspective', methods=['POST'])
def scan_perspective():
    # lightweight stub returning uploaded image (no perspective correction)
    f = request.files.get('image') or request.files.get('file')
    if not f or f.filename == '':
        return {'error': 'no image provided'}, 400
    filename = secure_filename(f.filename)
    tmp = os.path.join(app.config['OUTPUT_FOLDER'], f'scan_{uuid.uuid4().hex}_{filename}')
    f.save(tmp)
    return send_file(tmp, mimetype='image/jpeg')


if __name__ == '__main__':
    # Read requested bind host and port from environment, with safe defaults
    # Safer bind-host handling: prefer explicit developer-friendly defaults
    # and avoid creating test sockets which can interfere with Werkzeug's
    # reloader on Windows (observed WinError 10038). We consider the host
    # valid only if it matches a local interface address, '0.0.0.0' or
    # 'localhost'. Otherwise fall back to 127.0.0.1 or 127.0.0.1
    import socket

    requested = os.environ.get('BIND_HOST', address)
    port = int(os.environ.get('PORT', 1000))

    # Gather likely local addresses (hostname resolution plus localhost)
    local_addrs = {address, '::1', 'localhost'}
    try:
        hn = socket.gethostname()
        for a in socket.gethostbyname_ex(hn)[2]:
            local_addrs.add(a)
    except Exception:
        # best-effort only; if resolution fails we still proceed with defaults
        pass

    if requested in ('0.0.0.0', '::', '') or requested in local_addrs:
        bind_host = requested
    else:
        logger.warning('Requested bind host %s is not found among local interfaces; falling back to 127.0.0.1', requested)
        bind_host = address

    logger.info(f'Starting server on http://{bind_host}:{port} (requested: {requested})')
    app.run(host=bind_host, port=port, debug=True)
"""Flask PDF utilities app (clean, single-definition).

Implements routes: / (index), /split, /merge, /images-to-pdf, /pdf-to-docx,
/edit, /upload-for-edit, /thumbnails, /thumbs/<file>/<name>.

Designed to be simple and import-safe: only one route per endpoint.
"""

import os
import uuid
import logging
import threading
import time
from flask import (
    Flask,
    render_template,
    request,
    send_file,
    redirect,
    url_for,
    flash,
    after_this_request,
)
from werkzeug.utils import secure_filename
from utils.pdf_utils import (
    split_pdf_custom,
    merge_pdfs,
    images_to_pdf,
    pdf_to_docx,
    edit_pdf,
    reorder_pages,
    generate_thumbnails,
)
from utils.sql_indexer import SQLIndexer

try:
    import magic
except Exception:
    magic = None

# Optional heavy imports: enable via environment variables when you need them.
# Default to enabled for full-image processing features in local dev.
ENABLE_OPENCV = os.environ.get('ENABLE_OPENCV', '1') in ('1', 'true', 'True')
if ENABLE_OPENCV:
    try:
        import cv2
    except Exception:
        cv2 = None
else:
    cv2 = None

# CSRF protection is optional for local development; disabled by default.
# Set ENABLE_CSRF=1 to enable Flask-WTF CSRF protection.
ENABLE_CSRF = os.environ.get('ENABLE_CSRF', '0') in ('1', 'true', 'True')
if ENABLE_CSRF:
    try:
        from flask_wtf import CSRFProtect
    except Exception:
        CSRFProtect = None
else:
    CSRFProtect = None


# Configuration
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
THUMB_FOLDER = os.path.join(BASE_DIR, 'thumbs')
DATA_DIR = os.path.join(BASE_DIR, 'data')
for d in (UPLOAD_FOLDER, OUTPUT_FOLDER, THUMB_FOLDER, DATA_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me')
# User requested to disable CSRF protection for easier local usage
# This turns off Flask-WTF CSRF checks. Remove if you want protection back.
app.config['WTF_CSRF_ENABLED'] = False

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'tiff'}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if CSRFProtect:
    csrf = CSRFProtect(app)
else:
    csrf = None

# Simple JSON-backed indexer shim instance
indexer = SQLIndexer(os.path.join(DATA_DIR, 'index.db'))


def allowed_ext(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_mime(path: str) -> str:
    if magic:
        try:
            return magic.from_file(path, mime=True)
        except Exception:
            pass
    ext = path.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        return 'application/pdf'
    if ext in ('jpg', 'jpeg'):
        return 'image/jpeg'
    if ext == 'png':
        return 'image/png'
    return 'application/octet-stream'


@app.context_processor
def inject_csrf_token():
    try:
        from flask_wtf.csrf import generate_csrf

        return dict(csrf_token=generate_csrf)
    except Exception:
        return dict(csrf_token=None)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/split', methods=['GET', 'POST'])
def split():
    if request.method == 'POST':
        f = request.files.get('pdf')
        ranges = request.form.get('ranges')
        if not f or f.filename == '':
            flash('No file uploaded')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        if not allowed_ext(filename) or filename.rsplit('.', 1)[-1].lower() != 'pdf':
            flash('PDF required')
            return redirect(request.url)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        if detect_mime(in_path) != 'application/pdf':
            os.remove(in_path)
            flash('Uploaded file is not a valid PDF')
            return redirect(request.url)
        out_name = f"split_{unique}"
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)
        split_pdf_custom(in_path, ranges, out_path)

        @after_this_request
        def cleanup(response):
            try:
                os.remove(in_path)
            except Exception:
                pass
            return response

        return send_file(out_path, as_attachment=True)
    return render_template('split.html')


@app.route('/merge-by-ids', methods=['POST'])
def merge_by_ids():
    # Accept ordered file ids as form fields 'file_id[]' or JSON {"file_ids": [...]}
    file_ids = request.form.getlist('file_id[]') or request.form.getlist('file_id')
    if not file_ids:
        try:
            data = request.get_json(force=True)
            file_ids = data.get('file_ids') or data.get('ids')
        except Exception:
            file_ids = None

    if not file_ids:
        return {'error': 'no file_ids provided'}, 400

    saved_paths = []
    for fid in file_ids:
        mapped = indexer.get(fid)
        if not mapped:
            return {'error': f'file_id not found: {fid}'}, 404
        p = os.path.join(app.config['UPLOAD_FOLDER'], mapped)
        if not os.path.exists(p):
            return {'error': f'file not found on disk: {fid}'}, 404
        saved_paths.append(p)

    out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'merged_{uuid.uuid4().hex}.pdf')
    merge_pdfs(saved_paths, out_path)
    return send_file(out_path, as_attachment=True)


@app.route('/merge', methods=['GET', 'POST'])
def merge():
    if request.method == 'POST':
        files = request.files.getlist('pdfs')
        saved = []
        for f in files:
            if not f or f.filename == '':
                continue
            filename = secure_filename(f.filename)
            unique = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            f.save(path)
            saved.append(path)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'merged_{uuid.uuid4().hex}.pdf')
        merge_pdfs(saved, out_path)
        return send_file(out_path, as_attachment=True)
    return render_template('merge.html')


@app.route('/images-to-pdf', methods=['GET', 'POST'])
def images_to_pdf_route():
    if request.method == 'POST':
        images = request.files.getlist('images')
        saved = []
        for img in images:
            if not img or img.filename == '':
                continue
            filename = secure_filename(img.filename)
            unique = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            img.save(path)
            saved.append(path)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'images_{uuid.uuid4().hex}.pdf')
        images_to_pdf(saved, out_path)
        return send_file(out_path, as_attachment=True)
    return render_template('images_to_pdf.html')


@app.route('/pdf-to-docx', methods=['GET', 'POST'])
def pdf_to_docx_route():
    if request.method == 'POST':
        f = request.files.get('pdf')
        if not f or f.filename == '':
            flash('No file uploaded')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], filename.rsplit('.', 1)[0] + '.docx')
        pdf_to_docx(in_path, out_path)
        return send_file(out_path, as_attachment=True)
    return render_template('pdf_to_docx.html')


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        f = request.files.get('pdf')
        file_id = request.form.get('file_id')
        rotate = int(request.form.get('rotate', '0'))
        pages = request.form.get('pages')
        delete = request.form.get('delete')

        if not f or f.filename == '':
            if not file_id:
                flash('No file uploaded')
                return redirect(request.url)
            mapped = indexer.get(file_id)
            if not mapped:
                flash('Referenced file not found')
                return redirect(request.url)
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], mapped)
            if not os.path.exists(in_path):
                flash('Referenced file not found on disk')
                return redirect(request.url)
            filename = os.path.basename(mapped)
        else:
            filename = secure_filename(f.filename)
            unique = f"{uuid.uuid4().hex}_{filename}"
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            f.save(in_path)
            if detect_mime(in_path) != 'application/pdf':
                os.remove(in_path)
                flash('Uploaded file is not a valid PDF')
                return redirect(request.url)
            indexer.set(unique, unique)

        out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'edited_' + filename)

        if pages:
            try:
                order = [int(x) for x in pages.split(',') if x.strip()]
            except Exception:
                flash('Invalid pages ordering')
                return redirect(request.url)
            delete_list = None
            if delete:
                try:
                    delete_list = [int(x) for x in delete.split(',') if x.strip()]
                except Exception:
                    flash('Invalid delete list')
                    return redirect(request.url)
            reorder_pages(in_path, order, out_path, rotate=rotate, delete_list=delete_list)
        else:
            edit_pdf(in_path, out_path, rotate=rotate)

        try:
            if not file_id:
                os.remove(in_path)
        except Exception:
            pass

        return send_file(out_path, as_attachment=True)
    return render_template('edit.html')


@app.route('/upload-for-edit', methods=['POST'])
def upload_for_edit():
    f = request.files.get('pdf')
    if not f or f.filename == '':
        return {'error': 'no file'}, 400
    filename = secure_filename(f.filename)
    unique = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
    f.save(path)
    indexer.set(unique, unique)
    return {'file_id': unique}, 200


@app.route('/thumbnails', methods=['POST'])
def thumbnails():
    file_id = request.form.get('file_id')
    f = request.files.get('pdf')
    if f and f.filename:
        filename = secure_filename(f.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        indexer.set(unique, unique)
    elif file_id:
        mapped = indexer.get(file_id)
        if not mapped:
            return {'error': 'file not found'}, 404
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], mapped)
        if not os.path.exists(in_path):
            return {'error': 'file not found on disk'}, 404
    else:
        return {'error': 'no input'}, 400

    thumb_dir = os.path.join(THUMB_FOLDER, os.path.splitext(os.path.basename(in_path))[0])
    thumbs = generate_thumbnails(in_path, thumb_dir)
    urls = [url_for('serve_thumb', file=os.path.basename(thumb_dir), name=os.path.basename(p)) for p in thumbs]
    return {'file_id': os.path.basename(in_path), 'thumbs': urls}


@app.route('/scan-perspective', methods=['POST'])
def scan_perspective():
    """Accepts a single image file and attempts to detect a document in the image
    and return a perspective-corrected (warped) JPEG. Uses OpenCV if available.
    Returns 500 if OpenCV not installed or 400 on missing file.
    """
    if cv2 is None:
        return {'error': 'opencv not installed on server'}, 500

    f = request.files.get('image') or request.files.get('file')
    if not f or f.filename == '':
        return {'error': 'no image provided'}, 400

    filename = secure_filename(f.filename)
    tmp_in = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{filename}")
    f.save(tmp_in)

    try:
        import numpy as np

        img = cv2.imread(tmp_in)
        if img is None:
            raise RuntimeError('failed to read image')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        # find contours
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        def order_points(pts):
            rect = np.zeros((4, 2), dtype='float32')
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect

        def four_point_transform(image, pts):
            rect = order_points(pts)
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype='float32')
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            return warped

        if screenCnt is not None:
            pts = screenCnt.reshape(4, 2)
            warped = four_point_transform(img, pts)
        else:
            # fallback: return a resized copy
            h, w = img.shape[:2]
            max_dim = 1200
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                warped = cv2.resize(img, (int(w * scale), int(h * scale)))
            else:
                warped = img

        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'scanned_{uuid.uuid4().hex}.jpg')
        # write as JPEG
        cv2.imwrite(out_path, warped)
        return send_file(out_path, mimetype='image/jpeg', as_attachment=False)

    except Exception as exc:
        app.logger.exception('scan-perspective failed')
        return {'error': 'processing failed', 'detail': str(exc)}, 500


@app.route('/thumbs/<file>/<name>')
def serve_thumb(file, name):
    p = os.path.join(THUMB_FOLDER, file, name)
    if not os.path.exists(p):
        return ('Not found', 404)
    return send_file(p, mimetype='image/jpeg')


@app.route('/merge-queue')
def merge_queue():
    return render_template('merge_queue.html')


@app.route('/camscanner')
def camscanner_index():
    return render_template('camscanner/index.html')


@app.route('/camscanner/scan')
def camscanner_scan():
    return render_template('camscanner/scan.html')


@app.route('/camscanner/id')
def camscanner_id():
    return render_template('camscanner/id.html')


@app.route('/scan-id', methods=['POST'])
def scan_id():
    """Accept 'front' and 'back' images and produce an A4 PDF with margins and scaled images.
    Supports form fields:
      - front (file)
      - back (file)
      - layout: 'side' or 'stack' (default 'side')
      - quality: JPEG quality 10-100 (optional, default 90)
    """
    front = request.files.get('front')
    back = request.files.get('back')
    layout = request.form.get('layout', 'side')
    try:
        quality = int(request.form.get('quality', '90'))
        quality = max(10, min(quality, 100))
    except Exception:
        quality = 90

    if not front or not back:
        return {'error': 'provide front and back images'}, 400

    temp_to_cleanup = []

    def process_blob(file_storage):
        fname = secure_filename(file_storage.filename)
        tmp_in = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{fname}")
        file_storage.save(tmp_in)
        temp_to_cleanup.append(tmp_in)
        # if cv2 available, try to warp, otherwise use the image as-is
        if cv2:
            try:
                import numpy as np
                img = cv2.imread(tmp_in)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                edged = cv2.Canny(gray, 75, 200)
                cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
                screenCnt = None
                for c in cnts:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    if len(approx) == 4:
                        screenCnt = approx; break

                def order_points(pts):
                    rect = np.zeros((4, 2), dtype='float32')
                    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
                    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]; return rect

                def four_point_transform(image, pts):
                    rect = order_points(pts); (tl, tr, br, bl) = rect
                    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                    maxWidth = max(int(widthA), int(widthB))
                    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                    maxHeight = max(int(heightA), int(heightB))
                    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype='float32')
                    M = cv2.getPerspectiveTransform(rect, dst)
                    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight)); return warped

                if screenCnt is not None:
                    pts = screenCnt.reshape(4, 2); warped = four_point_transform(img, pts)
                else:
                    h, w = img.shape[:2]; max_dim = 1600
                    if max(h, w) > max_dim:
                        scale = max_dim / float(max(h, w)); warped = cv2.resize(img, (int(w * scale), int(h * scale)))
                    else:
                        warped = img
                out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'camscan_{uuid.uuid4().hex}.jpg')
                cv2.imwrite(out_path, warped, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                temp_to_cleanup.append(out_path)
                return out_path
            except Exception:
                # fallback to saved file
                return tmp_in
        else:
            return tmp_in

    front_path = process_blob(front)
    back_path = process_blob(back)

    # Compose into an A4 page (landscape by default for side-by-side)
    from PIL import Image, ImageDraw, ImageFont

    # A4 at 300 DPI
    DPI = 300
    A4_W_IN, A4_H_IN = 8.27, 11.69
    a4_w = int(A4_W_IN * DPI)
    a4_h = int(A4_H_IN * DPI)
    margin_in = 0.4
    margin = int(margin_in * DPI)

    imgs = [Image.open(front_path).convert('RGB'), Image.open(back_path).convert('RGB')]

    # Determine layout
    layout = layout if layout in ('side', 'stack') else 'side'

    canvas = Image.new('RGB', (a4_w, a4_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # simple font attempt
    try:
        font = ImageFont.truetype('arial.ttf', 36)
    except Exception:
        font = None

    if layout == 'side':
        # available area
        area_w = a4_w - 2 * margin
        area_h = a4_h - 2 * margin
        half_w = int(area_w / 2)
        positions = [(margin, margin), (margin + half_w, margin)]
        bounds = [(half_w, area_h), (half_w, area_h)]
    else:
        area_w = a4_w - 2 * margin
        area_h = a4_h - 2 * margin
        half_h = int(area_h / 2)
        positions = [(margin, margin), (margin, margin + half_h)]
        bounds = [(area_w, half_h), (area_w, half_h)]

    labels = ['Front', 'Back']
    for i, im in enumerate(imgs):
        bw, bh = bounds[i]
        # fit image into bw x bh preserving aspect
        w, h = im.size
        scale = min(bw / w, bh / h)
        nw, nh = int(w * scale), int(h * scale)
        im_resized = im.resize((nw, nh), Image.LANCZOS)
        px = positions[i][0] + (bounds[i][0] - nw) // 2
        py = positions[i][1] + (bounds[i][1] - nh) // 2
        canvas.paste(im_resized, (px, py))
        # label
        lab = labels[i]
        tx = px + 10; ty = py + 10
        if font:
            draw.text((tx, ty), lab, fill=(0, 0, 0), font=font)
        else:
            draw.text((tx, ty), lab, fill=(0, 0, 0))

    out_pdf = os.path.join(app.config['OUTPUT_FOLDER'], f'idscan_a4_{uuid.uuid4().hex}.pdf')
    # save canvas as PDF at DPI
    canvas.save(out_pdf, 'PDF', resolution=DPI)

    # schedule cleanup of temp files after response is complete using a background thread
    def cleanup_later(paths):
        def _worker():
            time.sleep(3)
            for p in paths:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
        th = threading.Thread(target=_worker, daemon=True)
        th.start()

    cleanup_later(temp_to_cleanup + [out_pdf])
    return send_file(out_pdf, as_attachment=True)


if __name__ == '__main__':
    bind_host = os.environ.get('BIND_HOST', address)
    port = int(os.environ.get('PORT', 1000))
    logger.info(f'Starting server on http://{bind_host}:{port}')
    try:
        app.run(host=bind_host, port=port, debug=True)
    except OSError:
        logger.exception('Failed to bind server')
        raise
"""Clean minimal Flask app for PDF utilities.

Single-definition file to avoid duplicate endpoint registration errors.
"""

import os
import uuid
import logging
from flask import Flask, render_template, request, send_file, redirect, flash
from werkzeug.utils import secure_filename
from utils.pdf_utils import edit_pdf
from utils.sql_indexer import SQLIndexer
from flask_wtf import CSRFProtect

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
for d in (UPLOAD_FOLDER, OUTPUT_FOLDER, DATA_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

csrf = CSRFProtect(app)
indexer = SQLIndexer(os.path.join(DATA_DIR, 'index.db'))


def detect_mime(path: str) -> str:
    try:
        import magic

        return magic.from_file(path, mime=True)
    except Exception:
        return 'application/pdf' if path.lower().endswith('.pdf') else 'application/octet-stream'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        f = request.files.get('pdf')
        if not f or f.filename == '':
            flash('No file uploaded')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        if detect_mime(in_path) != 'application/pdf':
            flash('Uploaded file is not a valid PDF')
            return redirect(request.url)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'edited_' + filename)
        edit_pdf(in_path, out_path)
        return send_file(out_path, as_attachment=True)
    return render_template('edit.html')


if __name__ == '__main__':
    bind_host = os.environ.get('BIND_HOST', address)
    port = int(os.environ.get('PORT', 1000))
    logger.info(f'Starting server on http://{bind_host}:{port}')
    try:
        app.run(host=bind_host, port=port, debug=True)
    except OSError:
        logger.exception('Failed to bind server')
        raise
"""Clean minimal Flask app for PDF utilities (single-definition).

This file intentionally contains a single set of routes so Flask will not
raise endpoint overwrite errors during import.
"""

import os
import uuid
import logging
from flask import Flask, render_template, request, send_file, redirect, flash
from werkzeug.utils import secure_filename
from utils.pdf_utils import edit_pdf
from utils.sql_indexer import SQLIndexer
from flask_wtf import CSRFProtect

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
for d in (UPLOAD_FOLDER, OUTPUT_FOLDER, DATA_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

csrf = CSRFProtect(app)
indexer = SQLIndexer(os.path.join(DATA_DIR, 'index.db'))


def detect_mime(path: str) -> str:
    try:
        import magic

        return magic.from_file(path, mime=True)
    except Exception:
        return 'application/pdf' if path.lower().endswith('.pdf') else 'application/octet-stream'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        f = request.files.get('pdf')
        if not f or f.filename == '':
            flash('No file uploaded')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        if detect_mime(in_path) != 'application/pdf':
            flash('Uploaded file is not a valid PDF')
            return redirect(request.url)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'edited_' + filename)
        edit_pdf(in_path, out_path)
        return send_file(out_path, as_attachment=True)
    return render_template('edit.html')


if __name__ == '__main__':
    bind_host = os.environ.get('BIND_HOST', address)
    port = int(os.environ.get('PORT', 1000))
    logger.info(f'Starting server on http://{bind_host}:{port}')
    try:
        app.run(host=bind_host, port=port, debug=True)
    except OSError:
        logger.exception('Failed to bind server')
        raise
"""Clean single-definition Flask app for PDF utilities.

This minimal file intentionally contains only one definition per route so
Flask will not raise endpoint overwrite errors during import.
"""

import os
import uuid
import logging
from flask import Flask, render_template, request, send_file, redirect, flash
from werkzeug.utils import secure_filename
from utils.pdf_utils import edit_pdf
from utils.sql_indexer import SQLIndexer
from flask_wtf import CSRFProtect

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
for d in (UPLOAD_FOLDER, OUTPUT_FOLDER, DATA_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

csrf = CSRFProtect(app)
indexer = SQLIndexer(os.path.join(DATA_DIR, 'index.db'))


def detect_mime(path: str) -> str:
    # lightweight mime detection (optional python-magic if available)
    try:
        import magic

        return magic.from_file(path, mime=True)
    except Exception:
        return 'application/pdf' if path.lower().endswith('.pdf') else 'application/octet-stream'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        f = request.files.get('pdf')
        if not f or f.filename == '':
            flash('No file uploaded')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        if detect_mime(in_path) != 'application/pdf':
            flash('Uploaded file is not a valid PDF')
            return redirect(request.url)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'edited_' + filename)
        edit_pdf(in_path, out_path)
        return send_file(out_path, as_attachment=True)
    return render_template('edit.html')


if __name__ == '__main__':
    bind_host = os.environ.get('BIND_HOST', address)
    port = int(os.environ.get('PORT', 1000))
    logger.info(f'Starting server on http://{bind_host}:{port}')
    try:
        app.run(host=bind_host, port=port, debug=True)
    except OSError:
        logger.exception('Failed to bind server')
        raise
"""Clean, single-definition Flask app for PDF utilities.

This file replaces prior duplicated/fragmented versions and ensures each
route is defined once so Flask does not raise endpoint overwrite errors.
"""

import os
import uuid
import logging
from flask import (
    Flask,
    render_template,
    request,
    send_file,
    redirect,
    url_for,
    flash,
)
from werkzeug.utils import secure_filename
from utils.pdf_utils import (
    split_pdf_custom,
    merge_pdfs,
    images_to_pdf,
    pdf_to_docx,
    edit_pdf,
    reorder_pages,
    generate_thumbnails,
)
from utils.sql_indexer import SQLIndexer

try:
    import magic
except Exception:
    magic = None

from flask_wtf import CSRFProtect


# Basic configuration
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
THUMB_FOLDER = os.path.join(BASE_DIR, 'thumbs')
DATA_DIR = os.path.join(BASE_DIR, 'data')
for d in (UPLOAD_FOLDER, OUTPUT_FOLDER, THUMB_FOLDER, DATA_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

csrf = CSRFProtect(app)
indexer = SQLIndexer(os.path.join(DATA_DIR, 'index.db'))


def detect_mime(path: str) -> str:
    if magic:
        try:
            return magic.from_file(path, mime=True)
        except Exception:
            pass
    ext = path.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        return 'application/pdf'
    if ext in ('jpg', 'jpeg'):
        return 'image/jpeg'
    if ext == 'png':
        return 'image/png'
    return 'application/octet-stream'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/split', methods=['GET', 'POST'])
def split():
    if request.method == 'POST':
        f = request.files.get('pdf')
        ranges = request.form.get('ranges')
        if not f or f.filename == '':
            flash('No file uploaded')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'split_{unique}')
        split_pdf_custom(in_path, ranges, out_path)
        return send_file(out_path, as_attachment=True)
    return render_template('split.html')


@app.route('/merge', methods=['GET', 'POST'])
def merge():
    if request.method == 'POST':
        files = request.files.getlist('pdfs')
        saved = []
        for f in files:
            if not f or f.filename == '':
                continue
            filename = secure_filename(f.filename)
            unique = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            f.save(path)
            saved.append(path)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'merged.pdf')
        merge_pdfs(saved, out_path)
        return send_file(out_path, as_attachment=True)
    return render_template('merge.html')


@app.route('/images-to-pdf', methods=['GET', 'POST'])
def images_to_pdf_route():
    if request.method == 'POST':
        images = request.files.getlist('images')
        saved = []
        for img in images:
            if not img or img.filename == '':
                continue
            filename = secure_filename(img.filename)
            unique = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            img.save(path)
            saved.append(path)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'images_{uuid.uuid4().hex}.pdf')
        images_to_pdf(saved, out_path)
        return send_file(out_path, as_attachment=True)
    return render_template('images_to_pdf.html')


@app.route('/pdf-to-docx', methods=['GET', 'POST'])
def pdf_to_docx_route():
    if request.method == 'POST':
        f = request.files.get('pdf')
        if not f or f.filename == '':
            flash('No file uploaded')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], filename.rsplit('.', 1)[0] + '.docx')
        pdf_to_docx(in_path, out_path)
        return send_file(out_path, as_attachment=True)
    return render_template('pdf_to_docx.html')


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        f = request.files.get('pdf')
        file_id = request.form.get('file_id')
        rotate = int(request.form.get('rotate', '0'))
        pages = request.form.get('pages')
        delete = request.form.get('delete')

        if not f or f.filename == '':
            if not file_id:
                flash('No file uploaded')
                return redirect(request.url)
            mapped = indexer.get(file_id)
            if not mapped:
                flash('Referenced file not found')
                return redirect(request.url)
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], mapped)
            if not os.path.exists(in_path):
                flash('Referenced file not found on disk')
                return redirect(request.url)
            filename = os.path.basename(mapped)
        else:
            filename = secure_filename(f.filename)
            unique = f"{uuid.uuid4().hex}_{filename}"
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            f.save(in_path)
            indexer.set(unique, unique)

        out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'edited_' + filename)
        if pages:
            try:
                order = [int(x) for x in pages.split(',') if x.strip()]
            except Exception:
                flash('Invalid pages ordering')
                return redirect(request.url)
            delete_list = None
            if delete:
                try:
                    delete_list = [int(x) for x in delete.split(',') if x.strip()]
                except Exception:
                    flash('Invalid delete list')
                    return redirect(request.url)
            reorder_pages(in_path, order, out_path, rotate=rotate, delete_list=delete_list)
        else:
            edit_pdf(in_path, out_path, rotate=rotate)

        try:
            if not file_id:
                os.remove(in_path)
        except Exception:
            pass

        return send_file(out_path, as_attachment=True)
    return render_template('edit.html')


@app.route('/upload-for-edit', methods=['POST'])
def upload_for_edit():
    f = request.files.get('pdf')
    if not f or f.filename == '':
        return {'error': 'no file'}, 400
    filename = secure_filename(f.filename)
    unique = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
    f.save(path)
    indexer.set(unique, unique)
    return {'file_id': unique}, 200


@app.route('/thumbnails', methods=['POST'])
def thumbnails():
    file_id = request.form.get('file_id')
    f = request.files.get('pdf')
    if f and f.filename:
        filename = secure_filename(f.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        indexer.set(unique, unique)
    elif file_id:
        mapped = indexer.get(file_id)
        if not mapped:
            return {'error': 'file not found'}, 404
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], mapped)
        if not os.path.exists(in_path):
            return {'error': 'file not found on disk'}, 404
    else:
        return {'error': 'no input'}, 400

    thumb_dir = os.path.join(THUMB_FOLDER, os.path.splitext(os.path.basename(in_path))[0])
    thumbs = generate_thumbnails(in_path, thumb_dir)
    urls = [url_for('serve_thumb', file=os.path.basename(thumb_dir), name=os.path.basename(p)) for p in thumbs]
    return {'file_id': os.path.basename(in_path), 'thumbs': urls}


@app.route('/thumbs/<file>/<name>')
def serve_thumb(file, name):
    p = os.path.join(THUMB_FOLDER, file, name)
    if not os.path.exists(p):
        return ('Not found', 404)
    return send_file(p, mimetype='image/jpeg')


if __name__ == '__main__':
    bind_host = os.environ.get('BIND_HOST', address)
    port = int(os.environ.get('PORT', 1000))
    logger.info(f'Starting server on http://{bind_host}:{port}')
    try:
        app.run(host=bind_host, port=port, debug=True)
    except OSError:
        logger.exception('Failed to bind server')
        raise
"""Flask PDF utilities app (single clean definition).

This file intentionally keeps the app concise and avoids duplicated
route/function definitions which previously caused endpoint overwrite
errors when importing the module.
"""

import os
import uuid
import logging
from flask import (
    Flask,
    render_template,
    request,
    send_file,
    redirect,
    url_for,
    flash,
    after_this_request,
)
from werkzeug.utils import secure_filename
from utils.pdf_utils import (
    split_pdf_custom,
    merge_pdfs,
    images_to_pdf,
    pdf_to_docx,
    edit_pdf,
    reorder_pages,
    generate_thumbnails,
)
from utils.sql_indexer import SQLIndexer

try:
    import magic
except Exception:
    magic = None

from flask_wtf import CSRFProtect


# Configuration
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
THUMB_FOLDER = os.path.join(BASE_DIR, 'thumbs')
DATA_DIR = os.path.join(BASE_DIR, 'data')
for _d in (UPLOAD_FOLDER, OUTPUT_FOLDER, THUMB_FOLDER, DATA_DIR):
    os.makedirs(_d, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me')

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'tiff'}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

csrf = CSRFProtect(app)
indexer = SQLIndexer(os.path.join(DATA_DIR, 'index.db'))


def allowed_ext(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_mime(path: str) -> str:
    if magic:
        try:
            return magic.from_file(path, mime=True)
        except Exception:
            pass
    ext = path.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        return 'application/pdf'
    if ext in ('jpg', 'jpeg'):
        return 'image/jpeg'
    if ext == 'png':
        return 'image/png'
    return 'application/octet-stream'


@app.context_processor
def inject_csrf_token():
    try:
        from flask_wtf.csrf import generate_csrf

        return dict(csrf_token=generate_csrf)
    except Exception:
        return dict(csrf_token=None)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    """Single edit endpoint: accepts either a direct PDF upload or a
    previously-uploaded file id. Returns an edited PDF as attachment.
    """
    if request.method == 'POST':
        f = request.files.get('pdf')
        file_id = request.form.get('file_id')
        rotate = int(request.form.get('rotate', '0'))
        pages = request.form.get('pages')
        delete = request.form.get('delete')

        if not f or f.filename == '':
            if not file_id:
                flash('No file uploaded')
                return redirect(request.url)
            mapped = indexer.get(file_id)
            if not mapped:
                flash('Referenced file not found')
                return redirect(request.url)
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], mapped)
            if not os.path.exists(in_path):
                flash('Referenced file not found on disk')
                return redirect(request.url)
            filename = os.path.basename(mapped)
        else:
            filename = secure_filename(f.filename)
            unique = f"{uuid.uuid4().hex}_{filename}"
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            f.save(in_path)
            if detect_mime(in_path) != 'application/pdf':
                os.remove(in_path)
                flash('Uploaded file is not a valid PDF')
                return redirect(request.url)
            indexer.set(unique, unique)

        out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'edited_' + filename)

        if pages:
            try:
                order = [int(x) for x in pages.split(',') if x.strip()]
            except Exception:
                flash('Invalid pages ordering')
                return redirect(request.url)
            delete_list = None
            if delete:
                try:
                    delete_list = [int(x) for x in delete.split(',') if x.strip()]
                except Exception:
                    flash('Invalid delete list')
                    return redirect(request.url)
            reorder_pages(in_path, order, out_path, rotate=rotate, delete_list=delete_list)
        else:
            edit_pdf(in_path, out_path, rotate=rotate)

        try:
            if not file_id:
                os.remove(in_path)
        except Exception:
            pass

        return send_file(out_path, as_attachment=True)

    return render_template('edit.html')


if __name__ == '__main__':
    bind_host = os.environ.get('BIND_HOST', address)
    port = int(os.environ.get('PORT', 1000))
    logger.info(f'Starting server on http://{bind_host}:{port}')
    try:
        app.run(host=bind_host, port=port, debug=True)
    except OSError:
        logger.exception('Failed to bind server')
        raise
"""Flask app for PDF utilities (clean, single-definition)."""

import os
import uuid
import logging
from flask import (
    Flask,
    render_template,
    request,
    send_file,
    redirect,
    url_for,
    flash,
    after_this_request,
)
from werkzeug.utils import secure_filename
from utils.pdf_utils import (
    split_pdf_custom,
    merge_pdfs,
    images_to_pdf,
    pdf_to_docx,
    edit_pdf,
    reorder_pages,
    generate_thumbnails,
)
from utils.sql_indexer import SQLIndexer

try:
    import magic
except Exception:
    magic = None

from flask_wtf import CSRFProtect


# Folders and configuration
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
THUMB_FOLDER = os.path.join(BASE_DIR, 'thumbs')
DATA_DIR = os.path.join(BASE_DIR, 'data')
for d in (UPLOAD_FOLDER, OUTPUT_FOLDER, THUMB_FOLDER, DATA_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me')

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'tiff'}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

csrf = CSRFProtect(app)
indexer = SQLIndexer(os.path.join(DATA_DIR, 'index.db'))


def allowed_ext(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_mime(path: str) -> str:
    if magic:
        try:
            return magic.from_file(path, mime=True)
        except Exception:
            pass
    ext = path.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        return 'application/pdf'
    if ext in ('jpg', 'jpeg'):
        return 'image/jpeg'
    if ext == 'png':
        return 'image/png'
    return 'application/octet-stream'


@app.context_processor
def inject_csrf_token():
    try:
        from flask_wtf.csrf import generate_csrf

        return dict(csrf_token=generate_csrf)
    except Exception:
        return dict(csrf_token=None)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/split', methods=['GET', 'POST'])
def split():
    if request.method == 'POST':
        f = request.files.get('pdf')
        ranges = request.form.get('ranges')
        if not f or f.filename == '':
            flash('No file uploaded')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        if not allowed_ext(filename) or filename.rsplit('.', 1)[-1].lower() != 'pdf':
            flash('PDF required')
            return redirect(request.url)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        if detect_mime(in_path) != 'application/pdf':
            os.remove(in_path)
            flash('Uploaded file is not a valid PDF')
            return redirect(request.url)
        out_name = f"split_{unique}"
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)
        split_pdf_custom(in_path, ranges, out_path)

        @after_this_request
        def cleanup(response):
            try:
                os.remove(in_path)
            except Exception:
                pass
            return response

        return send_file(out_path, as_attachment=True)
    return render_template('split.html')


@app.route('/merge', methods=['GET', 'POST'])
def merge():
    if request.method == 'POST':
        files = request.files.getlist('pdfs')
        if not files:
            flash('No files uploaded')
            return redirect(request.url)
        saved = []
        for f in files:
            if not f or f.filename == '':
                continue
            filename = secure_filename(f.filename)
            if filename.rsplit('.', 1)[-1].lower() != 'pdf':
                flash('Only PDF files allowed for merge')
                return redirect(request.url)
            unique = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            f.save(path)
            if detect_mime(path) != 'application/pdf':
                os.remove(path)
                flash('One of the uploaded files is not a valid PDF')
                return redirect(request.url)
            indexer.set(unique, unique)
            saved.append(path)
        if not saved:
            flash('No valid PDF files')
            return redirect(request.url)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'merged.pdf')
        merge_pdfs(saved, out_path)
        for p in saved:
            try:
                os.remove(p)
            except Exception:
                pass
        return send_file(out_path, as_attachment=True)
    return render_template('merge.html')


@app.route('/images-to-pdf', methods=['GET', 'POST'])
def images_to_pdf_route():
    if request.method == 'POST':
        images = request.files.getlist('images')
        if not images:
            flash('No images uploaded')
            return redirect(request.url)
        saved = []
        for img in images:
            if not img or img.filename == '':
                continue
            filename = secure_filename(img.filename)
            ext = filename.rsplit('.', 1)[-1].lower()
            if ext not in {'png', 'jpg', 'jpeg', 'tiff', 'gif'}:
                flash('Unsupported image type')
                return redirect(request.url)
            unique = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            img.save(path)
            saved.append(path)
        if not saved:
            flash('No valid images')
            return redirect(request.url)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'images_{uuid.uuid4().hex}.pdf')
        images_to_pdf(saved, out_path)
        for p in saved:
            try:
                os.remove(p)
            except Exception:
                pass
        return send_file(out_path, as_attachment=True)
    return render_template('images_to_pdf.html')


@app.route('/pdf-to-docx', methods=['GET', 'POST'])
def pdf_to_docx_route():
    if request.method == 'POST':
        f = request.files.get('pdf')
        if not f or f.filename == '':
            flash('No file uploaded')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        if filename.rsplit('.', 1)[-1].lower() != 'pdf':
            flash('Please upload a PDF')
            return redirect(request.url)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        if detect_mime(in_path) != 'application/pdf':
            os.remove(in_path)
            flash('Uploaded file is not a valid PDF')
            return redirect(request.url)
        indexer.set(unique, unique)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], filename.rsplit('.', 1)[0] + '.docx')
        pdf_to_docx(in_path, out_path)
        try:
            os.remove(in_path)
        except Exception:
            pass
        try:
            os.remove(in_path)
        except Exception:
            logger.exception('Failed to remove temp pdf')

        return send_file(out_path, as_attachment=True)
    return render_template('pdf_to_docx.html')



@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        f = request.files.get('pdf')
        file_id = request.form.get('file_id')
        rotate = int(request.form.get('rotate', '0'))
        pages = request.form.get('pages')
        delete = request.form.get('delete')
        if not f or f.filename == '':
            if not file_id:
                flash('No file uploaded')
                return redirect(request.url)
            mapped = indexer.get(file_id)
            if not mapped:
                flash('Referenced file not found')
                return redirect(request.url)
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], mapped)
            if not os.path.exists(in_path):
                flash('Referenced file not found on disk')
                return redirect(request.url)
            filename = os.path.basename(mapped)
        else:
            filename = secure_filename(f.filename)
            unique = f"{uuid.uuid4().hex}_{filename}"
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            f.save(in_path)
            if detect_mime(in_path) != 'application/pdf':
                os.remove(in_path)
                flash('Uploaded file is not a valid PDF')
                return redirect(request.url)
            indexer.set(unique, unique)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'edited_' + filename)
        if pages:
            try:
                order = [int(x) for x in pages.split(',') if x.strip()]
            except Exception:
                flash('Invalid pages ordering')
                return redirect(request.url)
            delete_list = None
            if delete:
                try:
                    delete_list = [int(x) for x in delete.split(',') if x.strip()]
                except Exception:
                    flash('Invalid delete list')
                    return redirect(request.url)
            reorder_pages(in_path, order, out_path, rotate=rotate, delete_list=delete_list)
        else:
            edit_pdf(in_path, out_path, rotate=rotate)
        try:
            if not file_id:
                os.remove(in_path)
        except Exception:
            pass
        return send_file(out_path, as_attachment=True)
    return render_template('edit.html')


@app.route('/upload-for-edit', methods=['POST'])
def upload_for_edit():
    f = request.files.get('pdf')
    if not f or f.filename == '':
        return {'error': 'no file'}, 400
    filename = secure_filename(f.filename)
    if not allowed_ext(filename) or filename.rsplit('.', 1)[-1].lower() != 'pdf':
        return {'error': 'pdf required'}, 400
    unique = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
    f.save(path)
    if detect_mime(path) != 'application/pdf':
        os.remove(path)
        return {'error': 'invalid pdf'}, 400
    indexer.set(unique, unique)
    return {'file_id': unique}, 200


@app.route('/thumbnails', methods=['POST'])
def thumbnails():
    file_id = request.form.get('file_id')
    f = request.files.get('pdf')
    if f and f.filename:
        filename = secure_filename(f.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        if detect_mime(in_path) != 'application/pdf':
            os.remove(in_path)
            return {'error': 'invalid pdf'}, 400
        indexer.set(unique, unique)
    elif file_id:
        mapped = indexer.get(file_id)
        if not mapped:
            return {'error': 'file not found'}, 404
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], mapped)
        if not os.path.exists(in_path):
            return {'error': 'file not found on disk'}, 404
    else:
        return {'error': 'no input'}, 400
    thumb_dir = os.path.join(THUMB_FOLDER, os.path.splitext(os.path.basename(in_path))[0])
    thumbs = generate_thumbnails(in_path, thumb_dir)
    urls = [url_for('serve_thumb', file=os.path.basename(thumb_dir), name=os.path.basename(p)) for p in thumbs]
    return {'file_id': os.path.basename(in_path), 'thumbs': urls}


@app.route('/thumbs/<file>/<name>')
def serve_thumb(file, name):
    p = os.path.join(THUMB_FOLDER, file, name)
    if not os.path.exists(p):
        return ('Not found', 404)
    return send_file(p, mimetype='image/jpeg')


if __name__ == '__main__':
    bind_host = os.environ.get('BIND_HOST', address)
    port = int(os.environ.get('PORT', 1000))
    logger.info(f'Starting server on http://{bind_host}:{port}')
    try:
        app.run(host=bind_host, port=port, debug=True)
    except OSError:
        logger.exception('Failed to bind server')
        raise


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        f = request.files.get('pdf')
        file_id = request.form.get('file_id')
        rotate = int(request.form.get('rotate', '0'))
        pages = request.form.get('pages')
        delete = request.form.get('delete')
        if not f or f.filename == '':
            if not file_id:
                flash('No file uploaded')
                return redirect(request.url)
            mapped = indexer.get(file_id)
            if not mapped:
                flash('Referenced file not found')
                return redirect(request.url)
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], mapped)
            if not os.path.exists(in_path):
                flash('Referenced file not found on disk')
                return redirect(request.url)
            filename = os.path.basename(mapped)
        else:
            filename = secure_filename(f.filename)
            unique = f"{uuid.uuid4().hex}_{filename}"
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            f.save(in_path)
            if detect_mime(in_path) != 'application/pdf':
                os.remove(in_path)
                flash('Uploaded file is not a valid PDF')
                return redirect(request.url)
            indexer.set(unique, unique)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'edited_' + filename)
        if pages:
            try:
                order = [int(x) for x in pages.split(',') if x.strip()]
            except Exception:
                flash('Invalid pages ordering')
                return redirect(request.url)
            delete_list = None
            if delete:
                try:
                    delete_list = [int(x) for x in delete.split(',') if x.strip()]
                except Exception:
                    flash('Invalid delete list')
                    return redirect(request.url)
            reorder_pages(in_path, order, out_path, rotate=rotate, delete_list=delete_list)
        else:
            edit_pdf(in_path, out_path, rotate=rotate)
        try:
            if not file_id:
                os.remove(in_path)
        except Exception:
            pass
        return send_file(out_path, as_attachment=True)
    return render_template('edit.html')


@app.route('/upload-for-edit', methods=['POST'])
def upload_for_edit():
    f = request.files.get('pdf')
    if not f or f.filename == '':
        return {'error': 'no file'}, 400
    filename = secure_filename(f.filename)
    if not allowed_ext(filename) or filename.rsplit('.', 1)[-1].lower() != 'pdf':
        return {'error': 'pdf required'}, 400
    unique = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
    f.save(path)
    if detect_mime(path) != 'application/pdf':
        os.remove(path)
        return {'error': 'invalid pdf'}, 400
    indexer.set(unique, unique)
    return {'file_id': unique}, 200


@app.route('/thumbnails', methods=['POST'])
def thumbnails():
    file_id = request.form.get('file_id')
    f = request.files.get('pdf')
    if f and f.filename:
        filename = secure_filename(f.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        if detect_mime(in_path) != 'application/pdf':
            os.remove(in_path)
            return {'error': 'invalid pdf'}, 400
        indexer.set(unique, unique)
    elif file_id:
        mapped = indexer.get(file_id)
        if not mapped:
            return {'error': 'file not found'}, 404
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], mapped)
        if not os.path.exists(in_path):
            return {'error': 'file not found on disk'}, 404
    else:
        return {'error': 'no input'}, 400
    thumb_dir = os.path.join(THUMB_FOLDER, os.path.splitext(os.path.basename(in_path))[0])
    thumbs = generate_thumbnails(in_path, thumb_dir)
    urls = [url_for('serve_thumb', file=os.path.basename(thumb_dir), name=os.path.basename(p)) for p in thumbs]
    return {'file_id': os.path.basename(in_path), 'thumbs': urls}


@app.route('/thumbs/<file>/<name>')
def serve_thumb(file, name):
    p = os.path.join(THUMB_FOLDER, file, name)
    if not os.path.exists(p):
        return ('Not found', 404)
    return send_file(p, mimetype='image/jpeg')


if __name__ == '__main__':
    bind_host = os.environ.get('BIND_HOST', address)
    port = int(os.environ.get('PORT', 1000))
    logger.info(f'Starting server on http://{bind_host}:{port}')
    try:
        app.run(host=bind_host, port=port, debug=True)
    except OSError:
        logger.exception('Failed to bind server')
        raise
    if __name__ == '__main__':
        bind_host = os.environ.get('BIND_HOST', address)
        port = int(os.environ.get('PORT', 1000))
        logger.info(f'Starting server on http://{bind_host}:{port}')
        try:
            app.run(host=bind_host, port=port, debug=True)
        except OSError:
            logger.exception('Failed to bind server')
            raise
@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        # The edit workflow can either accept file upload or a previously uploaded file_id
        f = request.files.get('pdf')
        file_id = request.form.get('file_id')
        rotate = int(request.form.get('rotate', '0'))
        pages = request.form.get('pages')  # optional reorder e.g. 3,1,2
        delete = request.form.get('delete')  # comma list of pages to delete
        if not f or f.filename == '':
            # If no file in request, check file_id mapped to upload
            if not file_id:
                flash('No file uploaded')
                return redirect(request.url)
            # file_id is expected to be filename saved in uploads
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
            if not os.path.exists(in_path):
                flash('Referenced file not found')
                return redirect(request.url)
        else:
            filename = secure_filename(f.filename)
            unique = f"{uuid.uuid4().hex}_{filename}"
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            f.save(in_path)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'edited_' + filename)

        # If reorder/delete provided, use reorder_pages helper
        if pages:
            try:
                order = [int(x) for x in pages.split(',') if x.strip()]
            except Exception:
                flash('Invalid pages ordering')
                return redirect(request.url)
            delete_list = None
            if delete:
                try:
                    delete_list = [int(x) for x in delete.split(',') if x.strip()]
                except Exception:
                    flash('Invalid delete list')
                    return redirect(request.url)
            reorder_pages(in_path, order, out_path, rotate=rotate, delete_list=delete_list)
        else:
            edit_pdf(in_path, out_path, rotate=rotate)

        try:
            os.remove(in_path)
        except Exception:
            logger.exception('Failed to remove temp upload')

        return send_file(out_path, as_attachment=True)
    return render_template('edit.html')


# Simple in-memory registry for recently uploaded files (maps file_id -> filename)
# Note: this is ephemeral (process memory) and only for local use.
recent_uploads = {}
INDEX_PATH = os.path.join(BASE_DIR, 'data', 'index.json')
indexer = SQLIndexer(INDEX_PATH)
csrf = CSRFProtect(app)


@app.route('/upload-for-edit', methods=['POST'])
def upload_for_edit():
    f = request.files.get('pdf')
    if not f or f.filename == '':
        return {'error': 'no file'}, 400
    filename = secure_filename(f.filename)
    unique = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
    f.save(path)
    # register on-disk
    indexer.set(unique, unique)
    return {'file_id': unique}, 200


@app.route('/thumbnails', methods=['POST'])
def thumbnails():
    # accepts either a file upload or a file_id referring to uploaded file
    file_id = request.form.get('file_id')
    f = request.files.get('pdf')
    if f and f.filename:
        # temporary save
        filename = secure_filename(f.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        indexer.set(unique, unique)
    elif file_id:
        mapped = indexer.get(file_id)
        if not mapped:
            return {'error': 'file not found'}, 404
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], mapped)
        if not os.path.exists(in_path):
            return {'error': 'file not found on disk'}, 404
    else:
        return {'error': 'no input'}, 400

    # generate thumbnails in thumbs/<file_id>/
    thumb_dir = os.path.join(THUMB_FOLDER, os.path.splitext(os.path.basename(in_path))[0])
    thumbs = generate_thumbnails(in_path, thumb_dir)
    # return list of URLs relative to /thumbs/
    urls = [url_for('serve_thumb', file=os.path.basename(thumb_dir), name=os.path.basename(p)) for p in thumbs]
    return {'file_id': os.path.basename(in_path), 'thumbs': urls}


@app.route('/thumbs/<file>/<name>')
def serve_thumb(file, name):
    p = os.path.join(THUMB_FOLDER, file, name)
    if not os.path.exists(p):
        return ('Not found', 404)
    return send_file(p, mimetype='image/jpeg')


if __name__ == '__main__':
    # Allow overriding bind host and port via environment variables.
    # Default host is set to the user's requested IP so other local project can use 127.0.0.1.
    bind_host = os.environ.get('BIND_HOST', address)
    port = int(os.environ.get('PORT', 1000))
    logger.info(f'Starting server on http://{bind_host}:{port}')
    try:
        app.run(host=bind_host, port=port, debug=True)
    except OSError as e:
        logger.exception('Failed to bind server')
        raise
import os
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
from utils.pdf_utils import split_pdf_custom, merge_pdfs, images_to_pdf, pdf_to_docx, edit_pdf
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.secret_key = 'replace-this-with-secure-key'
import uuid
import logging
from flask import after_this_request

# Configuration
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
# Max upload 100MB by default (adjust as needed)
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me')

# Security: allowed extensions
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'tiff'}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/split', methods=['GET', 'POST'])
def split():
    if request.method == 'POST':
        f = request.files.get('pdf')
        ranges = request.form.get('ranges')
        if not f:
            flash('No file uploaded')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(in_path)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'split_{filename}')
        filename = secure_filename(f.filename)
        ext = filename.rsplit('.', 1)[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            flash('File type not allowed')
            return redirect(request.url)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        out_name = f"split_{unique}"
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)
        split_pdf_custom(in_path, ranges, out_path)

        @after_this_request
        def cleanup(response):
            try:
                os.remove(in_path)
            except Exception:
                logger.exception('Failed to remove upload')
            return response

        return send_file(out_path, as_attachment=True)
def merge():
    if request.method == 'POST':
        files = request.files.getlist('pdfs')
        if not files:
            flash('No files uploaded')
            return redirect(request.url)
        saved = []
        for f in files:
            filename = secure_filename(f.filename)
            ext = filename.rsplit('.', 1)[-1].lower()
            if ext != 'pdf':
                flash('Only PDF files allowed for merge')
                return redirect(request.url)
            unique = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            f.save(path)
            saved.append(path)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'merged.pdf')
        merge_pdfs(saved, out_path)
        for p in saved:
            try:
                os.remove(p)
            except Exception:
                logger.exception('Failed to remove temp file')
        return send_file(out_path, as_attachment=True)
    return render_template('merge.html')


@app.route('/images-to-pdf', methods=['GET', 'POST'])
def images_to_pdf_route():
    if request.method == 'POST':
        images = request.files.getlist('images')
        if not images:
            flash('No images uploaded')
            return redirect(request.url)
        saved = []
        for img in images:
            filename = secure_filename(img.filename)
            ext = filename.rsplit('.', 1)[-1].lower()
            if ext not in {'png', 'jpg', 'jpeg', 'tiff', 'gif'}:
                flash('Unsupported image type')
                return redirect(request.url)
            unique = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            img.save(path)
            saved.append(path)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'images.pdf')
        images_to_pdf(saved, out_path)
        for p in saved:
            try:
                os.remove(p)
            except Exception:
                logger.exception('Failed to remove temp image')
        return send_file(out_path, as_attachment=True)
    return render_template('images_to_pdf.html')


@app.route('/pdf-to-docx', methods=['GET', 'POST'])
def pdf_to_docx_route():
    if request.method == 'POST':
        f = request.files.get('pdf')
        if not f:
            flash('No file uploaded')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        ext = filename.rsplit('.', 1)[-1].lower()
        if ext != 'pdf':
            flash('Please upload a PDF')
            return redirect(request.url)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], filename.rsplit('.', 1)[0] + '.docx')
        pdf_to_docx(in_path, out_path)

        try:
            os.remove(in_path)
        except Exception:
            logger.exception('Failed to remove temp pdf')

        return send_file(out_path, as_attachment=True)
    return render_template('pdf_to_docx.html')


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        f = request.files.get('pdf')
        rotate = int(request.form.get('rotate', '0'))
        pages = request.form.get('pages')  # optional reorder e.g. 3,1,2
        delete = request.form.get('delete')  # comma list of pages to delete
        if not f:
            flash('No file uploaded')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'edited_' + filename)

        # If reorder/delete provided, use reorder_pages helper
        if pages:
            # expect comma separated integers
            try:
                order = [int(x) for x in pages.split(',') if x.strip()]
            except Exception:
                flash('Invalid pages ordering')
                return redirect(request.url)
            reorder_pages(in_path, order, out_path, rotate=rotate, delete_list=None if not delete else [int(x) for x in delete.split(',') if x.strip()])
        else:
            edit_pdf(in_path, out_path, rotate=rotate)

        try:
            os.remove(in_path)
        except Exception:
            logger.exception('Failed to remove temp upload')

        return send_file(out_path, as_attachment=True)
    return render_template('edit.html')


if __name__ == '__main__':
    app.run(debug=True)
