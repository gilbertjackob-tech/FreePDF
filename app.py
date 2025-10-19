"""Clean Flask app for FreePDF (development-only).

This file provides a single, unambiguous definition of the app and its
routes. It relies on simple shims in `utils/` so the server can start in
development without heavyweight native dependencies. Replace utilities
with production-ready implementations before deploying.
"""

# ------------------------ All Imports ------------------------
import os, time, threading, numpy as np
import uuid
import logging
from flask import Flask, render_template, request, send_file, redirect, flash, url_for, jsonify, after_this_request
from werkzeug.utils import secure_filename
from utils.pdf_utils import (
    
    split_pdf_custom,
    merge_pdfs,
    merge_pdfs_bytes,
    pdf_to_docx,
    images_to_pdf_util,
    reorder_pages,
    generate_thumbnails,
)
from utils.sql_indexer import SQLIndexer
try:
    import magic
except Exception:
    magic = None

# CSRF protection is optional for local development; disabled by default.
# Set ENABLE_CSRF=1 to enable Flask-WTF CSRF protection.
ENABLE_OPENCV = os.environ.get('ENABLE_OPENCV', '1') in ('1', 'true', 'True')
ENABLE_CSRF = os.environ.get('ENABLE_CSRF', '0') in ('1', 'true', 'True')
if ENABLE_CSRF:
    try:
        from flask_wtf import CSRFProtect
    except Exception:
        CSRFProtect = None
else:
    CSRFProtect = None

if ENABLE_OPENCV:
    try:
        import cv2
    except Exception:
        cv2 = None
else:
    cv2 = None

#------------------------ Configuration ------------------------
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
THUMB_FOLDER = os.path.join(BASE_DIR, 'thumbs')
DATA_DIR = os.path.join(BASE_DIR, 'data')
for d in (UPLOAD_FOLDER, OUTPUT_FOLDER, THUMB_FOLDER, DATA_DIR):
    os.makedirs(d, exist_ok=True)

#------------------------ Flask App ------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', '!@#$%^&*()_+freepdf-dev-key')  # Replace in production!
# User requested to disable CSRF protection for easier local usage
# This turns off Flask-WTF CSRF checks. Remove if you want protection back.
app.config['WTF_CSRF_ENABLED'] = False


#------------------------ Logging ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#------------------------ Allowed Extensions ------------------------
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'tiff'}
address = '127.0.0.1' # default bind address
indexer = SQLIndexer(os.path.join(DATA_DIR, 'index.json'))
def allowed_ext(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def detect_mime(path: str) -> str:
    # lightweight mime detection
    if path.lower().endswith('.pdf'):
        return 'application/pdf'
    if path.lower().endswith(('.jpg', '.jpeg')):
        return 'image/jpeg'
    if path.lower().endswith('.png'):
        return 'image/png'
    return 'application/octet-stream'

#------------------------ CSRF Protection ------------------------
if CSRFProtect:
    csrf = CSRFProtect(app)
else:
    csrf = None
@app.context_processor
def inject_csrf_token():
    try:
        from flask_wtf.csrf import generate_csrf

        return dict(csrf_token=generate_csrf)
    except Exception:
        return dict(csrf_token=None)

#------------------------ Cleanup Old Files Thread ------------------------
# Cleanup function
def cleanup_old_files(interval_sec=60, max_age_sec=300):
    """Delete files older than max_age_sec every interval_sec seconds"""
    folders_to_clean = [UPLOAD_FOLDER, OUTPUT_FOLDER, THUMB_FOLDER]

    def cleaner():
        while True:
            now = time.time()
            for folder in folders_to_clean:
                if not os.path.exists(folder):
                    continue
                for f in os.listdir(folder):
                    path = os.path.join(folder, f)
                    if os.path.isfile(path) and os.path.getmtime(path) < now - max_age_sec:
                        try:
                            os.remove(path)
                            #print(f"Deleted old file: {path}")
                            pass
                        except Exception:
                            #print(f"Failed to delete {path}: {e}")
                            pass
            time.sleep(interval_sec)

    thread = threading.Thread(target=cleaner, daemon=True)
    thread.start()
try:
# Start cleanup thread when app starts
    cleanup_old_files()
except Exception:
    pass


#------------------------ Routes ------------------------
#------------------------ -------------------------------

#------------------------ 0. Home Page------------------------
@app.route('/')
def index():
    return render_template('index.html')

#------------------------ 1. Split PDF Page------------------------
@app.route('/split', methods=['GET', 'POST'])
def split():
    if request.method == 'POST':
        f = request.files.get('pdf')
        ranges = request.form.get('ranges')

        # --- Check for upload ---
        if not f or f.filename == '':
            msg = 'No file uploaded'
            if request.accept_mimetypes.best == 'application/json':
                return {'error': msg}, 400
            flash(msg)
            return redirect(request.url)

        # --- Validate file type ---
        filename = secure_filename(f.filename)
        ext = filename.rsplit('.', 1)[-1].lower()
        if ext not in ALLOWED_EXTENSIONS or ext != 'pdf':
            msg = 'Only PDF files are allowed'
            if request.accept_mimetypes.best == 'application/json':
                return {'error': msg}, 400
            flash(msg)
            return redirect(request.url)

        # --- Unique filename + save ---
        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)

        # --- Verify it's really a PDF ---
        if detect_mime(in_path) != 'application/pdf':
            os.remove(in_path)
            msg = 'Uploaded file is not a valid PDF'
            if request.accept_mimetypes.best == 'application/json':
                return {'error': msg}, 400
            flash(msg)
            return redirect(request.url)

        # --- Output setup ---
        out_name = f"split_{unique}.pdf"
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)

        # --- Split the PDF ---
        split_pdf_custom(in_path, ranges, out_path)

        # --- Cleanup after sending file ---
        @after_this_request
        def cleanup(response):
            try:
                os.remove(in_path)
            except Exception:
                logger.exception('Failed to remove uploaded file')
            return response

        # --- Prepare and return response ---
        resp = send_file(out_path, as_attachment=True)
        try:
            resp.headers['X-Download-Name'] = out_name
            resp.headers['Content-Disposition'] = f'attachment; filename="{out_name}"'
        except Exception:
            pass
        return resp
    return render_template('split.html')


#------------------------ 2. Merge PDF Page------------------------

#------------------------2.1 Main Merge Route----------------------
@app.route('/merge', methods=['GET', 'POST'])
def merge():
    if request.method == 'POST':
        # Accept 'files' or 'pdfs' field names
        files = request.files.getlist('files') or request.files.getlist('pdfs')
        if not files:
            msg = 'No files uploaded'
            if request.accept_mimetypes.best == 'application/json':
                return {'error': msg}, 400
            flash(msg)
            return redirect(request.url)

        saved = []

        # --- Validate and save PDFs ---
        for f in files:
            if not f or f.filename == '':
                continue
            filename = secure_filename(f.filename)
            ext = filename.rsplit('.', 1)[-1].lower()
            if ext != 'pdf':
                msg = 'Only PDF files allowed for merge'
                if request.accept_mimetypes.best == 'application/json':
                    return {'error': msg}, 400
                flash(msg)
                return redirect(request.url)

            unique = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            f.save(path)

            # MIME validation
            if detect_mime(path) != 'application/pdf':
                os.remove(path)
                msg = f'{filename} is not a valid PDF'
                if request.accept_mimetypes.best == 'application/json':
                    return {'error': msg}, 400
                flash(msg)
                return redirect(request.url)

            try:
                indexer.set(unique, unique)
            except Exception:
                logger.exception('Failed to index uploaded file')

            saved.append(path)

        if not saved:
            msg = 'No valid PDF files uploaded'
            if request.accept_mimetypes.best == 'application/json':
                return {'error': msg}, 400
            flash(msg)
            return redirect(request.url)

        # --- Try in-memory merge first ---
        try:
            merged_bytes, backend, elapsed = merge_pdfs_bytes(saved)
            if merged_bytes:
                from io import BytesIO
                resp = send_file(BytesIO(merged_bytes),
                                 as_attachment=True,
                                 download_name='merged.pdf',
                                 mimetype='application/pdf')
                try:
                    resp.headers['X-Merge-Backend'] = backend
                    resp.headers['X-Merge-Time'] = f"{elapsed:.3f}"
                except Exception:
                    pass
                # Cleanup temporary files
                for p in saved:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                return resp
        except Exception:
            logger.exception('In-memory merge failed; falling back to disk merge.')

        # --- Fallback: disk-based merge ---
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'merged_{uuid.uuid4().hex}.pdf')
        merge_pdfs(saved, out_path)

        # Cleanup uploaded files
        for p in saved:
            try:
                os.remove(p)
            except Exception:
                logger.exception('Failed to remove uploaded file')

        # Send merged PDF
        resp = send_file(out_path, as_attachment=True)
        try:
            resp.headers['X-Download-Name'] = os.path.basename(out_path)
        except Exception:
            pass
        return resp

    return render_template('merge.html')

#--------------------- 2.2 Merge by File IDs ----------------------
@app.route('/merge-by-ids', methods=['POST'])
def merge_by_ids():
    """
    Merge selected PDFs by their file_ids from indexer.
    Returns merged PDF, headers include backend info and timing.
    Cleans up temporary merged files automatically.
    """
    import threading, time
    # --- Accept file IDs from form or JSON ---
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
        if detect_mime(p) != 'application/pdf':
            return {'error': f'{fid} is not a valid PDF'}, 400
        saved_paths.append(p)

    if not saved_paths:
        return {'error': 'no valid files found'}, 400

    # --- Try in-memory merge ---
    try:
        merged_bytes, backend, elapsed = merge_pdfs_bytes(saved_paths)
        if merged_bytes:
            from io import BytesIO
            resp = send_file(BytesIO(merged_bytes),
                             as_attachment=True,
                             download_name='merged.pdf',
                             mimetype='application/pdf')
            try:
                resp.headers['X-Merge-Backend'] = backend
                resp.headers['X-Merge-Time'] = f"{elapsed:.3f}"
            except Exception:
                pass
            return resp
    except Exception:
        logger.exception('In-memory merge failed, falling back to disk merge.')

    # --- Disk-based merge fallback ---
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'merged_{uuid.uuid4().hex}.pdf')
    merge_pdfs(saved_paths, out_path)

    # --- Cleanup function ---
    def cleanup_later(paths):
        def _worker():
            time.sleep(5)  # wait for response to complete
            for p in paths:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
        threading.Thread(target=_worker, daemon=True).start()

    # Schedule cleanup of disk merge file
    cleanup_later([out_path])

    # Send merged PDF
    resp = send_file(out_path, as_attachment=True)
    try:
        resp.headers['X-Download-Name'] = os.path.basename(out_path)
    except Exception:
        pass

    return resp


#------------------------ 3. Images to PDF Page------------------------
@app.route('/images-to-pdf', methods=['GET', 'POST'])
def images_to_pdf():
    if request.method == 'POST':
        images = request.files.getlist('images')
        if not images:
            flash('No images uploaded')
            return redirect(request.url)

        # Get optional custom margin (in mm)
        try:
            custom_margin_mm = float(request.form.get('margin', '10'))
            custom_margin_mm = max(0, min(custom_margin_mm, 50))  # limit to 0-50 mm
        except Exception:
            custom_margin_mm = 10  # default 10 mm

        saved = []
        allowed_ext = {'png', 'jpg', 'jpeg', 'tiff', 'gif'}

        for img in images:
            if not img or img.filename.strip() == '':
                continue
            filename = secure_filename(img.filename)
            ext = filename.rsplit('.', 1)[-1].lower()
            if ext not in allowed_ext:
                flash(f'Unsupported image type: {ext.upper()}')
                return redirect(request.url)

            unique = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            img.save(path)
            saved.append(path)

        if not saved:
            flash('No valid images uploaded')
            return redirect(request.url)

        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'images_{uuid.uuid4().hex}.pdf')

        try:
            # Convert images to PDF with A4 sizing and custom margin
            images_to_pdf_util(saved, out_path, margin_mm=custom_margin_mm)
        except Exception:
            logger.exception('Error converting images to PDF')
            flash('Failed to convert images.')
            return redirect(request.url)
        finally:
            # Clean up temp images
            for p in saved:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    logger.exception('Failed to remove temp image')

        # Send PDF safely
        from io import BytesIO
        with open(out_path, 'rb') as f:
            pdf_bytes = BytesIO(f.read())
        # optional: remove after reading
        os.remove(out_path)

        return send_file(pdf_bytes, as_attachment=True, download_name='images.pdf', mimetype='application/pdf')

    return render_template('images_to_pdf.html')


#------------------------ 4. PDF to DOCX Page------------------------
@app.route('/pdf-to-docx', methods=['GET', 'POST'])
def pdf_to_docx():
    if request.method == 'POST':
        f = request.files.get('pdf')
        if not f or f.filename.strip() == '':
            flash('No file uploaded')
            return redirect(request.url)

        filename = secure_filename(f.filename)
        if filename.rsplit('.', 1)[-1].lower() != 'pdf':
            flash('Please upload a valid PDF file')
            return redirect(request.url)

        unique = f"{uuid.uuid4().hex}_{filename}"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        f.save(in_path)

        # Verify MIME
        if detect_mime(in_path) != 'application/pdf':
            os.remove(in_path)
            flash('Uploaded file is not a valid PDF')
            return redirect(request.url)

        # Optional processing parameters
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

        indexer.set(unique, unique)
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], filename.rsplit('.', 1)[0] + '.docx')

        diagnostics = None
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
            pdf_to_docx(in_path, out_path)
        except Exception as e:
            logger.exception('Error converting PDF to DOCX')
            flash('Failed to convert file.')
            return redirect(request.url)
        finally:
            try:
                os.remove(in_path)
            except Exception:
                logger.exception('Failed to remove temp PDF')

        # Prepare response
        resp = send_file(out_path, as_attachment=True)
        try:
            import json
            if diagnostics and isinstance(diagnostics, dict):
                resp.headers['X-PDF2DOCX-Backend'] = diagnostics.get('backend', 'unknown')
                resp.headers['X-PDF2DOCX-PerPage'] = json.dumps(diagnostics.get('per_page', []))
        except Exception:
            pass

        return resp

    return render_template('pdf_to_docx.html')


#------------------------ 5.3 PDF Pages Generate Thumbnails Part------------------------
@app.route('/thumbnails', methods=['POST'])
def thumbnails():
    """Generate page thumbnails for a PDF (uploaded or existing by file_id)."""
    from utils.pdf_utils import generate_thumbnails  # ensure this is implemented

    f = request.files.get('pdf')
    file_id = request.form.get('file_id')

    # --- Determine input PDF path ---
    if f and f.filename.strip():
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

    # --- Prepare thumbnail directory ---
    thumb_dir = os.path.join(THUMB_FOLDER, os.path.splitext(os.path.basename(in_path))[0])
    os.makedirs(thumb_dir, exist_ok=True)

    # --- Generate thumbnails ---
    thumbs = generate_thumbnails(in_path, thumb_dir)

    # --- Prepare URLs for frontend ---
    urls = []
    for p in thumbs:
        try:
            name = os.path.basename(p)
            folder = os.path.basename(thumb_dir)
            # use external URLs to avoid client ambiguity when behind a proxy/tunnel
            u = url_for('serve_thumb', file=folder, name=name, _external=True)
            urls.append(u)
            # log for debugging: whether file exists and its size
            if not os.path.exists(p):
                logger.warning('Thumbnail expected but missing: %s', p)
            else:
                try:
                    sz = os.path.getsize(p)
                    logger.debug('Thumbnail generated: %s (%d bytes)', p, sz)
                except Exception:
                    logger.debug('Thumbnail generated: %s', p)
        except Exception:
            logger.exception('Failed to prepare thumbnail URL for %s', p)

    return {'file_id': os.path.basename(in_path), 'thumbs': urls}





#------------------------ 5.4 PDF Edit Pages Show Thumbnails Endpoint------------------------
@app.route('/thumbs/<file>/<name>')
def serve_thumb(file, name):
    path = os.path.join(THUMB_FOLDER, file, name)
    if not os.path.exists(path):
        logger.warning('serve_thumb: requested thumb not found: %s', path)
        return ('Not found', 404)
    try:
        # rely on send_file to set appropriate headers; ensure we return an image mimetype
        return send_file(path, mimetype='image/jpeg')
    except Exception:
        logger.exception('serve_thumb: failed to send file %s', path)
        return ('Server error', 500)

#------------------------ 5.5 All PDF Edits Merge Queue Page------------------------
@app.route('/merge-queue')
def merge_queue():
    """
    Display all indexed PDFs for merging.
    Frontend can send selected file_ids to /merge-by-ids endpoint.
    """
    try:
        all_files = indexer.all()  # returns {file_id: filename}
    except Exception:
        all_files = {}
        logger.exception('Failed to retrieve indexer files')

    return render_template('merge_queue.html', files=all_files)


#------------------------ 6. CamScanner Page ------------------------

#------------------6.1 CamScanner Main Page-----------------------
@app.route('/camscanner')
def camscanner_index():
    return render_template('camscanner/index.html')

#------------------6.2 CamScanner Scan Page-----------------------
@app.route('/camscanner/scan')
def camscanner_scan():
    return render_template('camscanner/scan.html')

#------------------6.3 CamScanner ID Page-----------------------
@app.route('/camscanner/id')
def camscanner_id():
    return render_template('camscanner/id.html')

#------------------6.4 Scan Perspective (single image) - used by camera capture-----------------------
@app.route('/scan-perspective', methods=['POST'])
def scan_perspective():
    if cv2 is None:
        return {'error': 'opencv not installed'}, 500
    f = request.files.get('image') or request.files.get('file')
    if not f or f.filename.strip() == '':
        return {'error': 'no image provided'}, 400

    from utils.cv_utils import four_point_transform

    filename = secure_filename(f.filename)
    tmp_in = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{filename}")
    f.save(tmp_in)

    try:
        img = cv2.imread(tmp_in)
        if img is None:
            raise RuntimeError('failed to read image')

        # Detect document contour
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5),0)
        edged = cv2.Canny(gray, 75, 200)
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        # Warp or fallback resize
        if screenCnt is not None:
            warped = four_point_transform(img, screenCnt.reshape(4,2))
        else:
            h,w = img.shape[:2]
            max_dim = 1200
            if max(h,w) > max_dim:
                scale = max_dim / float(max(h,w))
                warped = cv2.resize(img, (int(w*scale), int(h*scale)))
            else:
                warped = img

        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'scanned_{uuid.uuid4().hex}.jpg')
        cv2.imwrite(out_path, warped)
        return send_file(out_path, mimetype='image/jpeg', as_attachment=False)

    except Exception as exc:
        logger.exception('scan-perspective failed')
        return {'error': 'processing failed', 'detail': str(exc)}, 500


#----------------------6.5 Merge multiple pages into PDF (used by scan.html final "Finish & Save")------------------------
@app.route('/merge-pdf', methods=['POST'])
def merge_pdf():
    from PIL import Image
    files = request.files.getlist('file')
    if not files:
        return {'error': 'No files uploaded'}, 400

    temp_paths = []
    imgs = []
    try:
        for f in files:
            fname = secure_filename(f.filename)
            tmp_in = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{fname}")
            f.save(tmp_in)
            temp_paths.append(tmp_in)
            img = Image.open(tmp_in).convert('RGB')
            imgs.append(img)

        out_pdf = os.path.join(app.config['OUTPUT_FOLDER'], f'scan_a4_{uuid.uuid4().hex}.pdf')
        imgs[0].save(out_pdf, save_all=True, append_images=imgs[1:])
        cleanup_later(temp_paths + [out_pdf])
        return send_file(out_pdf, as_attachment=True)
    except Exception as e:
        return {'error': str(e)}, 500

#------------------6.6 Scan ID (front/back) and create labeled A4 PDF-----------------------
@app.route('/scan-id', methods=['POST'])
def scan_id():
    """Accept 'front' and 'back' images and produce an A4 PDF with labels."""
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

        if cv2:
            try:
                import numpy as np
                img = cv2.imread(tmp_in)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5,5),0)
                edged = cv2.Canny(gray, 75, 200)
                cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
                screenCnt = None
                img_area = img.shape[0] * img.shape[1]
                for c in cnts:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02*peri, True)
                    if len(approx) == 4:
                        # Validate contour by area and aspect ratio to avoid bad warps
                        cnt_area = cv2.contourArea(approx)
                        if cnt_area < 0.005 * img_area or cnt_area > 0.95 * img_area:
                            logger.debug('Ignoring approx contour by area: %s (img_area=%s)', cnt_area, img_area)
                            continue
                        # compute bounding rect aspect ratio
                        x,y,w,h = cv2.boundingRect(approx)
                        ar = float(w)/float(h) if h>0 else 0
                        # ID cards typically have aspect ratio ~1.5-1.8 (landscape) or ~0.6-0.7 (portrait)
                        if not (0.5 <= ar <= 2.5):
                            logger.debug('Ignoring approx contour by aspect ratio: %s', ar)
                            continue
                        screenCnt = approx
                        break

                def order_points(pts):
                    rect = np.zeros((4,2), dtype='float32')
                    s = pts.sum(axis=1); rect[0]=pts[np.argmin(s)]; rect[2]=pts[np.argmax(s)]
                    diff = np.diff(pts, axis=1); rect[1]=pts[np.argmin(diff)]; rect[3]=pts[np.argmax(diff)]
                    return rect

                def four_point_transform(image, pts):
                    rect = order_points(pts)
                    (tl,tr,br,bl)=rect
                    widthA = np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))
                    widthB = np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))
                    maxWidth = max(int(widthA), int(widthB))
                    heightA = np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
                    heightB = np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
                    maxHeight = max(int(heightA), int(heightB))
                    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype='float32')
                    M = cv2.getPerspectiveTransform(rect,dst)
                    warped = cv2.warpPerspective(image,M,(maxWidth,maxHeight))
                    return warped

                if screenCnt is not None:
                    warped = four_point_transform(img, screenCnt.reshape(4,2))
                    # sanity check: if warped is too small or nearly uniform color, fallback to original
                    try:
                        import numpy as _np
                        if warped is None or warped.size == 0:
                            warped = img
                        else:
                            # check variance; if very low variance it's likely a bad warp
                            if _np.var(warped) < 10:
                                logger.warning('Warped image has very low variance; falling back to original')
                                warped = img
                    except Exception:
                        # if any check fails, keep warped as-is to avoid crashing
                        pass
                else:
                    h,w = img.shape[:2]; max_dim=1600
                    if max(h,w)>max_dim:
                        scale=max_dim/float(max(h,w))
                        warped = cv2.resize(img,(int(w*scale),int(h*scale)))
                    else:
                        warped=img
                out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'camscan_{uuid.uuid4().hex}.jpg')
                cv2.imwrite(out_path, warped, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                temp_to_cleanup.append(out_path)
                return out_path
            except Exception as e:
                # Log the exception so server logs show what failed during processing
                logger.exception('process_blob failed while processing %s', tmp_in)
                return tmp_in
        else:
            return tmp_in

    front_path = process_blob(front)
    back_path = process_blob(back)

    from PIL import Image, ImageDraw, ImageFont
    DPI = 300
    A4_W_IN, A4_H_IN = 8.27, 11.69
    a4_w = int(A4_W_IN*DPI)
    a4_h = int(A4_H_IN*DPI)
    margin = int(0.4*DPI)

    imgs = [Image.open(front_path).convert('RGB'), Image.open(back_path).convert('RGB')]
    layout = layout if layout in ('side','stack') else 'side'
    canvas = Image.new('RGB',(a4_w,a4_h),(255,255,255))
    draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype('arial.ttf',36)
    except: font=None

    if layout=='side':
        area_w = a4_w-2*margin; area_h=a4_h-2*margin
        half_w=int(area_w/2)
        positions=[(margin,margin),(margin+half_w,margin)]
        bounds=[(half_w,area_h),(half_w,area_h)]
    else:
        area_w=a4_w-2*margin; area_h=a4_h-2*margin
        half_h=int(area_h/2)
        positions=[(margin,margin),(margin,margin+half_h)]
        bounds=[(area_w,half_h),(area_w,half_h)]

    labels=['Front','Back']
    for i,im in enumerate(imgs):
        bw,bh=bounds[i]; w,h=im.size; scale=min(bw/w,bh/h)
        nw,nh=int(w*scale),int(h*scale)
        im_resized = im.resize((nw,nh), Image.LANCZOS)
        px = positions[i][0]+(bounds[i][0]-nw)//2
        py = positions[i][1]+(bounds[i][1]-nh)//2
        canvas.paste(im_resized,(px,py))
        tx,ty = px+10, py+10
        if font: draw.text((tx,ty),labels[i],fill=(0,0,0),font=font)
        else: draw.text((tx,ty),labels[i],fill=(0,0,0))

    out_pdf = os.path.join(app.config['OUTPUT_FOLDER'], f'id_A4.pdf')
    canvas.save(out_pdf,'PDF',resolution=DPI)
    cleanup_later(temp_to_cleanup+[out_pdf])
    return send_file(out_pdf, as_attachment=True)

#------------------ Utility: cleanup temp files -----------------------
def cleanup_later(paths, delay=3):
    import threading, time, os
    def _worker():
        time.sleep(delay)
        for p in paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    th=threading.Thread(target=_worker, daemon=True)
    th.start()

#------------------------ Main ------------------------
if __name__ == '__main__':
    # Read requested bind host and port from environment, with safe defaults
    # Safer bind-host handling: prefer explicit developer-friendly defaults
    # and avoid creating test sockets which can interfere with Werkzeug's
    # reloader on Windows (observed WinError 10038). We consider the host
    # valid only if it matches a local interface address, '0.0.0.0' or
    # 'localhost'. Otherwise fall back to 127.0.0.1 and log a warning.
    import socket

    # Default bind address and port
    bind_host_env = os.environ.get('BIND_HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 1000))

    # Gather likely local addresses
    local_addrs = {'127.0.0.1', '::1', 'localhost'}
    try:
        hn = socket.gethostname()
        for a in socket.gethostbyname_ex(hn)[2]:
            local_addrs.add(a)
    except Exception:
        logger.warning('Failed to resolve local host addresses; using defaults.')

    # Validate requested host
    if bind_host_env in ('0.0.0.0', '::', '') or bind_host_env in local_addrs:
        bind_host = bind_host_env
    else:
        logger.warning(
            'Requested bind host "%s" is not among local interfaces; falling back to 127.0.0.1',
            bind_host_env
        )
        bind_host = '127.0.0.1'

    logger.info(f'Starting server on http://{bind_host}:{port} (requested: {bind_host_env})')

    # Run Flask app with graceful error handling
    try:
        app.run(host=bind_host, port=port, debug=True)
    except OSError as e:
        logger.exception('Failed to bind server on %s:%d', bind_host, port)
        raise e
    except Exception as e:
        logger.exception('Unexpected error while running server')
        raise e

#------------------------ End of app.py ------------------------