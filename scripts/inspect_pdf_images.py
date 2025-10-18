import os, sys, json
from pathlib import Path
p = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(p))

PDF_PATH = r'P:\Hasnat\study\CI\CI-Final.pdf'
OUT_DIR = r'P:\Hasnat\pdf\output\debug_images'
os.makedirs(OUT_DIR, exist_ok=True)

try:
    import fitz
    from PIL import Image
except Exception as e:
    print('missing deps:', e)
    raise

doc = fitz.open(PDF_PATH)
page_index = 0
page = doc.load_page(page_index)
d = page.get_text('dict')
page_w = page.rect.width
page_h = page.rect.height
print('page size (pts):', page_w, page_h)
images_info = []
# collect block-level images
for block in d.get('blocks', []):
    if block.get('type') == 1:
        bbox = block.get('bbox', [0,0,0,0])
        im = block.get('image') or {}
        xref = None
        if isinstance(im, dict):
            xref = im.get('xref')
        if not xref:
            xref = block.get('xref') or (im if isinstance(im, int) else None)
        if xref:
            images_info.append({'xref': xref, 'bbox': bbox, 'from_block': True})
# fallback to page.get_images
if not images_info:
    for img_meta in page.get_images(full=True) or []:
        xref = img_meta[0]
        images_info.append({'xref': xref, 'bbox': None, 'from_block': False})

print('found images:', len(images_info))
results = []
for idx, info in enumerate(images_info, start=1):
    xref = info['xref']
    bbox = info.get('bbox')
    img_dict = None
    img_bytes = None
    try:
        img_dict = doc.extract_image(xref)
    except Exception:
        img_dict = None
    if img_dict:
        img_bytes = img_dict.get('image')
        img_ext = img_dict.get('ext', 'bin')
    else:
        try:
            pix = fitz.Pixmap(doc, xref)
            img_bytes = pix.tobytes('png')
            img_ext = 'png'
        except Exception:
            img_bytes = None
            img_ext = 'bin'
    if not img_bytes:
        results.append({'idx': idx, 'xref': xref, 'kept': False, 'reason': 'no_bytes'})
        continue
    # analyze with PIL
    try:
        im = Image.open(io := __import__('io').BytesIO(img_bytes))
        w,h = im.size
        # small downsampled grayscale
        im_small = im.convert('L').resize((64,64))
        pixels = list(im_small.getdata())
        non_black = sum(1 for p in pixels if p > 10)
        pct_non_black = non_black / len(pixels) if pixels else 0.0
        page_area = page_w * page_h
        bbox_area = None
        if bbox and len(bbox) >= 4:
            bbox_w = max(0.0, float(bbox[2]) - float(bbox[0]))
            bbox_h = max(0.0, float(bbox[3]) - float(bbox[1]))
            bbox_area = bbox_w * bbox_h
        area_frac = (bbox_area / page_area) if (bbox_area and page_area) else 0.0
        reason = None
        kept = True
        if pct_non_black < 0.02 or (w*h < 64*64 and pct_non_black < 0.05) or (area_frac > 0.8 and pct_non_black < 0.05):
            kept = False
            reason = 'mostly_black_or_small_or_fullpage_mask'
        # write out sample image for manual inspection
        fn = os.path.join(OUT_DIR, f'img_{idx}_xref_{xref}.{img_ext}')
        with open(fn, 'wb') as fh:
            fh.write(img_bytes)
    except Exception as e:
        kept = True
        reason = 'analysis_failed'
        fn = None
    results.append({'idx': idx, 'xref': xref, 'bbox': bbox, 'size': (w,h) if 'w' in locals() else None, 'pct_non_black': round(pct_non_black,4) if 'pct_non_black' in locals() else None, 'area_frac': round(area_frac,3) if 'area_frac' in locals() else None, 'kept': kept, 'reason': reason, 'saved_file': fn})

print(json.dumps(results, indent=2))
print('wrote sample images to', OUT_DIR)
doc.close()
