// Minimal scanner helper used by templates/camscanner/scan.html
// Exposes scannerInit(videoEl, canvasEl, callbacks)
(function(){
  function initScanner(videoEl, canvasEl, opts){
    if(!videoEl || !canvasEl) return;
    const callbacks = opts || {};
    let stream = null;
    let facingMode = 'environment';
    const gallery = [];

    async function start(){
      try{
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode }, audio:false });
        videoEl.srcObject = stream;
        await videoEl.play();
        callbacks.onStatus && callbacks.onStatus('Camera ready');
      }catch(err){
        console.error('camera error', err);
        callbacks.onStatus && callbacks.onStatus('Camera error');
      }
    }

    function stop(){
      if(stream){ stream.getTracks().forEach(t=>t.stop()); stream = null; }
      videoEl.pause(); videoEl.srcObject = null;
      callbacks.onStatus && callbacks.onStatus('Camera stopped');
    }

    function flip(){ facingMode = (facingMode === 'environment') ? 'user' : 'environment'; stop(); start(); }

    function capture(){
      const w = videoEl.videoWidth, h = videoEl.videoHeight;
      canvasEl.width = w; canvasEl.height = h;
      const ctx = canvasEl.getContext('2d');
      ctx.drawImage(videoEl, 0, 0, w, h);
      return new Promise(resolve => canvasEl.toBlob(blob => {
        const url = URL.createObjectURL(blob);
        gallery.push({ blob, url });
        callbacks.onGalleryUpdate && callbacks.onGalleryUpdate(gallery.slice());
        resolve({ blob, url });
      }, 'image/jpeg', 0.9));
    }

    async function exportPdf(){
      if(!gallery.length) return;
      const fd = new FormData();
      gallery.forEach((it,idx)=> fd.append('images', it.blob, `scan_${idx+1}.jpg`));
      callbacks.onStatus && callbacks.onStatus('Uploading...');
      try{
        const res = await fetch('/images-to-pdf', { method: 'POST', body: fd });
        if(!res.ok){ callbacks.onStatus && callbacks.onStatus('Upload failed'); alert('Export failed'); return; }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url; a.download = 'scans.pdf'; a.click();
        callbacks.onStatus && callbacks.onStatus('Saved');
      }catch(err){ console.error(err); callbacks.onStatus && callbacks.onStatus('Network error'); }
    }

    function clearSession(){
      gallery.forEach(it=>{ try{ URL.revokeObjectURL(it.url) }catch(e){} });
      gallery.length = 0; callbacks.onGalleryUpdate && callbacks.onGalleryUpdate([]);
    }

    // attach to buttons if present
    document.getElementById('shutter_btn')?.addEventListener('click', async ()=>{ await capture(); });
    document.getElementById('flip_btn')?.addEventListener('click', ()=> flip());
    document.getElementById('done_btn')?.addEventListener('click', ()=> exportPdf());
    document.getElementById('clear_session')?.addEventListener('click', ()=> clearSession());
    document.getElementById('export_pdf')?.addEventListener('click', ()=> exportPdf());

    // expose small API
    return { start, stop, capture, exportPdf, clearSession, getGallery:()=>gallery };
  }

  // attach to window
  window.scannerInit = function(videoEl, canvasEl, callbacks){
    const inst = initScanner(videoEl, canvasEl, callbacks || {});
    // auto start
    inst.start();
    // connect gallery updates to UI
    const galleryRoot = document.getElementById('scan_gallery');
    if(galleryRoot){
      setInterval(()=>{
        const g = inst.getGallery();
        galleryRoot.innerHTML = '';
        g.forEach((it, idx)=>{
          const d = document.createElement('div');
          d.className = 'd-flex align-items-center gap-2 scanner-thumb';
          d.innerHTML = `<img src="${it.url}" class="scanner-thumb-img"/><div class="flex-fill small">Page ${idx+1}</div>`;
          galleryRoot.appendChild(d);
        });
      }, 600);
    }
    return inst;
  };
})();
