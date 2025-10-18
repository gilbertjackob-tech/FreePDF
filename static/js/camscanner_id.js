// Minimal ID capture flow helper used by templates/camscanner/id.html
// Exposes idScannerInit(videoEl, callbacks)
(function(){
  function init(videoEl, callbacks){
    let stream = null;
    let facingMode = 'environment';

    async function start(){
      try{
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode }, audio:false });
        videoEl.srcObject = stream; await videoEl.play();
        callbacks.onStatus && callbacks.onStatus('Camera ready');
      }catch(err){ console.error(err); callbacks.onStatus && callbacks.onStatus('Camera error'); }
    }

    function stop(){ if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null;} videoEl.pause(); videoEl.srcObject=null; callbacks.onStatus && callbacks.onStatus('Camera stopped'); }
    function flip(){ facingMode = (facingMode === 'environment') ? 'user' : 'environment'; stop(); start(); }

    async function capture(){
      const canvas = document.createElement('canvas');
      canvas.width = videoEl.videoWidth; canvas.height = videoEl.videoHeight;
      const ctx = canvas.getContext('2d'); ctx.drawImage(videoEl,0,0,canvas.width, canvas.height);
      return new Promise(resolve => canvas.toBlob(blob => {
        const url = URL.createObjectURL(blob);
        callbacks.onCapture && callbacks.onCapture(url, blob);
        resolve({ url, blob });
      }, 'image/jpeg', 0.92));
    }

    // wire default buttons if they exist
    document.getElementById('id_capture_btn')?.addEventListener('click', async ()=>{ await capture(); });
    document.getElementById('id_flip')?.addEventListener('click', () => flip());
    document.getElementById('id_finish')?.addEventListener('click', ()=>{
      // optional: upload captured images to server
      callbacks.onFinish && callbacks.onFinish();
    });
    document.getElementById('id_retake')?.addEventListener('click', ()=>{ document.getElementById('id_captures').innerHTML = ''; });

    start();
    return { start, stop, flip, capture };
  }

  window.idScannerInit = function(videoEl, callbacks){ return init(videoEl, callbacks || {}); };
})();
