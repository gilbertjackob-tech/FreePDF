(function(){
  // small client helper to create previews for split and merge pages
  // exposes window.pdfPreviews

  function el(tag, cls, html){ const d = document.createElement(tag); if(cls) d.className = cls; if(html) d.innerHTML = html; return d }

  function getCsrfHeaders(){
    const token = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content')
    return token ? { 'X-CSRFToken': token } : {}
  }

  // small toast helper
  function ensureToastContainer(){
    let c = document.querySelector('.pdfpro-toast-container')
    if(!c){ c = document.createElement('div'); c.className = 'pdfpro-toast-container'; document.body.appendChild(c) }
    return c
  }

  function showMessage(text, type='info', timeout=4000){
    const c = ensureToastContainer();
    const t = document.createElement('div'); t.className = 'pdfpro-toast '+type; t.textContent = text; c.appendChild(t);
    setTimeout(()=>{ t.style.opacity = '0'; setTimeout(()=>t.remove(),300) }, timeout);
    return t
  }

  function formatBytes(n){ if(n===0) return '0 B'; const units=['B','KB','MB','GB','TB']; const i=Math.floor(Math.log(n)/Math.log(1024)); return (n/Math.pow(1024,i)).toFixed(i?1:0)+' '+units[i] }

  async function safeFetch(url, options, { successMsg=null, errorMsg=null, disableEl=null }={}){
    try{
      if(disableEl){ disableEl.disabled = true; const spinner = document.createElement('span'); spinner.className='spinner spinner-sm'; disableEl.prepend(spinner); disableEl._spinner = spinner }
      const res = await fetch(url, options)
      if(disableEl && disableEl._spinner){ disableEl._spinner.remove(); delete disableEl._spinner; disableEl.disabled=false }
      if(!res.ok){ const errText = await res.text().catch(()=>res.statusText); if(errorMsg) showMessage(errorMsg+': '+errText, 'error'); else showMessage('Request failed: '+(errText||res.status), 'error'); throw new Error(errText||res.status) }
      if(successMsg) showMessage(successMsg, 'success')
      return res
    }catch(e){ if(disableEl && disableEl._spinner){ disableEl._spinner.remove(); delete disableEl._spinner; disableEl.disabled=false } throw e }
  }

  async function generateThumbsForFile(file){
    const fd = new FormData(); fd.append('pdf', file)
    const res = await fetch('/thumbnails', { method: 'POST', headers: getCsrfHeaders(), body: fd })
    if(!res.ok) throw new Error('thumbnail generation failed')
    const data = await res.json();
    return data
  }

  // size: optional 'small'|'medium'|'large' or pixel number
  async function generateForSplit(file, containerId, rangesInput, size='large'){
    const container = document.getElementById(containerId)
    container.innerHTML = ''
    const data = await generateThumbsForFile(file)
    const thumbs = data.thumbs || []
    thumbs.forEach((url, idx) => {
      const cls = 'card preview-thumb preview-thumb-' + (typeof size === 'number' ? 'custom' : size)
      const c = el('div', cls)
      c.dataset.page = idx+1
  const innerWidth = (typeof size === 'number') ? `${size}px` : ''
  // use data-src for lazy loading
  c.innerHTML = `<div style="position:relative"><div class="select-badge" aria-hidden="true">✓</div><div class="thumb" style="width:${innerWidth}"><img data-src="${url}" style="width:100%; display:block"></div></div><div class="p-2"><div class="fw-semibold">Page ${idx+1}</div><div class="mt-1 small text-muted">click to select</div></div>`
      c.addEventListener('click', ()=>{
        // toggle selection and update ranges input
        c.classList.toggle('selected')
        updateRanges(container, rangesInput)
      })
      container.appendChild(c)
    })
    // initialize lazy loading for images in this container
    initLazyLoad(container)
  }

  // render split thumbnails from an existing server response (data from /thumbnails)
  function generateForSplitFromData(data, containerId, rangesInput){
    const container = document.getElementById(containerId)
    container.innerHTML = ''
    const thumbs = (data && data.thumbs) || []
    thumbs.forEach((url, idx) => {
      const c = el('div','card preview-thumb preview-thumb-large')
      c.dataset.page = idx+1
  c.innerHTML = `<div style="position:relative"><div class="select-badge" aria-hidden="true">✓</div><div class="thumb"><img data-src="${url}" style="width:100%; display:block"></div></div><div class="p-2"><div class="fw-semibold">Page ${idx+1}</div><div class="mt-1 small text-muted">click to select</div></div>`
      c.addEventListener('click', ()=>{
        c.classList.toggle('selected')
        updateRanges(container, rangesInput)
      })
      container.appendChild(c)
    })
    initLazyLoad(container)
  }

  // parse a ranges string like "1,3-5" into a Set of page numbers
  function parseRangesString(rangesStr){
    const out = new Set()
    if(!rangesStr) return out
    rangesStr.split(',').map(s=>s.trim()).forEach(part=>{
      if(!part) return
      if(part.includes('-')){
        const [a,b] = part.split('-').map(x=>parseInt(x,10)).filter(n=>!Number.isNaN(n))
        if(a && b && b>=a){ for(let i=a;i<=b;i++) out.add(i) }
      }else{
        const n = parseInt(part,10); if(!Number.isNaN(n)) out.add(n)
      }
    })
    return out
  }

  // Render thumbnails from server data and auto-select pages present in rangesStr
  function generateForSplitFromDataSelected(data, containerId, rangesStr, size='large'){
    const container = document.getElementById(containerId)
    container.innerHTML = ''
    const thumbs = (data && data.thumbs) || []
    const selectedSet = parseRangesString(rangesStr)
    thumbs.forEach((url, idx) => {
      const page = idx+1
      const c = el('div','card preview-thumb preview-thumb-' + (typeof size === 'number' ? 'custom' : size))
      c.dataset.page = page
      const innerWidth = (typeof size === 'number') ? `${size}px` : ''
  c.innerHTML = `<div style="position:relative"><div class="select-badge" aria-hidden="true">✓</div><div class="thumb" style="width:${innerWidth}"><img data-src="${url}" style="width:100%; display:block"></div></div><div class="p-2"><div class="fw-semibold">Page ${page}</div><div class="mt-1 small text-muted">click to select</div></div>`
      if(selectedSet.has(page)) c.classList.add('selected')
      c.addEventListener('click', ()=>{
        c.classList.toggle('selected')
        // if there's an input we try to update it
        if(typeof rangesStr === 'object' && rangesStr && rangesStr._isInput){
          updateRanges(container, rangesStr)
        }
      })
      container.appendChild(c)
    })
      // initialize lazy loading and update input value if needed
      initLazyLoad(container)
      if(typeof rangesStr === 'object' && rangesStr && rangesStr._isInput){ updateRanges(container, rangesStr) }
  }

  function updateRanges(container, input){
    const pages = Array.from(container.querySelectorAll('.preview-thumb.selected')).map(n=>parseInt(n.dataset.page))
    if(pages.length === 0){ input.value = ''; return }
    // for simplicity create comma-separated pages and ranges
    pages.sort((a,b)=>a-b)
    const ranges = []
    let start = pages[0], last = pages[0]
    for(let i=1;i<pages.length;i++){
      const p = pages[i]
      if(p === last+1){ last = p; continue }
      if(start === last) ranges.push(''+start)
      else ranges.push(start+'-'+last)
      start = last = p
    }
    if(start === last) ranges.push(''+start)
    else ranges.push(start+'-'+last)
    input.value = ranges.join(',')
  }

  async function generateForMerge(files, containerId){
    const container = document.getElementById(containerId)
    container.innerHTML = ''
    // for each file, upload and request thumbs
    for(const f of files){
      const data = await generateThumbsForFile(f)
      const preview = el('div','card preview-file')
      preview.dataset.origname = f.name
      // server-side saved file id for merges
      if(data && data.file_id) preview.dataset.fileId = data.file_id
  preview.classList.add('preview-file-padding','grab-cursor')
      // lazy load the small preview image
      preview.innerHTML = `<div class="d-flex"><div style="width:110px;margin-right:12px"><img data-src="${data.thumbs[0]}" style="width:100%"></div><div class="flex-fill"><div class="fw-semibold text-truncate" style="max-width:160px">${f.name}</div><div class="small text-muted">${data.thumbs.length} pages</div></div></div>`
      // small remove button
      const rm = el('button','remove-item')
      rm.innerText = '×'
      rm.title = 'Remove'
      rm.addEventListener('click', ()=> {
        // revoke any blob URLs inside this preview to avoid leaking
        preview.querySelectorAll('img').forEach(img => { try{ if(img.src.startsWith('blob:')) URL.revokeObjectURL(img.src) }catch(e){} });
        preview.remove()
      })
      preview.appendChild(rm)
      container.appendChild(preview)
    }
    initLazyLoad(container)
    enableMergeDrag(container)
  }

  // Lazy load helper using IntersectionObserver. Loads images when they enter viewport
  function initLazyLoad(root){
    if(!('IntersectionObserver' in window)){
      // fallback: load all images immediately
      root.querySelectorAll('img[data-src]').forEach(img => { img.src = img.dataset.src })
      return
    }
    const imgs = Array.from(root.querySelectorAll('img[data-src]'))
    if(imgs.length === 0) return
    const obs = new IntersectionObserver((entries, o) => {
      entries.forEach(en => {
        if(en.isIntersecting){
          const img = en.target
          // set src and when loaded, revoke blob: URLs if present
          img.onload = () => { try{ if(img.src && img.src.startsWith('blob:')) URL.revokeObjectURL(img.src) }catch(e){} }
          img.onerror = () => { /* keep placeholder */ }
          img.src = img.dataset.src
          img.removeAttribute('data-src')
          o.unobserve(img)
        }
      })
    },{rootMargin:'200px'})
    imgs.forEach(i => obs.observe(i))
    // store observer so it can be disconnected if needed
    root._pdfPreviewObserver = obs
  }

  // Revoke any blob: URLs inside a container and disconnect observers
  function revokeBlobURLsInContainer(root){
    try{
      root.querySelectorAll('img').forEach(img => { try{ if(img.src && img.src.startsWith('blob:')) URL.revokeObjectURL(img.src) }catch(e){} })
      if(root._pdfPreviewObserver){ root._pdfPreviewObserver.disconnect(); delete root._pdfPreviewObserver }
    }catch(e){}
  }

  function enableMergeDrag(container){
    let dragged = null
    container.querySelectorAll('.preview-file').forEach(item=>{
      item.draggable = true
  item.addEventListener('dragstart', e=>{ dragged=item; item.classList.add('dragging') })
  item.addEventListener('dragend', e=>{ dragged=null; item.classList.remove('dragging') })
      item.addEventListener('dragover', e=> e.preventDefault())
      item.addEventListener('drop', e=>{
        e.preventDefault(); if(!dragged || dragged===item) return;
        const children = Array.from(container.children)
        const from = children.indexOf(dragged); const to = children.indexOf(item)
        if(from < to) container.insertBefore(dragged, item.nextSibling); else container.insertBefore(dragged, item)
      })
    })
  }

  // Select all thumbnails in a container and update ranges input
  function selectAllThumbnails(containerId, rangesInput){
    const container = document.getElementById(containerId)
    if(!container) return
    container.querySelectorAll('.preview-thumb').forEach(n=>n.classList.add('selected'))
    if(rangesInput) updateRanges(container, rangesInput)
  }

  // Clear all selections in a container and update ranges input
  function clearSelectionThumbnails(containerId, rangesInput){
    const container = document.getElementById(containerId)
    if(!container) return
    container.querySelectorAll('.preview-thumb.selected').forEach(n=>n.classList.remove('selected'))
    if(rangesInput) updateRanges(container, rangesInput)
  }

  window.pdfPreviews = {
    generateForSplit, generateForMerge, generateForSplitFromData, generateForSplitFromDataSelected,
    revokeBlobURLsInContainer, selectAllThumbnails, clearSelectionThumbnails
  }

  // small utilities exported globally
  window.pdfpro = { showMessage, formatBytes, safeFetch }

})();
