(function(){
  const KEY = 'pdfpro_darkmode'

  function apply(isDark){
    try{
      if(isDark){
        document.documentElement.setAttribute('data-theme','dark')
        document.documentElement.classList.add('dark')
        document.body && document.body.classList.add('dark')
      } else {
        document.documentElement.removeAttribute('data-theme')
        document.documentElement.classList.remove('dark')
        document.body && document.body.classList.remove('dark')
      }
      // update toggle if present
      const toggleEl = document.getElementById('dark-mode-toggle')
      if(toggleEl){ toggleEl.setAttribute('aria-pressed', String(!!isDark)); toggleEl.textContent = isDark ? '☀️' : '🌙' }
      console.log('[darkmode] apply ->', isDark)
    }catch(e){
      console.warn('darkmode apply failed', e)
    }
  }
  // determine initial preference and apply
  function initialApply(){
    try{
      const saved = localStorage.getItem(KEY)
      if(saved === '1' || saved === '0'){
        apply(saved === '1')
        return
      }
    }catch(e){ /* ignore */ }
    const prefers = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
    apply(!!prefers)
  }

  // attach toggle listener (if present). If not present yet, wait for DOMContentLoaded then attach.
  function attachToggle(){
    const toggle = document.getElementById('dark-mode-toggle')
    if(!toggle) return false
    // ensure toggle shows current state
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark'
    toggle.setAttribute('aria-pressed', String(!!isDark))
    toggle.textContent = isDark ? '☀️' : '🌙'

    toggle.addEventListener('click', ()=>{
      const currentlyDark = document.documentElement.getAttribute('data-theme') === 'dark'
      const next = !currentlyDark
      apply(next)
      try{ localStorage.setItem(KEY, next ? '1' : '0') }catch(e){}
    })
    return true
  }

  // Listen to system preference changes only if there is no saved explicit choice
  function watchSystemPref(){
    try{
      const mq = window.matchMedia('(prefers-color-scheme: dark)')
      mq.addEventListener && mq.addEventListener('change', e => {
        try{
          const saved = localStorage.getItem(KEY)
          if(saved === '1' || saved === '0') return // user choice wins
        }catch(err){ }
        apply(e.matches)
      })
    }catch(e){ /* ignore */ }
  }

  // run
  initialApply()
  if(!attachToggle()){
    if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ()=>{ attachToggle() })
    else attachToggle()
  }
  watchSystemPref()

})();