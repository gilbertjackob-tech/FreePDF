(function(){
  // Respect reduced motion
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  if(reduce) return

  // page fade in
  document.documentElement.style.opacity = 0
  window.addEventListener('load', ()=>{
    document.documentElement.style.transition = 'opacity 320ms cubic-bezier(.2,.9,.3,1)'
    document.documentElement.style.opacity = 1
  })

  // Add ripple class to common interactive elements
  function attachRipples(){
    const buttons = document.querySelectorAll('button, .btn, .theme-toggle')
    buttons.forEach(b => b.classList.add('ripple'))
  }
  attachRipples()

  // soft entrance animation for cards on the page
  function revealCards(){
    const cards = document.querySelectorAll('.card')
    cards.forEach((c,i)=>{
      c.style.opacity = 0
      c.style.transform = 'translateY(8px)'
      setTimeout(()=>{
        c.style.transition = 'opacity 420ms cubic-bezier(.2,.9,.3,1), transform 420ms cubic-bezier(.2,.9,.3,1)'
        c.style.opacity = 1
        c.style.transform = 'translateY(0)'
      }, 60 * i)
    })
  }
  if(document.readyState === 'complete') revealCards()
  else window.addEventListener('load', revealCards)

})();

/* Bootstrap-enhanced animations */
(function(){
  // fade-in using Bootstrap's .fade/.show classes for an extra-level polish
  function bsFadeIn(){
    const cards = document.querySelectorAll('.card')
    cards.forEach((c,i)=>{
      c.classList.add('fade')
      // stagger and then add show which Bootstrap uses for opacity transition
      setTimeout(()=> c.classList.add('show'), 90 * i)
    })

    const brand = document.querySelector('.navbar-brand')
    if(brand){ brand.classList.add('fade'); setTimeout(()=>brand.classList.add('show'), 80) }
  }

  // toast helper using Bootstrap Toast component
  function showToast(message, opts){
    try{
      const root = document.getElementById('toast-root')
      if(!root) return
      const wrapper = document.createElement('div')
      wrapper.innerHTML = `\n        <div class="toast align-items-center text-bg-dark border-0" role="alert" aria-live="assertive" aria-atomic="true">\n          <div class="d-flex">\n            <div class="toast-body">${message}</div>\n            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>\n          </div>\n        </div>\n      `
      const toastEl = wrapper.firstElementChild
      root.appendChild(toastEl)
      // initialize bootstrap toast
      const bsToast = new bootstrap.Toast(toastEl, { autohide: true, delay: (opts && opts.delay) || 2500 })
      bsToast.show()
      // remove from DOM after hidden
      toastEl.addEventListener('hidden.bs.toast', ()=> toastEl.remove())
      return bsToast
    }catch(e){ console.warn('showToast failed', e) }
  }

  if(document.readyState === 'complete') bsFadeIn()
  else window.addEventListener('load', bsFadeIn)

  // expose helper globally for quick testing in console
  window.pdfpro = window.pdfpro || {}
  window.pdfpro.showToast = showToast
})();