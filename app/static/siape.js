/* JS mínimo de SIAPE: gráficas Chart.js con respaldo en PNG, ejemplos de PNL,
   validación de cadena y demostración de alteración. */
(function () {
  // ---- Gráficas: si Chart.js no cargó (sin internet) se muestra la imagen PNG de respaldo ----
  document.querySelectorAll('.grafica[data-chart]').forEach(function (caja) {
    var fallback = caja.getAttribute('data-fallback');
    if (typeof Chart === 'undefined') {
      if (fallback) {
        caja.classList.add('con-imagen');
        caja.innerHTML = '<img src="' + fallback + '" alt="Gráfica (imagen de respaldo)">';
      } else {
        caja.classList.add('sin-grafica');
        caja.textContent = 'La gráfica interactiva necesita conexión a internet (Chart.js). Los datos están en la tabla.';
      }
      return;
    }
    var canvas = document.createElement('canvas');
    caja.appendChild(canvas);
    try { new Chart(canvas, JSON.parse(caja.getAttribute('data-chart'))); }
    catch (e) { caja.textContent = 'No se pudo dibujar la gráfica.'; }
  });

  // ---- Botones de ejemplo en la página PNL ----
  document.querySelectorAll('[data-ejemplo]').forEach(function (b) {
    b.addEventListener('click', function () { document.getElementById('texto-pnl').value = b.getAttribute('data-ejemplo'); });
  });

  function esc(t) { return String(t).replace(/[&<>]/g, function (c) { return {'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c]; }); }

  // ---- Validar cadena ----
  var bv = document.getElementById('btn-validar');
  if (bv) bv.addEventListener('click', function () {
    var caja = document.getElementById('validar-res');
    bv.disabled = true;
    fetch(bv.dataset.url, {method: 'POST'}).then(function (r) { return r.json(); }).then(function (j) {
      caja.hidden = false;
      caja.className = 'resultado-js ' + (j.valida ? 'bien' : 'mal');
      caja.innerHTML = j.valida ? '<strong>✔ Cadena válida</strong> · ' + j.n_bloques + ' bloques revisados, sin alteraciones.'
        : '<strong>✖ Cadena INVÁLIDA</strong><ul>' + j.errores.map(function (e) { return '<li>' + esc(e) + '</li>'; }).join('') + '</ul>';
    }).finally(function () { bv.disabled = false; });
  });

  // ---- Demostración de alteración ----
  var bd = document.getElementById('btn-demo');
  if (bd) bd.addEventListener('click', function () {
    var caja = document.getElementById('demo-res');
    bd.disabled = true; bd.textContent = 'Ejecutando…';
    fetch(bd.dataset.url, {method: 'POST'}).then(function (r) { return r.json(); }).then(function (j) {
      caja.hidden = false;
      caja.className = 'resultado-js ' + (j.todo_correcto ? 'bien' : 'mal');
      caja.innerHTML = '<ol class="pasos">' + j.pasos.map(function (p) {
        return '<li class="' + (p.ok ? 'ok' : 'mal') + '"><strong>' + esc(p.titulo) + '</strong><br>' + esc(p.detalle) + '</li>';
      }).join('') + '</ol><p><strong>' + (j.todo_correcto ? 'Todas las alteraciones fueron detectadas.' : 'Revisar: alguna comprobación falló.') + '</strong></p>';
    }).finally(function () { bd.disabled = false; bd.textContent = 'Ejecutar demostración'; });
  });

  // ---- Generar reporte: aviso de espera ----
  var fg = document.getElementById('form-generar');
  if (fg) fg.addEventListener('submit', function () {
    var b = document.getElementById('btn-generar'); b.textContent = 'Generando…'; setTimeout(function () { b.disabled = true; }, 0);
  });
})();
