// Headless logic test for snake.html: stubs the DOM/canvas and drives the game loop.
const fs = require('fs');
const path = require('path');

const file = path.join(__dirname, 'snake.html');
const html = fs.readFileSync(file, 'utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

// ---- stubs -------------------------------------------------------------
const ctxStub = new Proxy({}, { get: (t, k) => (k in t ? t[k] : () => {}) });
const listeners = {};
const els = {
  c: { width: 480, height: 480, getContext: () => ctxStub, addEventListener: () => {} },
  score: { textContent: '0' },
  best: { textContent: '0' },
  msg: { textContent: '', innerHTML: '' },
};
const store = {};
let rafCb = null;

global.document = { getElementById: id => els[id] };
global.localStorage = {
  getItem: k => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); },
};
global.performance = { now: () => 0 };
global.requestAnimationFrame = cb => { rafCb = cb; };
global.window = {
  addEventListener: (type, fn) => { listeners[type] = fn; },
};

// ---- run the game ------------------------------------------------------
eval(code);

const key = k => listeners.keydown({ key: k, preventDefault() {} });
const tick = (ms, stepMs = 16) => {
  for (let t = 0; t < ms; t += stepMs) { const cb = rafCb; rafCb = null; cb(t + stepMs); }
};

const results = [];
const check = (name, cond, extra = '') =>
  results.push(`${cond ? 'PASS' : 'FAIL'}  ${name}${extra ? '  -> ' + extra : ''}`);

// 1. initial state
check('starts alive with score 0', String(els.score.textContent) === '0' && els.msg.innerHTML.includes('WASD'));

// 2. snake moves right by default: after ~1s it should still be alive (no wall hit yet)
tick(500);
check('survives 0.5s moving right', els.msg.innerHTML.includes('WASD'), els.msg.textContent);

// 3. steering up then into the top wall kills it
key('ArrowUp');
tick(2000);
check('dies on hitting the wall', els.msg.textContent.includes('Game over'), els.msg.textContent);

// 4. R restarts
key('r');
tick(50);
check('R restarts the game', els.msg.innerHTML.includes('WASD') && String(els.score.textContent) === '0');

// 5. no 180-degree turn: going right, pressing left must not kill it
key('ArrowLeft');
tick(300);
check('ignores 180-degree turn', els.msg.innerHTML.includes('WASD'), els.msg.textContent);

// 6. P pauses: no movement, no death while paused
key('p');
const before = els.msg.innerHTML;
tick(3000);
check('P pauses the game', els.msg.innerHTML === before && els.msg.innerHTML.includes('WASD'));
key('p');

// 7. eating food raises the score (drive it onto the food by brute force)
let ate = false;
for (let i = 0; i < 4000 && !ate; i++) {
  // random legal-ish steering, restarting if we die
  if (els.msg.textContent.includes('Game over')) { key('r'); }
  key(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'][Math.floor(Math.random() * 4)]);
  tick(140);
  if (Number(els.score.textContent) > 0) ate = true;
}
check('eating food increases the score', ate, 'score=' + els.score.textContent);

// 8. best score is persisted to localStorage
check('best score persisted', Number(store.snakeBest) > 0, 'snakeBest=' + store.snakeBest);

console.log(results.join('\n'));
console.log('\n' + results.filter(r => r.startsWith('PASS')).length + '/' + results.length + ' checks passed');
