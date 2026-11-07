// node check.mjs [pureblas.wasm] [reference.json]: run each case of reference.json (wasm/reference.jl, native Julia
// with LinearAlgebra on PureBLAS) with the module and compare: normwise relative difference at most 1e-9. Vectors
// (eigenvectors, singular vectors) are compared by magnitude: each is unique only up to a sign or a phase.
// Needs julia.mjs (WasmCodegen's JavaScript host, in the release) next to this file.
import fs from 'node:fs';
import { instantiate } from './julia.mjs';

const dir = new URL('.', import.meta.url).pathname;
const [wasm = dir + 'pureblas.wasm', ref = dir + 'reference.json'] = process.argv.slice(2);
const t0 = performance.now();
const { instance } = await instantiate(fs.readFileSync(wasm));
const e = instance.exports;
console.log(`instantiated ${wasm} in ${(performance.now() - t0).toFixed(0)} ms`);

// the output's parts whose sign/phase is free, as [start, length] ranges (in Float64s)
function vectorParts(op, dims, len) {
  const c = op.endsWith('c64') ? 2 : 1;
  if (op.startsWith('eigen_hermitian')) { const n = dims[0]; return [[n, len - n]]; }
  if (op.startsWith('eigen')) { const n = dims[0]; return [[2 * n, len - 2 * n]]; }
  if (op.startsWith('svd_')) { const k = Math.min(dims[0], dims[1]); return [[k, len - k]]; }
  return [];
}
function magnitudes(a, start, length, complex) {
  const r = a.slice();
  for (let i = start; i < start + length; i += complex ? 2 : 1) {
    if (complex) { r[i] = Math.hypot(a[i], a[i + 1]); r[i + 1] = 0; } else r[i] = Math.abs(a[i]);
  }
  return r;
}

let bad = 0;
const cases = JSON.parse(fs.readFileSync(ref, 'utf8'));
for (const { op, dims, status, input, output } of cases) {
  e.input_resize(BigInt(input.length));
  input.forEach((x, i) => e.input_set(BigInt(i + 1), x));
  const t = performance.now();
  const st = Number(e[op](...dims.map(BigInt)));
  const ms = performance.now() - t;
  let got = Array.from({ length: Number(e.output_length()) }, (_, i) => e.output_get(BigInt(i + 1)));
  let want = output;
  for (const [s, l] of vectorParts(op, dims, want.length)) {
    got = magnitudes(got, s, l, op.endsWith('c64')); want = magnitudes(want, s, l, op.endsWith('c64'));
  }
  let diff = 0, scale = 0;
  for (let i = 0; i < want.length; i++) { diff = Math.max(diff, Math.abs(got[i] - want[i])); scale = Math.max(scale, Math.abs(want[i])); }
  const rel = got.length === want.length ? diff / (scale || 1) : Infinity;
  const ok = st === status && rel <= 1e-9;
  if (!ok) bad++;
  console.log(`${ok ? 'ok ' : 'BAD'} ${op}(${dims.join(', ')}): status ${st}, relative difference ${rel.toExponential(2)}, ${ms.toFixed(1)} ms`);
}
console.log(`${cases.length} cases: ${bad ? bad + ' differ' : 'all match the native reference'}`);
process.exit(bad ? 1 : 0);
