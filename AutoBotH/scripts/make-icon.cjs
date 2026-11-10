// Update log
// - [2026-11-05] type=feat (AutoBotH icon generator, same method as NoteH/MDViewerH: teal rounded-square gradient with three rising white
//   candlesticks (replaced by a line chart on 2026-11-10), drawn directly with supersampling, no external packages, producing build/icon.png (256), build/icon.ico (16-256,
//   PNG-compressed entries) and resources/icon.png (window icon))
// - [2026-11-10] type=feat (Design changed to a line chart: white zig-zag rising polyline with round joins and an end dot over a faint
//   baseline/axis on the same teal background; also writes src/renderer/public/icon.png, used left of "AutoBotH" in the title bar
//   and in the about dialog)
//
// Run: node scripts/make-icon.cjs  (only rerun when changing the icon design - the output is stored in build/ and resources/)

'use strict';

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const outDir = path.resolve(__dirname, '..', 'build');
const resourcesDir = path.resolve(__dirname, '..', 'resources');
const publicDir = path.resolve(__dirname, '..', 'src', 'renderer', 'public');

function inRoundRect(x, y, x0, y0, x1, y1, r) {
  if (x < x0 || x > x1 || y < y0 || y > y1) return false;
  const cx = Math.min(Math.max(x, x0 + r), x1 - r);
  const cy = Math.min(Math.max(y, y0 + r), y1 - r);
  return (x - cx) ** 2 + (y - cy) ** 2 <= r * r;
}

// Background gradient (top-left #14b8a6 -> bottom-right #0f5f73)
function bgColor(x, y) {
  const t = Math.min(1, Math.max(0, (x + y) / 512));
  return [0x14 + (0x0f - 0x14) * t, 0xb8 + (0x5f - 0xb8) * t, 0xa6 + (0x73 - 0xa6) * t];
}

// Line chart in a 256-based coordinate system: rising zig-zag polyline, end dot, faint axis (left + bottom)
const LINE = [[50, 178], [90, 136], [122, 158], [164, 100], [206, 66]];
const LINE_HALF = 9; // half stroke width
const DOT_R = 15;
const AXIS = [[[46, 52], [46, 206]], [[46, 206], [212, 206]]];
const AXIS_HALF = 4;
const AXIS_ALPHA = 0.45;

function distSeg(x, y, [ax, ay], [bx, by]) {
  const dx = bx - ax, dy = by - ay;
  const t = Math.min(1, Math.max(0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(x - (ax + t * dx), y - (ay + t * dy));
}

function pixel(x, y) {
  if (!inRoundRect(x, y, 8, 8, 248, 248, 52)) return null;
  const [ex, ey] = LINE[LINE.length - 1];
  if (Math.hypot(x - ex, y - ey) <= DOT_R) return [255, 255, 255];
  for (let i = 1; i < LINE.length; i++) {
    if (distSeg(x, y, LINE[i - 1], LINE[i]) <= LINE_HALF) return [255, 255, 255];
  }
  const bg = bgColor(x, y);
  if (AXIS.some(([a, b]) => distSeg(x, y, a, b) <= AXIS_HALF)) {
    return bg.map((v) => v + (255 - v) * AXIS_ALPHA);
  }
  return bg;
}

function render(size) {
  const SS = 4;
  const scale = 256 / size;
  const rgba = Buffer.alloc(size * size * 4);
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      let r = 0, g = 0, b = 0, a = 0;
      for (let sy = 0; sy < SS; sy++) {
        for (let sx = 0; sx < SS; sx++) {
          const c = pixel((px + (sx + 0.5) / SS) * scale, (py + (sy + 0.5) / SS) * scale);
          if (!c) continue;
          r += c[0]; g += c[1]; b += c[2]; a += 1;
        }
      }
      const i = (py * size + px) * 4;
      if (a) {
        rgba[i] = Math.round(r / a);
        rgba[i + 1] = Math.round(g / a);
        rgba[i + 2] = Math.round(b / a);
      }
      rgba[i + 3] = Math.round((a / (SS * SS)) * 255);
    }
  }
  return rgba;
}
const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function crc32(buf) {
  let c = 0xffffffff;
  for (const byte of buf) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
function toPng(size, rgba) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  const raw = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) rgba.copy(raw, y * (size * 4 + 1) + 1, y * size * 4, (y + 1) * size * 4);
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

const sizes = [16, 24, 32, 48, 64, 128, 256];
const pngs = sizes.map((s) => toPng(s, render(s)));
const header = Buffer.alloc(6);
header.writeUInt16LE(0, 0);
header.writeUInt16LE(1, 2);
header.writeUInt16LE(sizes.length, 4);
const dir = Buffer.alloc(16 * sizes.length);
let offset = 6 + dir.length;
sizes.forEach((s, i) => {
  const e = i * 16;
  dir[e] = s >= 256 ? 0 : s;
  dir[e + 1] = s >= 256 ? 0 : s;
  dir.writeUInt16LE(1, e + 4);
  dir.writeUInt16LE(32, e + 6);
  dir.writeUInt32LE(pngs[i].length, e + 8);
  dir.writeUInt32LE(offset, e + 12);
  offset += pngs[i].length;
});
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, 'icon.ico'), Buffer.concat([header, dir, ...pngs]));
fs.writeFileSync(path.join(outDir, 'icon.png'), pngs[sizes.length - 1]);
fs.mkdirSync(resourcesDir, { recursive: true });
fs.writeFileSync(path.join(resourcesDir, 'icon.png'), pngs[sizes.length - 1]);
fs.writeFileSync(path.join(publicDir, 'icon.png'), pngs[sizes.length - 1]);
console.log('[icon] build/icon.ico, build/icon.png, resources/icon.png, src/renderer/public/icon.png 생성');
