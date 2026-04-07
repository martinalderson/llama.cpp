// Canvas rendering for MoE expert routing visualisation.
// Called from Blazor via JS interop.

const canvases = {};

export function initCanvas(id, width, height) {
    const el = document.getElementById(id);
    if (!el) return;
    el.width = width;
    el.height = height;
    const ctx = el.getContext('2d');
    canvases[id] = { el, ctx, width, height };
}

export function renderRoutingFrame(id, nExpert, nLayers, frameJson) {
    const c = canvases[id];
    if (!c) return;
    const { ctx, width, height } = c;
    const frame = JSON.parse(frameJson);

    const cellW = width / nExpert;
    const cellH = height / nLayers;

    ctx.fillStyle = '#0e0e16';
    ctx.fillRect(0, 0, width, height);

    for (let li = 0; li < nLayers; li++) {
        const sel = frame.sel[li] || [];
        const top = frame.top[li] || [];

        // Build probability lookup
        const probMap = {};
        for (const [eid, p] of top) probMap[eid] = p;

        // Draw selected experts with probability-based brightness
        for (const eid of sel) {
            if (eid < 0) continue;
            const p = probMap[eid] || 0;
            const bright = 80 + 175 * Math.min(p * frame.neu, 1);
            const r = Math.floor(bright * 0.3);
            const g = Math.floor(bright * 0.85);
            const b = Math.floor(bright);
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(eid * cellW, li * cellH, cellW - 0.5, cellH - 0.5);
        }
    }
}

export function renderCumulative(id, nExpert, nLayers, countsFlat, maxCount) {
    const c = canvases[id];
    if (!c) return;
    const { ctx, width, height } = c;

    const cellW = width / nExpert;
    const cellH = height / nLayers;

    ctx.fillStyle = '#0e0e16';
    ctx.fillRect(0, 0, width, height);

    if (maxCount <= 0) return;

    for (let li = 0; li < nLayers; li++) {
        for (let ei = 0; ei < nExpert; ei++) {
            const v = countsFlat[li * nExpert + ei];
            if (v === 0) continue;

            const t = v / maxCount;
            let r, g, b;
            if (t < 0.5) {
                const s = t * 2;
                r = Math.round(10 + s * 20);
                g = Math.round(14 + s * 186);
                b = Math.round(22 + s * 230);
            } else {
                const s = (t - 0.5) * 2;
                r = Math.round(30 + s * 225);
                g = Math.round(200 + s * 55);
                b = Math.round(252 + s * 3);
            }
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(ei * cellW, li * cellH, cellW - 0.5, cellH - 0.5);
        }
    }
}

export function getCellFromMouse(id, nExpert, nLayers, offsetX, offsetY) {
    const c = canvases[id];
    if (!c) return null;
    const cellW = c.width / nExpert;
    const cellH = c.height / nLayers;
    const ei = Math.floor(offsetX / cellW);
    const li = Math.floor(offsetY / cellH);
    if (ei < 0 || ei >= nExpert || li < 0 || li >= nLayers) return null;
    return { layer: li, expert: ei };
}
