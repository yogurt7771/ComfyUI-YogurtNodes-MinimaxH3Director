// Stage overlay: repaint-mask painting and 局部精修 box drawing, in project canvas pixels.

export class Painter {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.mask = document.createElement("canvas");
        this.mode = "view";
        this.tool = "brush";
        this.size = 40;
        this.boxes = [];
        this.active = -1;
        this.onbox = null;
        this.dirty = false;
        this.drag = null;
        canvas.addEventListener("pointerdown", e => this.down(e));
        canvas.addEventListener("pointermove", e => this.move(e));
        canvas.addEventListener("pointerup", e => this.up(e));
        canvas.addEventListener("pointerleave", () => { this.hover = null; this.render(); });
    }

    resize(width, height) {
        if (this.canvas.width === width && this.canvas.height === height) return;
        this.canvas.width = this.mask.width = width;
        this.canvas.height = this.mask.height = height;
        this.render();
    }

    setMode(mode) {
        this.mode = mode;
        this.canvas.style.pointerEvents = mode === "view" ? "none" : "auto";
        this.canvas.style.cursor = mode === "box" ? "crosshair" : mode === "mask" ? "none" : "default";
        this.render();
    }

    clearMask() {
        this.mask.getContext("2d").clearRect(0, 0, this.mask.width, this.mask.height);
        this.dirty = true;
        this.render();
    }

    invertMask() {
        const ctx = this.mask.getContext("2d");
        const data = ctx.getImageData(0, 0, this.mask.width, this.mask.height);
        for (let i = 0; i < data.data.length; i += 4) {
            data.data[i] = data.data[i + 1] = data.data[i + 2] = 255;
            data.data[i + 3] = 255 - data.data[i + 3];
        }
        ctx.putImageData(data, 0, 0);
        this.dirty = true;
        this.render();
    }

    /** Load a black/white mask image (white = repaint) as the editable mask. */
    async loadMask(url) {
        this.clearMask();
        this.dirty = false;
        if (!url) return;
        const image = await new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = url;
        });
        const ctx = this.mask.getContext("2d");
        ctx.drawImage(image, 0, 0, this.mask.width, this.mask.height);
        const data = ctx.getImageData(0, 0, this.mask.width, this.mask.height);
        for (let i = 0; i < data.data.length; i += 4) {
            const value = data.data[i + 3] < 255 ? data.data[i + 3] : Math.max(data.data[i], data.data[i + 1], data.data[i + 2]);
            data.data[i] = data.data[i + 1] = data.data[i + 2] = 255;
            data.data[i + 3] = value;
        }
        ctx.putImageData(data, 0, 0);
        this.render();
    }

    hasMask() {
        const data = this.mask.getContext("2d").getImageData(0, 0, this.mask.width, this.mask.height).data;
        for (let i = 3; i < data.length; i += 16) if (data[i] > 8) return true;
        return false;
    }

    exportMask() {
        const out = document.createElement("canvas");
        out.width = this.mask.width;
        out.height = this.mask.height;
        const ctx = out.getContext("2d");
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, out.width, out.height);
        ctx.drawImage(this.mask, 0, 0);
        return new Promise(resolve => out.toBlob(resolve, "image/png"));
    }

    point(e) {
        const rect = this.canvas.getBoundingClientRect();
        return [(e.clientX - rect.left) * this.canvas.width / rect.width, (e.clientY - rect.top) * this.canvas.height / rect.height];
    }

    down(e) {
        if (this.mode === "view") return;
        this.canvas.setPointerCapture(e.pointerId);
        const p = this.point(e);
        this.drag = { start: p, last: p };
        if (this.mode === "mask" && this.tool !== "rect") this.stroke(p, p);
        this.render();
    }

    move(e) {
        const p = this.point(e);
        this.hover = p;
        if (this.drag && this.mode === "mask" && this.tool !== "rect") this.stroke(this.drag.last, p);
        if (this.drag) this.drag.last = p;
        this.render();
    }

    up(e) {
        if (!this.drag) return;
        const [x0, y0] = this.drag.start, [x1, y1] = this.point(e);
        const box = [Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)];
        if (this.mode === "mask" && this.tool === "rect" && box[2] > 2 && box[3] > 2) {
            const ctx = this.mask.getContext("2d");
            ctx.fillStyle = "#fff";
            ctx.fillRect(...box);
            this.dirty = true;
        }
        if (this.mode === "box" && box[2] > 6 && box[3] > 6) {
            const { width, height } = this.canvas;
            this.onbox?.([box[0] / width, box[1] / height, box[2] / width, box[3] / height]);
        }
        this.drag = null;
        this.render();
    }

    stroke(a, b) {
        const ctx = this.mask.getContext("2d");
        ctx.save();
        ctx.globalCompositeOperation = this.tool === "eraser" ? "destination-out" : "source-over";
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = this.size;
        ctx.lineCap = ctx.lineJoin = "round";
        ctx.beginPath();
        ctx.moveTo(...a);
        ctx.lineTo(...b);
        ctx.stroke();
        ctx.restore();
        this.dirty = true;
    }

    render() {
        const { ctx, canvas } = this;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (this.mode === "mask" || this.showMask) {
            ctx.save();
            ctx.globalAlpha = this.mode === "mask" ? 0.55 : 0.3;
            ctx.drawImage(this.tinted(), 0, 0);
            ctx.restore();
        }
        const line = Math.max(2, canvas.width / 400);
        this.boxes.forEach((box, i) => {
            if (!box) return;
            const [x, y, w, h] = [box[0] * canvas.width, box[1] * canvas.height, box[2] * canvas.width, box[3] * canvas.height];
            ctx.lineWidth = line;
            ctx.strokeStyle = i === this.active ? "#ffd257" : "#7aa7ffaa";
            ctx.setLineDash(i === this.active ? [] : [line * 3, line * 2]);
            ctx.strokeRect(x, y, w, h);
            ctx.setLineDash([]);
            ctx.fillStyle = ctx.strokeStyle;
            ctx.font = `${Math.round(canvas.height / 28)}px sans-serif`;
            ctx.fillText(`精修 ${i + 1}`, x + line * 2, y + canvas.height / 26);
        });
        if (this.drag && (this.mode === "box" || this.tool === "rect")) {
            const [x0, y0] = this.drag.start, [x1, y1] = this.drag.last;
            ctx.lineWidth = line;
            ctx.strokeStyle = "#ffd257";
            ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
        }
        if (this.mode === "mask" && this.hover && this.tool !== "rect") {
            ctx.beginPath();
            ctx.lineWidth = line;
            ctx.strokeStyle = this.tool === "eraser" ? "#7aa7ff" : "#ffd257";
            ctx.arc(...this.hover, this.size / 2, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    tinted() {
        if (!this.tint) this.tint = document.createElement("canvas");
        const tint = this.tint;
        tint.width = this.mask.width;
        tint.height = this.mask.height;
        const ctx = tint.getContext("2d");
        ctx.drawImage(this.mask, 0, 0);
        ctx.globalCompositeOperation = "source-in";
        ctx.fillStyle = "#ff8a2a";
        ctx.fillRect(0, 0, tint.width, tint.height);
        ctx.globalCompositeOperation = "source-over";
        return tint;
    }
}
