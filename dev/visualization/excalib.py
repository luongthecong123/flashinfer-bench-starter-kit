"""Shared Excalidraw drawing primitives for all diagram generators.

Usage:
    from excalib import Diagram

    d = Diagram()
    d.rect(0, 0, 100, 50, color="#1e1e1e", bg="#dbe4ff")
    d.text(10, 10, "hello")
    d.write("output.excalidraw")
"""
import json, random, string

# ── Pixel dimension map (log-ish scale for readability) ──
# Dimension name → pixel size.  Consistent across ALL diagrams.
DIM = {
    1:       8,
    5:       60,     # T=5 batch
    16:      40,
    64:      80,
    128:     120,
    192:     160,
    289:     200,    # mean V (valid per token)
    512:     400,
    1024:    480,    # KV cache len (vanilla)
    2048:    550,
    8462:    180,    # pages
    541568:  160,    # flattened total — compact for gather previews
}

# 3D depth offset: how many px to shift right (dx) and up (dy) per depth unit
DEPTH_DX = 20
DEPTH_DY = 40

def px(dim):
    """Look up pixel size for a logical dimension. Falls back to log scale."""
    if dim in DIM:
        return DIM[dim]
    import math
    return int(40 * math.log2(max(dim, 2)))

# ── Colors ──
C_Q    = "#228be6"   # query
C_KV   = "#40c057"   # key / value
C_SCR  = "#be4bdb"   # scores / logits
C_ATTN = "#fd7e14"   # attention weights
C_OUT  = "#20c997"   # output
C_IDX  = "#e03131"   # indices / mask
C_DIM  = "#868e96"   # dimension labels
C_TXT  = "#1e1e1e"   # titles / body
C_OP   = "#e03131"   # operation arrows / labels

BG_Q    = "#dbe4ff"
BG_KV   = "#d3f9d8"
BG_SCR  = "#f3d9fa"
BG_ATTN = "#fff4e6"
BG_OUT  = "#c3fae8"
BG_IDX  = "#ffe3e3"
BG_MASK = "#fff3bf"

# ── Fill styles ──
FILL_IN  = "hachure"       # input tensors
FILL_OUT = "cross-hatch"   # output tensors

# ── Font sizes ──
FS_DIM   = 14
FS_NAME  = 16
FS_TITLE = 22
FS_NOTE  = 13
FS_BIG   = 28

LGAP = 12
SECTION_GAP = 100


class Diagram:
    def __init__(self):
        self.elements = []
        self._idx = 0

    def _uid(self):
        return ''.join(random.choices(string.ascii_letters + string.digits + '-_', k=21))

    def _next_index(self):
        i = self._idx
        self._idx += 1
        c = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        bucket = i // len(c)
        pos = i % len(c)
        prefix = chr(ord('a') + bucket) if bucket < 26 else chr(ord('A') + bucket - 26)
        return prefix + c[pos]

    # ── primitives ──

    def rect(self, x, y, w, h, color="#1e1e1e", bg="transparent", fill="solid", roundness=3):
        rn = {"type": roundness} if roundness else None
        self.elements.append({
            "id": self._uid(), "type": "rectangle",
            "x": x, "y": y, "width": w, "height": h,
            "angle": 0, "strokeColor": color, "backgroundColor": bg,
            "fillStyle": fill, "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
            "index": self._next_index(), "roundness": rn,
            "seed": random.randint(1, 2**31), "version": 1,
            "versionNonce": random.randint(1, 2**31),
            "isDeleted": False, "boundElements": None,
            "updated": 1773529200000, "link": None, "locked": False,
        })

    def text(self, x, y, txt, font_size=20, color="#1e1e1e"):
        self.elements.append({
            "id": self._uid(), "type": "text",
            "x": x, "y": y,
            "width": len(txt) * font_size * 0.6,
            "height": font_size * 1.25,
            "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
            "index": self._next_index(), "roundness": None,
            "seed": random.randint(1, 2**31), "version": 1,
            "versionNonce": random.randint(1, 2**31),
            "isDeleted": False, "boundElements": None,
            "updated": 1773529200000, "link": None, "locked": False,
            "text": txt, "fontSize": font_size, "fontFamily": 5,
            "textAlign": "left", "verticalAlign": "top",
            "containerId": None, "originalText": txt,
            "autoResize": True, "lineHeight": 1.25,
        })

    def line(self, x, y, points, color="#1e1e1e", style="solid"):
        """Free-form line from (x,y) through relative points."""
        # compute bounding box
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        self.elements.append({
            "id": self._uid(), "type": "line",
            "x": x, "y": y,
            "width": max(xs) - min(xs), "height": max(ys) - min(ys),
            "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": style,
            "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
            "index": self._next_index(), "roundness": {"type": 2},
            "seed": random.randint(1, 2**31), "version": 1,
            "versionNonce": random.randint(1, 2**31),
            "isDeleted": False, "boundElements": None,
            "updated": 1773529200000, "link": None, "locked": False,
            "points": points,
            "startBinding": None, "endBinding": None,
            "startArrowhead": None, "endArrowhead": None,
        })

    def polygon(self, x, y, points, color="#1e1e1e", bg="transparent", fill="solid"):
        """Closed polygon with fill. Points are relative to (x, y)."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        self.elements.append({
            "id": self._uid(), "type": "line",
            "x": x, "y": y,
            "width": max(xs) - min(xs), "height": max(ys) - min(ys),
            "angle": 0, "strokeColor": color, "backgroundColor": bg,
            "fillStyle": fill, "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
            "index": self._next_index(), "roundness": None,
            "seed": random.randint(1, 2**31), "version": 1,
            "versionNonce": random.randint(1, 2**31),
            "isDeleted": False, "boundElements": None,
            "updated": 1773529200000, "link": None, "locked": False,
            "points": points,
            "startBinding": None, "endBinding": None,
            "startArrowhead": None, "endArrowhead": None,
        })

    def rect_3d(self, x, y, w, h, depth_px, color="#1e1e1e", bg="transparent", fill="solid"):
        """Draw a 3D box: front face rectangle + 2 parallelogram faces.

        depth_px: how many pixels of depth (controls the 3D extrusion).
        The extrusion goes up-right: dx = depth_px, dy = depth_px * 2.
        Returns (front_x, front_y, front_w, front_h) of the front face.
        """
        dx = depth_px
        dy = depth_px * 2
        # front face — sharp corners for 3D
        self.rect(x, y, w, h, color, bg, fill, roundness=None)
        # top face parallelogram
        self.polygon(x, y,
                     [[0, 0], [dx, -dy], [w + dx, -dy], [w, 0], [0, 0]],
                     color, bg, fill)
        # right face parallelogram
        self.polygon(x + w, y,
                     [[0, 0], [dx, -dy], [dx, h - dy], [0, h], [0, 0]],
                     color, bg, fill)
        return x, y, w, h

    def labeled_rect_3d(self, x, y, w, h, depth_px, name, color, bg,
                        dim_top=None, dim_side=None, dim_depth=None, shape=None,
                        fill="solid"):
        """3D rect with name, dim labels, and optional shape annotation.

        dim_top: label above the front face (width dim)
        dim_side: label left of the front face (height dim)
        dim_depth: label on the depth axis (upper-left of depth)
        """
        self.rect_3d(x, y, w, h, depth_px, color, bg, fill)
        self.auto_name(x, y, w, h, name, color)
        if dim_top:
            self.dim_above(x, y, w, dim_top)
        if dim_side:
            self.dim_left(x, y, h, dim_side)
        if dim_depth:
            dx = depth_px
            dy = depth_px * 2
            tw = len(dim_depth) * FS_DIM * 0.6
            self.text(x - tw - 4, y - dy - FS_DIM * 0.6, dim_depth, FS_DIM, C_DIM)
        if shape:
            # Placed below the front face to avoid obstructing center text
            self.text(x, y + h + 6, shape, FS_NOTE, C_DIM)
        return y + h

    def arrow_h(self, x1, y, x2, color="#1e1e1e"):
        dx = x2 - x1
        self.elements.append({
            "id": self._uid(), "type": "arrow",
            "x": x1, "y": y, "width": abs(dx), "height": 0,
            "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
            "index": self._next_index(), "roundness": {"type": 2},
            "seed": random.randint(1, 2**31), "version": 1,
            "versionNonce": random.randint(1, 2**31),
            "isDeleted": False, "boundElements": None,
            "updated": 1773529200000, "link": None, "locked": False,
            "points": [[0, 0], [dx, 0]],
            "startBinding": None, "endBinding": None,
            "startArrowhead": None, "endArrowhead": "arrow",
            "elbowed": False,
        })

    def arrow_v(self, x, y1, y2, color="#1e1e1e"):
        dy = y2 - y1
        self.elements.append({
            "id": self._uid(), "type": "arrow",
            "x": x, "y": y1, "width": 0, "height": abs(dy),
            "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
            "index": self._next_index(), "roundness": {"type": 2},
            "seed": random.randint(1, 2**31), "version": 1,
            "versionNonce": random.randint(1, 2**31),
            "isDeleted": False, "boundElements": None,
            "updated": 1773529200000, "link": None, "locked": False,
            "points": [[0, 0], [0, dy]],
            "startBinding": None, "endBinding": None,
            "startArrowhead": None, "endArrowhead": "arrow",
            "elbowed": False,
        })

    # ── label helpers ──

    def dim_above(self, rx, ry, rw, label):
        tw = len(label) * FS_DIM * 0.6
        self.text(rx + rw/2 - tw/2, ry - FS_DIM*1.25 - 4, label, FS_DIM, C_DIM)

    def dim_left(self, rx, ry, rh, label):
        tw = len(label) * FS_DIM * 0.6
        self.text(rx - tw - 8, ry + rh/2 - FS_DIM*0.6, label, FS_DIM, C_DIM)

    def dim_right(self, rx, ry, rw, rh, label):
        self.text(rx + rw + 8, ry + rh/2 - FS_DIM*0.6, label, FS_DIM, C_DIM)

    def dim_below(self, rx, ry, rw, rh, label):
        tw = len(label) * FS_DIM * 0.6
        self.text(rx + rw/2 - tw/2, ry + rh + 4, label, FS_DIM, C_DIM)

    def name_below(self, rx, ry, rw, rh, label, color):
        tw = len(label) * FS_NAME * 0.6
        self.text(rx + rw/2 - tw/2, ry + rh + 6, label, FS_NAME, color)

    def name_inside(self, rx, ry, rw, rh, label, color):
        tw = len(label) * FS_NAME * 0.6
        th = FS_NAME * 1.25
        self.text(rx + rw/2 - tw/2, ry + rh/2 - th/2, label, FS_NAME, color)

    def name_right(self, rx, ry, rw, rh, label, color):
        self.text(rx + rw + 12, ry + rh/2 - FS_NAME*0.6, label, FS_NAME, color)

    def shape_right(self, rx, ry, rw, rh, label, color=C_DIM, fs=FS_NOTE):
        self.text(rx + rw + 12, ry + rh/2 - fs*0.6, label, fs, color)

    def auto_name(self, rx, ry, rw, rh, label, color):
        """Place name inside if rect is big enough, else below."""
        if rh > FS_NAME * 2.5 and rw > len(label)*FS_NAME*0.6+20:
            self.name_inside(rx, ry, rw, rh, label, color)
        elif rh > FS_NAME * 2.5:
            self.name_right(rx, ry, rw, rh, label, color)
        else:
            self.name_below(rx, ry, rw, rh, label, color)

    # ── composite helpers ──

    def labeled_rect(self, x, y, w, h, name, color, bg,
                     dim_top=None, dim_side=None, shape=None, fill="solid"):
        """Draw a rectangle with name, optional dim labels and shape annotation."""
        self.rect(x, y, w, h, color, bg, fill)
        self.auto_name(x, y, w, h, name, color)
        if dim_top:
            self.dim_above(x, y, w, dim_top)
        if dim_side:
            self.dim_left(x, y, h, dim_side)
        if shape:
            self.shape_right(x, y, w, h, shape)
        return y + h

    def transform_arrow(self, x1, y, x2, top_label, bot_label=None):
        """Horizontal arrow with labels above/below."""
        self.arrow_h(x1, y, x2, C_OP)
        self.text(x1 + 5, y - 22, top_label, FS_NOTE, C_OP)
        if bot_label:
            self.text(x1 + 5, y + 6, bot_label, FS_NOTE, C_OP)

    def op_text(self, x, y, label, fs=22):
        """Operation symbol like +, ×, =."""
        self.text(x, y, label, fs, C_OP)

    def matmul_L(self, ox, oy, a_name, a_color, a_bg, a_h, a_w,
                 b_name, b_color, b_bg, b_h, b_w,
                 c_name, c_color, c_bg,
                 row_dim, shared_dim, col_dim,
                 a_fill="solid", b_fill="solid", c_fill="solid"):
        """L-shaped matmul: A (left) × B (top) = C (intersection).

        Returns (bottom_y, c_x, c_y, c_w, c_h).
        """
        c_w, c_h = b_w, a_h
        c_x = ox + a_w + LGAP
        c_y = oy
        a_x, a_y = ox, oy
        b_x = c_x
        b_y = c_y - b_h - LGAP

        self.rect(a_x, a_y, a_w, a_h, a_color, a_bg, a_fill)
        self.rect(b_x, b_y, b_w, b_h, b_color, b_bg, b_fill)
        self.rect(c_x, c_y, c_w, c_h, c_color, c_bg, c_fill)

        # Names
        self.auto_name(a_x, a_y, a_w, a_h, a_name, a_color)

        if b_h > FS_NAME * 2.5 and b_w > len(b_name)*FS_NAME*0.6+20:
            self.name_inside(b_x, b_y, b_w, b_h, b_name, b_color)
        elif b_h > FS_NAME * 2.5:
            self.name_right(b_x, b_y, b_w, b_h, b_name, b_color)
        else:
            tw = len(b_name)*FS_NAME*0.6
            self.text(b_x + b_w/2 - tw/2, b_y - FS_NAME*1.25 - 20, b_name, FS_NAME, b_color)

        self.auto_name(c_x, c_y, c_w, c_h, c_name, c_color)

        # Dims
        self.dim_left(a_x, a_y, a_h, row_dim)
        self.dim_above(a_x, a_y, a_w, shared_dim)
        self.dim_above(b_x, b_y, b_w, col_dim)

        return max(a_y + a_h, c_y + c_h), c_x, c_y, c_w, c_h

    def bmm_L_3d(self, ox, oy, depth_px,
                 a_name, a_color, a_bg, a_h, a_w,
                 b_name, b_color, b_bg, b_h, b_w,
                 c_name, c_color, c_bg,
                 row_dim, shared_dim, col_dim, batch_dim=None,
                 a_fill="solid", b_fill="solid", c_fill="solid"):
        """L-shaped batched matmul with 3D depth on all three matrices.

        Returns (bottom_y, c_x, c_y, c_w, c_h).
        """
        dx = depth_px
        dy = depth_px * 2
        c_w, c_h = b_w, a_h
        c_x = ox + a_w + LGAP + dx          # extra gap for A's right face
        c_y = oy
        a_x, a_y = ox, oy
        b_x = c_x
        b_y = c_y - b_h - LGAP - dy         # extra gap for C's top face

        self.rect_3d(a_x, a_y, a_w, a_h, depth_px, a_color, a_bg, a_fill)
        self.rect_3d(b_x, b_y, b_w, b_h, depth_px, b_color, b_bg, b_fill)
        self.rect_3d(c_x, c_y, c_w, c_h, depth_px, c_color, c_bg, c_fill)

        # Names
        self.auto_name(a_x, a_y, a_w, a_h, a_name, a_color)
        if b_h > FS_NAME * 2.5 and b_w > len(b_name)*FS_NAME*0.6+20:
            self.name_inside(b_x, b_y, b_w, b_h, b_name, b_color)
        elif b_h > FS_NAME * 2.5:
            self.name_right(b_x, b_y, b_w, b_h, b_name, b_color)
        else:
            tw = len(b_name)*FS_NAME*0.6
            self.text(b_x + b_w/2 - tw/2, b_y - FS_NAME*1.25 - 20, b_name, FS_NAME, b_color)
        self.auto_name(c_x, c_y, c_w, c_h, c_name, c_color)

        # Dims
        self.dim_left(a_x, a_y, a_h, row_dim)
        self.dim_above(a_x, a_y, a_w, shared_dim)
        self.dim_above(b_x, b_y, b_w, col_dim)
        if batch_dim:
            tw = len(batch_dim) * FS_DIM * 0.6
            self.text(a_x - tw - 4, a_y - dy - FS_DIM * 0.6, batch_dim, FS_DIM, C_DIM)

        return max(a_y + a_h, c_y + c_h), c_x, c_y, c_w, c_h

    # ── output ──

    def write(self, path):
        doc = {
            "type": "excalidraw",
            "version": 2,
            "source": "https://excalidraw.com",
            "elements": self.elements,
            "appState": {
                "gridSize": 20, "gridStep": 5,
                "gridModeEnabled": False,
                "viewBackgroundColor": "#ffffff",
            },
            "files": {},
        }
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"Written {path}  ({len(self.elements)} elements)")
