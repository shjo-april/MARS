"""
Hero GIF (CPU only): compose resources/hero.gif from panels of the paper figures.

Five scenes: the biased problem (paper Fig. 1), separating biased and target objects with USS features (Fig. 2),
selecting the debiased centroids by the distance to backgrounds (Fig. 3), from the conventional WSSS label to the
debiased label (Fig. 7), and the results (supplementary Fig. G, Tables 3 and 5, Fig. 9). Every image is a panel
of those figures (panels/), every label and number is in scenes.json, so no model and no dataset are needed.

    python hero_gif/build_gif.py                  # writes resources/hero.gif
    python hero_gif/build_gif.py --preview DIR    # writes every frame to DIR as a PNG instead

The committed GIF was rendered on Windows with Segoe UI. On other systems the script falls back to the
DejaVu Sans bundled with matplotlib: same layout, slightly different glyphs.
"""
import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(ROOT, 'resources', 'hero.gif'))
ap.add_argument('--preview', metavar='DIR', help='write the frames as PNGs to DIR and skip the GIF')
args = ap.parse_args()

W, H = 1280, 960
TEAL, TEAL_BG, MINT_LINE = (0, 102, 102), (232, 245, 244), (150, 205, 200)
INK, MUTED, LINE, WHITE, GREY = (16, 24, 31), (91, 105, 117), (220, 229, 233), (255, 255, 255), (178, 188, 196)
TAB_DIM, SOFT = (191, 217, 217), (215, 238, 236)                   # inactive scene tab · small text on teal
BASE = (0, 110, 175)                                               # conventional WSSS / RS+EPM
PINK, ORANGE, BLUE = (232, 30, 99), (255, 152, 0), (0, 0, 248)     # the two clusters and the backgrounds in the paper figures
CREAM = (224, 224, 192)                                            # biased class (-1) in the debiased labels
M, TOP, NOTE_Y = 24, 214, 938                                      # side margin · top of the panels · baseline note
V = json.load(open(os.path.join(HERE, 'scenes.json'), encoding='utf-8'))

WIN_FONTS = 'C:/Windows/Fonts'
# DejaVu is wider than Segoe UI, so the fallback takes a smaller size (second value) and the layout still fits
FALLBACK = {'segoeui.ttf': ('DejaVuSans.ttf', 0.84), 'seguisb.ttf': ('DejaVuSans-Bold.ttf', 0.77), 'segoeuib.ttf': ('DejaVuSans-Bold.ttf', 0.80)}


def F(name, size):
    path = os.path.join(WIN_FONTS, name)
    if not os.path.exists(path):
        import matplotlib
        path, size = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', FALLBACK[name][0]), int(size * FALLBACK[name][1])
    return ImageFont.truetype(path, size)


F_TITLE, F_BADGE, F_SUBT, F_TAB = F('segoeuib.ttf', 40), F('seguisb.ttf', 16), F('segoeui.ttf', 17), F('segoeuib.ttf', 16)
F_HEAD, F_H, F_HS, F_BODY = F('segoeuib.ttf', 22), F('segoeuib.ttf', 20), F('segoeui.ttf', 16), F('segoeui.ttf', 17)
F_HB, F_MSG, F_NOTE, F_TICK = F('segoeuib.ttf', 17), F('segoeuib.ttf', 20), F('segoeui.ttf', 15), F('segoeui.ttf', 14)
F_BIG, F_TAG, F_ORG, F_PILL = F('segoeuib.ttf', 34), F('seguisb.ttf', 14), F('seguisb.ttf', 19), F('segoeuib.ttf', 16)


def logo(name, height):
    im = Image.open(os.path.join(HERE, 'logos', name)).convert('RGBA')
    return im.resize((round(im.width * height / im.height), height), Image.LANCZOS)


# affiliations in the order of the paper: OGQ and Samsung Electronics on the left, Sungkyunkwan University on the right
LOGO_OGQ, LOGO_SAMSUNG, LOGO_SKKU = logo('ogq.png', 40), logo('samsung.png', 22), logo('skku.png', 46)
_cache = {}


def panel(rel, size):
    """A figure panel from panels/, scaled to `size` (w, h)."""
    if (rel, size) not in _cache:
        _cache[rel, size] = Image.open(os.path.join(HERE, 'panels', rel)).convert('RGB').resize(size, Image.LANCZOS)
    return _cache[rel, size]


def rrect(dr, box, r, fill=None, outline=None, width=1):
    dr.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def wrap(dr, text, font, width):
    lines, cur = [], ''
    for word in text.split():
        t = (cur + ' ' + word).strip()
        if cur and dr.textlength(t, font=font) > width:
            lines.append(cur); cur = word
        else:
            cur = t
    return lines + [cur]


def para(dr, x, y, text, font, fill, width, step):
    """Wrapped text; returns the y after the last line."""
    for t in wrap(dr, text, font, width):
        dr.text((x, y), t, font=font, fill=fill, anchor='lm'); y += step
    return y


def line(dr, y, text, font, fill):
    """One centered line; the build stops if it does not fit between the side margins."""
    assert dr.textlength(text, font=font) <= W - 2 * M, text
    dr.text((W / 2, y), text, font=font, fill=fill, anchor='mm')


def fit(dr, text, font, width):
    assert dr.textlength(text, font=font) <= width, text
    return text


def canvas(scene, head, note):
    """White frame with the header: logos, title and venue, subtitle, scene tabs, the scene headline and its note."""
    im = Image.new('RGB', (W, H), WHITE); dr = ImageDraw.Draw(im)
    im.paste(LOGO_OGQ, (34, 40 - LOGO_OGQ.height // 2), LOGO_OGQ)
    x = 34 + LOGO_OGQ.width + 30
    im.paste(LOGO_SAMSUNG, (x, 40 - LOGO_SAMSUNG.height // 2), LOGO_SAMSUNG)
    dr.text((x + LOGO_SAMSUNG.width + 9, 40), 'Electronics', font=F_ORG, fill=INK, anchor='lm')
    im.paste(LOGO_SKKU, (W - 34 - LOGO_SKKU.width, 40 - LOGO_SKKU.height // 2), LOGO_SKKU)
    tw = dr.textlength(V['title'], font=F_TITLE); bw = dr.textlength(V['badge'], font=F_BADGE) + 28
    x0 = (W - (tw + 14 + bw)) / 2
    dr.text((x0, 6), V['title'], font=F_TITLE, fill=TEAL)
    rrect(dr, (x0 + tw + 14, 22, x0 + tw + 14 + bw, 50), 14, fill=TEAL)
    dr.text((x0 + tw + 14 + bw / 2, 36), V['badge'], font=F_BADGE, fill=WHITE, anchor='mm')
    line(dr, 76, V['subtitle'], F_SUBT, MUTED)
    tabs = V['tabs']                                               # fixed position and size, only the current one is solid
    tw_ = (W - 2 * M - 16 * (len(tabs) - 1)) / len(tabs); x = M
    for i, t in enumerate(tabs):
        rrect(dr, (x, 96, x + tw_, 122), 8, fill=TEAL if i == scene else TAB_DIM)
        dr.text((x + tw_ / 2, 109), t, font=F_TAB, fill=WHITE if i == scene else TEAL, anchor='mm')
        x += tw_ + 16
    line(dr, 154, head, F_HEAD, INK)
    line(dr, NOTE_Y, note, F_NOTE, MUTED)
    return im, dr


def framed(im, dr, pic, x, y, col, width=3):
    im.paste(pic, (x, y))
    dr.rectangle((x - 2, y - 2, x + pic.width + 1, y + pic.height + 1), outline=col, width=width)


def placeholder(dr, x, y, w, h, label):
    rrect(dr, (x, y, x + w - 1, y + h - 1), 10, fill=TEAL_BG, outline=MINT_LINE)
    dr.text((x + w / 2, y + h / 2), label, font=F_HEAD, fill=MINT_LINE, anchor='mm')


def tag(dr, x, y, text):
    """Small label in the top-left corner of a panel."""
    w = dr.textlength(text, font=F_TAG) + 16
    rrect(dr, (x + 8, y + 8, x + 8 + w, y + 32), 6, fill=WHITE)
    dr.text((x + 16, y + 20), text, font=F_TAG, fill=INK, anchor='lm')


def card(dr, box, state):
    """state: 'cur' (solid teal), 'on' (tinted) or 'off' (empty). Returns the text colours (title, body)."""
    rrect(dr, box, 12, fill={'cur': TEAL, 'on': TEAL_BG, 'off': WHITE}[state],
          outline={'cur': None, 'on': MINT_LINE, 'off': LINE}[state])
    return {'cur': (WHITE, SOFT), 'on': (INK, MUTED), 'off': (GREY, GREY)}[state]


def smooth(im, draw, ss=3):
    """Draw with `draw(layer_draw, scale)` on a larger transparent layer and scale it down: smooth diagonal lines."""
    layer = Image.new('RGBA', (W * ss, H * ss), (0, 0, 0, 0))
    draw(ImageDraw.Draw(layer), ss)
    im.paste(Image.alpha_composite(im.convert('RGBA'), layer.resize((W, H), Image.LANCZOS)).convert('RGB'))
    return ImageDraw.Draw(im)


# ---------------------------------------------------------------- scene 1: the biased problem (paper Fig. 1)
P1 = 300
P1_X = [M + k * (P1 + 12) for k in range(3)]
P1_RX0, P1_RX1 = P1_X[2] + P1 + 24, W - M


def problem_frame(s):
    S = V['problem']
    im, dr = canvas(0, S['head'], S['note'])
    shown = [True, True, s >= 1]
    for k, (key, title) in enumerate(S['columns']):
        col = BASE if key == 'wsss' else MUTED
        dr.text((P1_X[k], TOP - 10), title, font=F_H, fill=(col if key == 'wsss' else INK) if shown[k] else GREY, anchor='ls')
        for r, name in enumerate(S['rows']):
            y = TOP + r * (P1 + 12)
            if shown[k]:
                framed(im, dr, panel('problem/%s_%s.jpg' % (name, key), (P1, P1)), P1_X[k], y, col if key == 'wsss' else LINE)
                if k == 0:
                    tag(dr, P1_X[k], y, name)
            else:
                placeholder(dr, P1_X[k], y, P1, P1, title)
    box = (P1_RX0, TOP, P1_RX1, TOP + 2 * P1 + 12)
    if s < 2:
        placeholder(dr, box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1, 'Fig. 1(d)')
    else:
        card(dr, box, 'on')
        dr.text((P1_RX0 + 18, TOP + 32), fit(dr, S['list_title'], F_H, P1_RX1 - P1_RX0 - 36), font=F_H, fill=TEAL, anchor='lm')
        dr.text((P1_RX0 + 18, TOP + 60), S['list_sub'], font=F_TICK, fill=MUTED, anchor='lm')
        for i, (cls, objs) in enumerate(S['list']):
            y = TOP + 104 + i * 72
            hot = cls in S['rows']
            dr.text((P1_RX0 + 18, y), cls, font=F_HB, fill=TEAL if hot else INK, anchor='lm')
            dr.text((P1_RX0 + 18, y + 25), fit(dr, objs, F_HS, P1_RX1 - P1_RX0 - 36), font=F_HS, fill=INK if hot else MUTED, anchor='lm')
    line(dr, TOP + 2 * P1 + 12 + 46, S['messages'][s], F_MSG, TEAL if s == 2 else BASE if s == 1 else INK)
    return im


# ---------------------------------------------------------------- scene 2: separating with USS features (paper Fig. 2)
P2W, P2H = 345, 300                                                # the panels of Fig. 2 are 386 x 336
P2_X = [(W - (3 * P2W + 2 * 24)) // 2 + k * (P2W + 24) for k in range(3)]
P2_Y = [TOP, TOP + P2H + 46]
P2_COL = [MUTED, BASE, TEAL]


def separate_frame(s):
    S = V['separate']
    im, dr = canvas(1, S['head'], S['note'])
    for k, (name, title, sub) in enumerate(S['top']):
        dr.text((P2_X[k], P2_Y[0] - 10), title, font=F_H, fill=BASE if sub else INK, anchor='ls')
        if sub:
            dr.text((P2_X[k] + dr.textlength(title, font=F_H) + 10, P2_Y[0] - 10), sub, font=F_HS, fill=MUTED, anchor='ls')
        framed(im, dr, panel('separate/' + name, (P2W, P2H)), P2_X[k], P2_Y[0], BASE if sub else LINE)
    x0, x1 = P2_X[2], P2_X[2] + P2W                                 # the explanation card next to the two inputs
    dr.text((x0, P2_Y[0] + 14), fit(dr, S['card_title'], F_H, P2W), font=F_H, fill=INK, anchor='lm')
    y = para(dr, x0, P2_Y[0] + 44, S['card_sub'], F_TICK, MUTED, P2W, 19)
    for k, (name, title, feats, verdict) in enumerate(S['bottom']):
        on, cur = k < s, k == s - 1
        yk = P2_Y[0] + 86 + k * 72
        fg, sub = card(dr, (x0, yk, x1, yk + 64), 'cur' if cur else 'on' if on else 'off')
        dr.text((x0 + 14, yk + 21), fit(dr, feats, F_HB, P2W - 28), font=F_HB, fill=fg, anchor='lm')
        dr.text((x0 + 14, yk + 45), fit(dr, verdict, F_TICK, P2W - 28), font=F_TICK, fill=sub, anchor='lm')
        dr.text((P2_X[k], P2_Y[1] - 10), title, font=F_H, fill=P2_COL[k] if on else GREY, anchor='ls')
        if on:
            framed(im, dr, panel('separate/' + name, (P2W, P2H)), P2_X[k], P2_Y[1], P2_COL[k])
        else:
            placeholder(dr, P2_X[k], P2_Y[1], P2W, P2H, title[:3])
    line(dr, P2_Y[1] + P2H + 36, S['messages'][s], F_MSG, TEAL if s == 3 else BASE if s else INK)
    return im


# ---------------------------------------------------------------- scene 3: selecting by the distance (paper Fig. 3)
P3 = 372
P3_X = [56, 56 + P3 + 40, 56 + 2 * P3 + 52]
KIND = {'target': PINK, 'biased': ORANGE, 'background': BLUE}


def arrow(ld, ss, p0, p1, col):
    """Arrow with a white edge from p0 to p1."""
    a = math.atan2(p1[1] - p0[1], p1[0] - p0[0]); ux, uy = math.cos(a), math.sin(a)
    tip = lambda L, wd, back: [(p1[0] * ss, p1[1] * ss)] + [((p1[0] - L * ux + sg * wd * uy) * ss, (p1[1] - L * uy - sg * wd * ux) * ss) for sg in (1, -1)]
    end = ((p1[0] - 14 * ux) * ss, (p1[1] - 14 * uy) * ss)
    ld.line([(p0[0] * ss, p0[1] * ss), end], fill=WHITE + (255,), width=9 * ss)
    ld.polygon([((p1[0] + 3 * ux) * ss, (p1[1] + 3 * uy) * ss)] + tip(22, 13, 0)[1:], fill=WHITE + (255,))
    ld.line([(p0[0] * ss, p0[1] * ss), end], fill=col + (255,), width=5 * ss)
    ld.polygon(tip(18, 9, 0), fill=col + (255,))


def cross(ld, ss, p, col):
    for wd, c, r in ((11, WHITE, 16), (6, col, 14)):
        for sg in (1, -1):
            ld.line([((p[0] - r) * ss, (p[1] - sg * r) * ss), ((p[0] + r) * ss, (p[1] + sg * r) * ss)], fill=c + (255,), width=wd * ss)


def select_frame(s):
    S = V['select']
    im, dr = canvas(2, S['head'], S['note'])
    dr.text((P3_X[0], TOP - 10), S['panels'][0][1], font=F_H, fill=INK, anchor='ls')
    dr.text(((P3_X[1] + P3_X[2] + P3) / 2, TOP - 10), S['others_title'], font=F_H, fill=INK, anchor='ms')
    for k, (name, _) in enumerate(S['panels']):
        framed(im, dr, panel('select/' + name, (P3, P3)), P3_X[k], TOP, LINE)
    at = lambda k, rel: (P3_X[k] + rel[0] * P3, TOP + rel[1] * P3)
    pts = {kind: at(0, rel) for kind, rel in S['points'].items()}

    def marks(ld, ss):
        if s >= 1:
            for kind, ends in S['ends'].items():
                for k, rel in enumerate(ends):
                    arrow(ld, ss, pts[kind], at(k + 1, rel), KIND[kind])
        for kind, p in pts.items():
            cross(ld, ss, p, KIND[kind])
    dr = smooth(im, marks)
    if s >= 1:                                                      # the two labels of Fig. 3, on the arrows between the panels
        cx = P3_X[0] + P3 + 20
        for kind, text in (('target', 'Long Distance'), ('biased', 'Short Distance')):
            (ax, ay), (bx, by) = pts[kind], at(1, S['ends'][kind][0])
            cy = ay + (by - ay) * (cx - ax) / (bx - ax) - 26
            w = dr.textlength(text, font=F_PILL) + 22
            rrect(dr, (cx - w / 2, cy - 15, cx + w / 2, cy + 15), 6, fill=KIND[kind], outline=WHITE, width=2)
            dr.text((cx, cy - 1), text, font=F_PILL, fill=WHITE, anchor='mm')
    y0, y1, mid = TOP + P3 + 22, TOP + P3 + 22 + 232, W // 2
    fg, sub = card(dr, (P3_X[0], y0, mid - 12, y1), 'on' if s >= 1 else 'off')
    dr.text((P3_X[0] + 20, y0 + 32), S['left_title'], font=F_H, fill=fg if s < 1 else TEAL, anchor='lm')
    for i, (kind, text) in enumerate(S['left_rows']):
        y = y0 + 84 + i * 52
        rrect(dr, (P3_X[0] + 20, y - 14, P3_X[0] + 48, y + 14), 6, fill=KIND[kind] if s >= 1 else LINE)
        dr.text((P3_X[0] + 62, y), fit(dr, text, F_HB, mid - 12 - P3_X[0] - 82), font=F_HB, fill=fg, anchor='lm')
    x0, x1 = mid + 12, P3_X[2] + P3
    fg, sub = card(dr, (x0, y0, x1, y1), 'on' if s >= 2 else 'off')
    dr.text((x0 + 20, y0 + 32), S['right_title'], font=F_H, fill=fg if s < 2 else TEAL, anchor='lm')
    y = y0 + 74
    for i, text in enumerate(S['right_text']):
        y = para(dr, x0 + 20, y, text, F_BODY, fg if i == 0 else sub, x1 - x0 - 40, 25) + 16
    assert y - 16 <= y1 - 8, 'the text of the right card does not fit'
    line(dr, y1 + 42, S['messages'][s], F_MSG, TEAL if s else INK)
    return im


# ---------------------------------------------------------------- scene 4: from the WSSS label to the debiased label (paper Fig. 7)
P4 = 212
P4_X = [M + k * (P4 + 12) for k in range(4)]
P4_RX0, P4_RX1 = P4_X[3] + P4 + 24, W - M
P4_COL = [LINE, BASE, MUTED, TEAL]


def debias_frame(s):
    S = V['debias']
    im, dr = canvas(3, S['head'], S['note'])
    for k, (key, title) in enumerate(S['columns']):
        on = k <= s
        dr.text((P4_X[k], TOP - 10), title, font=F_H, fill=(P4_COL[k] if k in (1, 3) else INK) if on else GREY, anchor='ls')
        for r, name in enumerate(S['rows']):
            y = TOP + r * (P4 + 10)
            if on:
                framed(im, dr, panel('debias/%s_%s.jpg' % (name, key), (P4, P4)), P4_X[k], y, P4_COL[k])
                if k == 0:
                    tag(dr, P4_X[k], y, name)
            else:
                placeholder(dr, P4_X[k], y, P4, P4, '')
    for k, (sec, title, text) in enumerate(S['cards']):
        y = TOP + k * 190
        fg, sub = card(dr, (P4_RX0, y, P4_RX1, y + 176), 'cur' if k == s - 1 else 'on' if k < s else 'off')
        dr.text((P4_RX0 + 16, y + 24), fit(dr, sec, F_TICK, P4_RX1 - P4_RX0 - 32), font=F_TICK, fill=sub, anchor='lm')
        dr.text((P4_RX0 + 16, y + 54), title, font=F_H, fill=fg, anchor='lm')
        assert para(dr, P4_RX0 + 16, y + 90, text, F_BODY, fg, P4_RX1 - P4_RX0 - 32, 25) <= y + 176 + 8, text
    y = TOP + 3 * 190 + 22                                          # legend: the colour of the biased class
    dr.rectangle((P4_RX0, y - 13, P4_RX0 + 34, y + 13), fill=CREAM if s >= 2 else WHITE, outline=GREY)
    dr.text((P4_RX0 + 46, y), fit(dr, S['legend'], F_NOTE, P4_RX1 - P4_RX0 - 46), font=F_NOTE, fill=INK if s >= 2 else GREY, anchor='lm')
    line(dr, TOP + 3 * P4 + 20 + 32, S['messages'][s], F_MSG, TEAL if s == 3 else BASE if s == 1 else INK)
    return im


# ---------------------------------------------------------------- scene 5: results (supplementary Fig. G, Tables 3 and 5, Fig. 9)
P5 = 210
P5_X = [M + k * (P5 + 8) for k in range(4)]
P5_RX0, P5_RX1 = P5_X[3] + P5 + 24, W - M


def results_frame(s):
    S = V['results']
    im, dr = canvas(4, S['head'], S['note'])
    for k, (suffix, title) in enumerate(S['columns']):
        ours, base = suffix.startswith('mars'), suffix.startswith('rsepm')
        on = s >= 1 or not ours
        col = TEAL if ours else BASE if base else LINE
        dr.text((P5_X[k] + P5 / 2, TOP - 10), title, font=F_H, fill=(col if ours or base else INK) if on else GREY, anchor='ms')
        for r, (name, where) in enumerate(S['rows']):
            y = TOP + r * (P5 + 10)
            if on:
                framed(im, dr, panel('results/%s_%s' % (name, suffix), (P5, P5)), P5_X[k], y, col)
                if k == 0:
                    tag(dr, P5_X[k], y, where)
            else:
                placeholder(dr, P5_X[k], y, P5, P5, '')
    ch = (3 * P5 + 20 - 3 * 14) / 4
    for k, (value, metric, desc) in enumerate(S['cards']):
        y = TOP + k * (ch + 14)
        fg, sub = card(dr, (P5_RX0, y, P5_RX1, y + ch), 'on' if s >= 1 else 'off')
        cx = (P5_RX0 + P5_RX1) / 2
        dr.text((cx, y + 46), fit(dr, value, F_BIG, P5_RX1 - P5_RX0 - 24), font=F_BIG, fill=TEAL if s >= 1 else GREY, anchor='mm')
        dr.text((cx, y + 92), fit(dr, metric, F_HB, P5_RX1 - P5_RX0 - 24), font=F_HB, fill=fg, anchor='mm')
        dr.text((cx, y + 120), fit(dr, desc, F_NOTE, P5_RX1 - P5_RX0 - 24), font=F_NOTE, fill=sub, anchor='mm')
    line(dr, TOP + 3 * P5 + 20 + 36, S['messages'][s], F_MSG, TEAL if s else BASE)
    return im


frames = []
add = lambda im, ms: frames.append((im, ms))
for s, ms in enumerate((1600, 2300, 3400)):
    add(problem_frame(s), ms)
for s, ms in enumerate((1700, 1900, 1900, 3400)):
    add(separate_frame(s), ms)
for s, ms in enumerate((1700, 2500, 3600)):
    add(select_frame(s), ms)
for s, ms in enumerate((1100, 2000, 2600, 3800)):
    add(debias_frame(s), ms)
for s, ms in enumerate((2000, 6000)):
    add(results_frame(s), ms)


# GIF palette: the flat UI colours and the main logo colours are reserved so text and logos stay clean; the rest of each frame is quantized.
def logo_colors(im, n=8):
    flat = Image.new('RGB', im.size, WHITE); flat.paste(im, (0, 0), im)
    p = flat.quantize(colors=n, method=Image.MEDIANCUT).getpalette()[:3 * n]
    return [tuple(p[i:i + 3]) for i in range(0, len(p), 3)]


RESERVED = list(dict.fromkeys([TEAL, TEAL_BG, MINT_LINE, INK, MUTED, LINE, WHITE, (0, 0, 0), GREY, TAB_DIM, SOFT, BASE, PINK, ORANGE, CREAM]
                              + logo_colors(LOGO_OGQ) + logo_colors(LOGO_SAMSUNG) + logo_colors(LOGO_SKKU)))


def to_palette(f):
    k = len(RESERVED)
    q = f.quantize(colors=256 - k, method=Image.MEDIANCUT)
    p = (q.getpalette() or [])[:3 * (256 - k)]
    p = [v for c in RESERVED for v in c] + p + [0] * (3 * (256 - k) - len(p))
    pi = Image.new('P', (1, 1)); pi.putpalette(p)
    out = f.quantize(palette=pi, dither=Image.FLOYDSTEINBERG)
    idx, rgb = np.array(out), np.array(f.convert('RGB'))
    for j, col in enumerate(RESERVED):
        idx[(rgb == col).all(-1)] = j
    fixed = Image.frombytes('P', out.size, idx.astype(np.uint8).tobytes()); fixed.putpalette(p)
    return fixed


total_ms = sum(ms for _, ms in frames)
if args.preview:
    os.makedirs(args.preview, exist_ok=True)
    for i, (f, _) in enumerate(frames):
        f.save(os.path.join(args.preview, f'frame_{i:03d}.png'))
    json.dump([ms for _, ms in frames], open(os.path.join(args.preview, 'durations.json'), 'w'))
    print(f'{len(frames)} frames ({total_ms / 1000:.1f} s) written to {args.preview}')
else:
    before = open(args.out, 'rb').read() if os.path.exists(args.out) else None
    pal = [to_palette(f) for f, _ in frames]
    pal[0].save(args.out, save_all=True, append_images=pal[1:], duration=[ms for _, ms in frames], loop=0, optimize=True, disposal=1)
    after = open(args.out, 'rb').read()
    state = 'new file' if before is None else 'identical to the previous file' if before == after else 'differs from the previous file'
    print(f'{args.out}: {len(frames)} frames, {total_ms / 1000:.1f} s, {len(after) / 1e6:.1f} MB ({state})')
