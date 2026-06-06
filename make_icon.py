from PIL import Image, ImageDraw
import math, os

def make_banana(size):
    scale = 4
    w = size * scale
    img = Image.new('RGBA', (w, w), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Ellipse bbox — wide and flat (banana proportions)
    bx1, by1 = w * 0.06, w * 0.28
    bx2, by2 = w * 0.94, w * 0.72
    aw = int(w * 0.15)  # arc stroke width

    # Shadow (offset, semi-transparent dark)
    off = w * 0.025
    draw.arc([bx1+off, by1+off, bx2+off, by2+off],
             start=5, end=175, fill=(120, 90, 0, 140), width=aw)

    # Underside (darker yellow)
    draw.arc([bx1, by1, bx2, by2],
             start=5, end=175, fill=(210, 160, 0), width=aw)

    # Main banana body (bright yellow, slightly raised)
    lift = aw * 0.3
    draw.arc([bx1, by1 - lift, bx2, by2 - lift],
             start=5, end=175, fill=(255, 215, 0), width=int(aw * 0.78))

    # Highlight strip (inner, light yellow)
    inn = aw * 0.3
    draw.arc([bx1 + inn, by1 - lift + inn, bx2 - inn, by2 - lift + inn],
             start=10, end=170, fill=(255, 248, 130), width=int(aw * 0.25))

    # Tips (dark brown)
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    rx = (bx2 - bx1) / 2
    ry = (by2 - by1) / 2
    r = aw * 0.58

    for angle in (5, 175):
        rad = math.radians(angle)
        tx = cx + rx * math.cos(rad)
        ty = cy + ry * math.sin(rad) - lift
        draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=(75, 45, 10))

    # Rotate 35° CCW → banana tilted (right end higher)
    img = img.rotate(35, resample=Image.BICUBIC)

    return img.resize((size, size), Image.LANCZOS)


sizes  = [256, 128, 64, 48, 32, 16]
frames = [make_banana(s) for s in sizes]

out = os.path.join(os.path.dirname(__file__), 'banana_sorter.ico')
frames[0].save(out, format='ICO',
               sizes=[(s, s) for s in sizes],
               append_images=frames[1:])
print(f'Saved: {out}')
