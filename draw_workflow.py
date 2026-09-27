"""Render a compact, readable snake-layout workflow diagram as PNG."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
W, H = 1600, 620
im = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(im)
font_dir = Path("/usr/share/fonts/truetype/dejavu")
font = ImageFont.truetype(str(font_dir / "DejaVuSans.ttf"), 28)
small = ImageFont.truetype(str(font_dir / "DejaVuSans.ttf"), 22)
stage = ImageFont.truetype(str(font_dir / "DejaVuSans-Bold.ttf"), 23)
blue = "#e2eff8"
stroke = "#253b53"
text = "#172536"
box_w, box_h = 430, 112
xs = [55, 585, 1115]
ys = [82, 272, 462]
labels = [
    ["Prepare benign / malignant\nimage folders", "Read RGB H&E\nmicroscopy image", "Resize to 128×128\n+ CLAHE enhancement"],
    ["HOG + LBP texture\n+ color histograms", "Scale feature vector\n+ train RBF SVM", "Patient-aware holdout\nevaluation"],
    ["Save model +\nJSON metrics", "Predict new image:\nbenign / malignant", "JSON result + input /\nenhanced visual panel"],
]


def box(x, y, label):
    draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=18, fill=blue, outline=stroke, width=3)
    lines = label.split("\n")
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    total = sum(line_heights) + 7 * (len(lines) - 1)
    cy = y + (box_h - total) // 2
    for line, lh in zip(lines, line_heights):
        bb = draw.textbbox((0, 0), line, font=font)
        tw = bb[2] - bb[0]
        draw.text((x + (box_w - tw) / 2, cy), line, font=font, fill=text)
        cy += lh + 7


def arrow(start, end):
    draw.line([start, end], fill=stroke, width=5)
    import math
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 18
    p1 = (end[0] - size * math.cos(angle - math.pi / 6), end[1] - size * math.sin(angle - math.pi / 6))
    p2 = (end[0] - size * math.cos(angle + math.pi / 6), end[1] - size * math.sin(angle + math.pi / 6))
    draw.polygon([end, p1, p2], fill=stroke)

# Stage captions
for y, caption in zip(ys, ["TRAINING: IMAGE PREPARATION", "FEATURES, MODEL AND EVALUATION", "SAVED MODEL: INFERENCE"]):
    draw.text((55, y - 34), caption, font=stage, fill="#365c7d")
for row, y in enumerate(ys):
    for col, x in enumerate(xs):
        box(x, y, labels[row][col])
# Training preparation, then snake back across row two, then forward across inference.
arrow((xs[0] + box_w + 14, ys[0] + box_h / 2), (xs[1] - 14, ys[0] + box_h / 2))
arrow((xs[1] + box_w + 14, ys[0] + box_h / 2), (xs[2] - 14, ys[0] + box_h / 2))
arrow((xs[2] + box_w / 2, ys[0] + box_h + 8), (xs[2] + box_w / 2, ys[1] - 8))
arrow((xs[2] - 14, ys[1] + box_h / 2), (xs[1] + box_w + 14, ys[1] + box_h / 2))
arrow((xs[1] - 14, ys[1] + box_h / 2), (xs[0] + box_w + 14, ys[1] + box_h / 2))
arrow((xs[0] + box_w / 2, ys[1] + box_h + 8), (xs[0] + box_w / 2, ys[2] - 8))
arrow((xs[0] + box_w + 14, ys[2] + box_h / 2), (xs[1] - 14, ys[2] + box_h / 2))
arrow((xs[1] + box_w + 14, ys[2] + box_h / 2), (xs[2] - 14, ys[2] + box_h / 2))

out = ROOT / "workflow.png"
im.save(out)
print(out)
