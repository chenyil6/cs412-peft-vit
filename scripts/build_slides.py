"""Build the project talk (editable .pptx) with python-pptx.

    python scripts/build_slides.py --out slides/cs412_peft_vit_talk.pptx

Figures come from latex/final/figures (scripts/analyze.py, scripts/analyze_ablations.py); numbers from
results/ and runs/. 16:9, Cambria headings / Calibri body, dark slate title and closing slides.
"""
import argparse, json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# palette: slate dominant, figure-matched orange accent, aqua secondary
SLATE, SLATE2 = RGBColor(0x1F, 0x2A, 0x37), RGBColor(0x2E, 0x3D, 0x4F)
INK, MUTED, LINE = RGBColor(0x1A, 0x1A, 0x1A), RGBColor(0x6B, 0x72, 0x80), RGBColor(0xD9, 0xDC, 0xE1)
ORANGE, AQUA, BLUE, YELLOW, PINK = RGBColor(0xEB, 0x68, 0x34), RGBColor(0x1B, 0xAF, 0x7A), RGBColor(0x2A, 0x78, 0xD6), RGBColor(0xED, 0xA1, 0x00), RGBColor(0xE8, 0x7B, 0xA4)
WHITE, TINT = RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xF3, 0xF5, 0xF8)
HEAD, BODY = "Cambria", "Calibri"
W, H = Inches(13.333), Inches(7.5)
FIG = Path("latex/final/figures")


def text(slide, x, y, w, h, s, size=16, bold=False, color=INK, font=BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, margin=0):
    tb = slide.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(margin)
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        runs = line if isinstance(line, list) else [(line, {})]
        for t, o in runs:
            r = p.add_run(); r.text = t; f = r.font; f.name = o.get("font", font); f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold); f.italic = o.get("italic", italic)
            f.color.rgb = o.get("color", color)
        p.space_after = Pt(lines and 4)
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=INK, gap=8):
    tb = slide.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        runs = it if isinstance(it, list) else [(it, {})]
        r0 = p.add_run(); r0.text = "•  "; r0.font.size = Pt(size); r0.font.color.rgb = ORANGE; r0.font.name = BODY
        for t, o in runs:
            r = p.add_run(); r.text = t; r.font.name = BODY; r.font.size = Pt(o.get("size", size)); r.font.bold = o.get("bold", False); r.font.color.rgb = o.get("color", color)
        p.space_after = Pt(gap)
    return tb


def rect(slide, x, y, w, h, fill, line=None, rounded=False, radius=0.08):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, x, y, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None: sh.line.fill.background()
    else: sh.line.color.rgb = line; sh.line.width = Pt(0.75)
    if rounded: sh.adjustments[0] = radius
    sh.shadow.inherit = False
    return sh


def title(slide, s, dark=False):
    text(slide, Inches(0.6), Inches(0.4), Inches(12.1), Inches(0.9), s, size=34, bold=True, color=WHITE if dark else SLATE, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)


def footer(slide, n, dark=False):
    text(slide, Inches(0.6), Inches(7.05), Inches(9), Inches(0.3), "CS-412 Fall 2026 · PEFT for Vision Transformers · Huiyi Chen", size=10, color=RGBColor(0x9A, 0xA4, 0xB2) if dark else MUTED)
    text(slide, Inches(12.0), Inches(7.05), Inches(0.75), Inches(0.3), str(n), size=10, color=RGBColor(0x9A, 0xA4, 0xB2) if dark else MUTED, align=PP_ALIGN.RIGHT)


def stat(slide, x, y, w, big, label, color=SLATE, size=44):
    hh = Inches(size * 0.02)            # 1.2 x line height of the number
    text(slide, x, y, w, hh, big, size=size, bold=True, color=color, font=HEAD, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.BOTTOM)
    text(slide, x, y + hh, w, Inches(0.55), label, size=13, color=MUTED)


def card(slide, x, y, w, h, head, body, accent, head_size=18, body_size=14):
    rect(slide, x, y, w, h, TINT, rounded=True)
    dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.25), y + Inches(0.28), Inches(0.22), Inches(0.22)); dot.fill.solid(); dot.fill.fore_color.rgb = accent; dot.line.fill.background()
    text(slide, x + Inches(0.6), y + Inches(0.18), w - Inches(0.8), Inches(0.45), head, size=head_size, bold=True, color=SLATE, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, x + Inches(0.25), y + Inches(0.75), w - Inches(0.5), h - Inches(0.9), body, size=body_size, color=INK)


def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), x, y, width=w, height=h)


def table(slide, x, y, w, rows, col_w, size=12, header_fill=SLATE, row_h=0.34, bold_first_col=False):
    nrows, ncols = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nrows, ncols, x, y, w, Inches(row_h * nrows)); tbl = shp.table
    for j, cw in enumerate(col_w): tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tbl.cell(i, j); c.text = ""; p = c.text_frame.paragraphs[0]; r = p.add_run(); r.text = str(val); r.font.size = Pt(size); r.font.name = BODY
            c.margin_left = c.margin_right = Inches(0.08); c.margin_top = c.margin_bottom = Inches(0.03); c.vertical_anchor = MSO_ANCHOR.MIDDLE
            if i == 0:
                c.fill.solid(); c.fill.fore_color.rgb = header_fill; r.font.color.rgb = WHITE; r.font.bold = True
            else:
                c.fill.solid(); c.fill.fore_color.rgb = WHITE if i % 2 else TINT; r.font.color.rgb = INK; r.font.bold = bold_first_col and j == 0
            if j > 0 and i > 0: p.alignment = PP_ALIGN.CENTER
    return tbl


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="slides/cs412_peft_vit_talk.pptx"); a = ap.parse_args()
    prs = Presentation(); prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]
    main_tbl = {r.split(",")[0]: r.split(",") for r in open("results/summary_by_method.csv").read().strip().split("\n")[1:]}
    acc = {m: (float(v[2]), float(v[3])) for m, v in main_tbl.items()}
    n = 0

    # 1 title (dark) --------------------------------------------------------------------------
    s = prs.slides.add_slide(blank); n += 1; rect(s, 0, 0, W, H, SLATE)
    text(s, Inches(0.9), Inches(1.6), Inches(11.5), Inches(2.2), ["A Unified Empirical Comparison of", "Parameter-Efficient Fine-Tuning Methods", "for Vision Transformers"], size=38, bold=True, color=WHITE, font=HEAD)
    text(s, Inches(0.9), Inches(4.3), Inches(11.5), Inches(0.5), "BitFit vs LoRA vs VPT-deep, against a linear probe and full fine-tuning, under one protocol", size=18, color=RGBColor(0xCA, 0xDC, 0xFC), italic=True)
    text(s, Inches(0.9), Inches(5.4), Inches(11.5), Inches(1.0), [[("Huiyi Chen", {"bold": True, "size": 18, "color": WHITE})], [("CS-412 Machine Learning, Fall 2026 · University of Illinois Chicago", {"size": 14, "color": RGBColor(0x9A, 0xA4, 0xB2)})], [("Anchor paper: Visual Prompt Tuning, Jia et al., ECCV 2022", {"size": 14, "color": RGBColor(0x9A, 0xA4, 0xB2)})]], size=14, color=WHITE)
    notes(s, "Hello, I am Huiyi Chen. My project is a controlled comparison of three parameter-efficient fine-tuning methods for vision transformers. The anchor paper is Visual Prompt Tuning from ECCV 2022. About ten minutes.")

    # 2 problem ---------------------------------------------------------------------------------
    s = prs.slides.add_slide(blank); n += 1; title(s, "Full fine-tuning is the default, and it is expensive"); footer(s, n)
    stat(s, Inches(0.6), Inches(1.5), Inches(3.6), "86M", "parameters copied per task when ViT-B/16 is fully fine-tuned", ORANGE, 54)
    stat(s, Inches(0.6), Inches(3.3), Inches(3.6), "< 1%", "of the model trained by a PEFT method; the backbone is frozen and shared", AQUA, 54)
    text(s, Inches(0.6), Inches(5.3), Inches(3.8), Inches(1.4), "Three families, three hypotheses about where adaptation should happen.", size=15, color=MUTED, italic=True)
    for i, (hd, bd, col) in enumerate([("BitFit", "Unfreeze the existing bias terms. No new parameters; shifts every sub-layer's input distribution.", BLUE), ("LoRA", "Add a low-rank update ΔW = (α/r)·BA to the attention weights W_q and W_v. Folds back into the weights at inference.", AQUA), ("VPT-deep", "Prepend p learnable prompt tokens to every layer's input. Changes what frozen attention attends to, not the weights.", YELLOW)]):
        card(s, Inches(4.9) + i * Inches(2.75), Inches(1.5), Inches(2.6), Inches(3.3), hd, bd, col)
    rect(s, Inches(4.9), Inches(5.1), Inches(8.0), Inches(1.5), TINT, rounded=True)
    text(s, Inches(5.15), Inches(5.25), Inches(7.5), Inches(1.2), [[("Shared by all five methods: ", {"bold": True}), ("one frozen ImageNet-21k ViT-B/16, one dataset, one linear head (768→100), one training protocol. ", {}), ("Linear probe", {"bold": True}), (" = lower bound (head only); ", {}), ("full fine-tuning", {"bold": True}), (" = upper reference.", {})]], size=14)
    notes(s, "Transfer learning from a pre-trained backbone is the default recipe, and full fine-tuning copies all 86 million parameters per task. PEFT freezes the backbone and trains under one percent. The three methods I compare make different bets about where adaptation should happen: biases, a low-rank weight update, or the input tokens. Two baselines bracket them.")

    # 3 research question ---------------------------------------------------------------------
    s = prs.slides.add_slide(blank); n += 1; title(s, "Published comparisons are not comparable"); footer(s, n)
    bullets(s, Inches(0.6), Inches(1.5), Inches(5.9), Inches(3.6), [
        [("Different backbones and benchmarks: ", {"bold": True}), ("BitFit and LoRA were tuned on GLUE with BERT / RoBERTa; VPT on vision with per-dataset search.", {})],
        [("Different search budgets: ", {"bold": True}), ("a method with more knobs looks stronger because more configurations were tried.", {})],
        [("Accuracy only: ", {"bold": True}), ("memory, time and inference cost are rarely reported side by side.", {})],
        [("The shared head is never isolated: ", {"bold": True}), ("how much of the gain is the mechanism, how much the classifier?", {})]], size=16, gap=12)
    rect(s, Inches(6.9), Inches(1.5), Inches(5.8), Inches(3.0), RGBColor(0xEE, 0xF2, 0xFB), line=SLATE2, rounded=True)
    text(s, Inches(7.2), Inches(1.65), Inches(5.2), Inches(2.7), [[("Research question", {"bold": True, "italic": True, "size": 15, "color": SLATE, "font": HEAD})], [("Under an identical backbone, dataset and training protocol, how do BitFit, LoRA and VPT-deep compare, and how much of their benefit comes from the adaptation mechanism rather than from the classification head that all methods share?", {"size": 16})]], size=16)
    text(s, Inches(0.6), Inches(5.4), Inches(12.1), Inches(1.3), [[("Design: ", {"bold": True, "color": SLATE}), ("from-scratch implementations · each method's own published hyper-parameters, no per-task search · accuracy + parameters + memory + time · 3 seeds", {})]], size=15)
    notes(s, "The problem with the literature is that results come from different backbones, benchmarks and search budgets. So my research question is explicitly about a controlled setting. The design: implement everything from scratch, take each method's hyper-parameters from its own paper with no per-task search, and measure cost as well as accuracy.")

    # 4 setup -----------------------------------------------------------------------------------
    s = prs.slides.add_slide(blank); n += 1; title(s, "One backbone, one dataset, one protocol"); footer(s, n)
    for i, (hd, bd, col) in enumerate([("Backbone", "ViT-B/16, supervised ImageNet-21k (the VPT paper's checkpoint). 85.8M parameters, frozen, loaded once.", BLUE),
                                        ("Data", "FGVC-Aircraft, 100 variants, 10,000 images. Official train 3,334 / val 3,333 / test 3,333. Fine wing, engine and tail cues: large domain gap.", ORANGE),
                                        ("Head", "One linear layer 768→100 (76,900 params), same init for every method. Reported with and without it.", AQUA),
                                        ("Protocol", "Batch 128, 100 epochs, cosine with 10 warm-up epochs, bf16, one A100 per run, 3 seeds. Selection on val only; test touched once, last epoch.", PINK)]):
        x, y = Inches(0.6) + (i % 2) * Inches(6.25), Inches(1.5) + (i // 2) * Inches(2.25)
        card(s, x, y, Inches(5.95), Inches(2.0), hd, bd, col)
    text(s, Inches(0.6), Inches(6.15), Inches(12.1), Inches(0.8), [[("Metrics per run: ", {"bold": True, "color": SLATE}), ("last-epoch test top-1 · trainable parameters · peak GPU memory · time per epoch · per-epoch curves. ", {}), ("Verified by 8 unit tests: ", {"bold": True, "color": SLATE}), ("exact parameter counts, gradient flow, LoRA identity-at-init and merge, per-layer prompt replacement.", {})]], size=14)
    notes(s, "Everything shared is fixed here. The backbone is the same checkpoint VPT used. FGVC-Aircraft is fine-grained with a large domain gap, which is what makes the adaptation mechanism matter. The head is identical for all methods, and selection is only ever on the validation split.")

    # 5 methods detail --------------------------------------------------------------------------
    s = prs.slides.add_slide(blank); n += 1; title(s, "Three mechanisms, implemented from scratch"); footer(s, n)
    cols = [("BitFit", BLUE, "102,912 params · 0.12%", ["All .bias tensors: q/k/v, proj, fc1, fc2, both LayerNorms, patch embed, final norm.", "Per block 8,448 × 12 blocks + 2 × 768.", "Only shifts pre-activations; cannot reshape a linear map."]),
            ("LoRA (r = 8)", AQUA, "294,912 params · 0.34%", ["timm fuses q,k,v into one 768→2,304 layer; the rank-8 path is added to the q and v slices only.", "B = 0 at init, so step 0 equals the pre-trained model; ∂L/∂A = BᵀG = 0 until B moves.", "merge(): W ← W + (α/r)·BA, zero inference cost."]),
            ("VPT-deep (p = 50)", YELLOW, "460,800 params · 0.54%", ["Block i sees [CLS, P_i, 196 patches]; P_{i-1}'s outputs are dropped and replaced.", "Uniform(±0.0625) init, dropout 0.1, no positional embedding.", "Sequence 197→247 tokens: ~1.25× cost per block, at training and inference."])]
    for i, (hd, col, pc, items) in enumerate(cols):
        x = Inches(0.6) + i * Inches(4.15)
        rect(s, x, Inches(1.5), Inches(3.95), Inches(5.3), TINT, rounded=True)
        dot = s.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.25), Inches(1.78), Inches(0.22), Inches(0.22)); dot.fill.solid(); dot.fill.fore_color.rgb = col; dot.line.fill.background()
        text(s, x + Inches(0.6), Inches(1.65), Inches(3.2), Inches(0.5), hd, size=20, bold=True, color=SLATE, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + Inches(0.25), Inches(2.25), Inches(3.5), Inches(0.4), pc, size=14, bold=True, color=col)
        bullets(s, x + Inches(0.25), Inches(2.8), Inches(3.5), Inches(3.9), items, size=13, gap=8)
    notes(s, "BitFit is a loop over bias tensors. LoRA had one real engineering wrinkle: timm fuses q, k and v into one projection, so the low-rank path is added to two slices of its output. A detail that surfaced in testing: with B initialised to zero, the gradient of A is exactly zero at the first step, which is a property of LoRA, not a bug. VPT-deep replaces the prompts at every block, as in the official code.")

    # 6 hyper-parameters ---------------------------------------------------------------------
    s = prs.slides.add_slide(blank); n += 1; title(s, "Hyper-parameters from each paper, no search"); footer(s, n)
    rows = [["Method", "Optimizer", "lr", "Weight decay", "Method-specific", "Source"],
            ["Linear probe", "SGD m=0.9", "3.0", "0", "—", "selected on val (plateau for lr ≥ 3); SGD as in VPT's Linear baseline"],
            ["BitFit", "AdamW", "1e-3", "0", "all bias terms", "BitFit paper: bias-only lr grid {1,4,7,10}e-4; NOAH's ViT default"],
            ["LoRA", "AdamW", "5e-4", "0.1", "r=8, α=8, W_q and W_v", "LoRA paper, RoBERTa-base GLUE recipe (Table 9)"],
            ["VPT-deep", "SGD m=0.9", "0.1", "0.01", "p=50, dropout 0.1", "VPT paper + released CUB config; p = grid value nearest FGVC mean"],
            ["Full FT", "SGD m=0.9", "0.01", "0", "grad-clip 1.0", "ViT paper, App. B.1.1"]]
    table(s, Inches(0.6), Inches(1.5), Inches(12.1), rows, [1.5, 1.3, 0.8, 1.3, 2.3, 4.9], size=12, row_h=0.5, bold_first_col=True)
    text(s, Inches(0.6), Inches(4.9), Inches(12.1), Inches(1.9), [[("Why: ", {"bold": True, "color": SLATE}), ("no method benefits from a larger search budget than another. The only searched value is the linear probe's learning rate, on train/val, because an under-trained lower bound would inflate every PEFT method's gain.", {})], [("Shared: ", {"bold": True, "color": SLATE}), ("100 epochs + 10 warm-up, cosine schedule (LoRA and BitFit originally use linear decay), batch 128.", {})]], size=14)
    notes(s, "This is the protocol's core commitment. Every number here is traceable to a paper or its official code. The one exception is the linear probe, which has no paper to inherit from; I swept it on the validation split, and the large SGD learning rate is expected for a frozen-feature head with 100 classes.")

    # 7 main result ---------------------------------------------------------------------------
    s = prs.slides.add_slide(blank); n += 1; title(s, "Full fine-tuning leads; LoRA is 1.4 points behind"); footer(s, n)
    picture(s, FIG / "fig_acc_vs_params.png", Inches(0.6), Inches(1.5), w=Inches(6.0))
    stat(s, Inches(7.7), Inches(1.45), Inches(2.5), f"{acc['full_ft'][0]:.1f}", "full fine-tuning, 85.8M params", PINK, 40)
    stat(s, Inches(10.3), Inches(1.45), Inches(2.5), f"{acc['lora'][0]:.1f}", "LoRA r=8, 294,912 params", AQUA, 40)
    stat(s, Inches(7.7), Inches(3.1), Inches(2.5), f"{acc['bitfit'][0]:.1f}", "BitFit, 102,912 params", ORANGE, 40)
    stat(s, Inches(10.3), Inches(3.1), Inches(2.5), f"{acc['vpt_deep'][0]:.1f}", "VPT-deep p=50, 460,800 params", YELLOW, 40)
    stat(s, Inches(7.7), Inches(4.75), Inches(2.5), f"{acc['linear_probe'][0]:.1f}", "linear probe, head only", BLUE, 40)
    text(s, Inches(10.3), Inches(4.8), Inches(2.6), Inches(1.6), "Test top-1 (%), mean of 3 seeds, 100 epochs. Std 0.1 to 0.7 for every method.", size=12, color=MUTED, italic=True)
    text(s, Inches(0.6), Inches(6.35), Inches(12.1), Inches(0.6), [[("Parameter count does not order the methods: ", {"bold": True, "color": SLATE}), ("VPT-deep has the most parameters of the three and the lowest accuracy. Where the parameters sit matters more than how many.", {})]], size=14)
    notes(s, "Here is the main result. Full fine-tuning is the most accurate at 73.9. LoRA reaches 72.5 with a third of a percent of the parameters, which is 98 percent of full fine-tuning. BitFit and VPT trail by 3 to 4 points. Note the ordering: VPT has the most parameters of the three PEFT methods and the lowest accuracy.")

    # 8 native bar chart: % of full FT + head contribution --------------------------------------
    s = prs.slides.add_slide(blank); n += 1; title(s, "The head gives 53.8; mechanisms add 16 to 20"); footer(s, n)
    cd = CategoryChartData(); order = ["linear_probe", "bitfit", "lora", "vpt_deep", "full_ft"]; names = ["Linear probe", "BitFit", "LoRA", "VPT-deep", "Full FT"]
    cd.categories = names; cd.add_series("Test top-1 (%)", [round(acc[m][0], 1) for m in order])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.5), Inches(7.2), Inches(5.2), cd); ch = gf.chart
    ch.has_legend = False; ch.has_title = False
    plot = ch.plots[0]; plot.gap_width = 70; plot.has_data_labels = True; plot.data_labels.font.size = Pt(13); plot.data_labels.font.name = BODY; plot.data_labels.number_format = '0.0'; plot.data_labels.number_format_is_linked = False; plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    for pt, col in zip(plot.series[0].points, [BLUE, ORANGE, AQUA, YELLOW, PINK]):
        pt.format.fill.solid(); pt.format.fill.fore_color.rgb = col
    va = ch.value_axis; va.minimum_scale = 0; va.maximum_scale = 80; va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = LINE; va.tick_labels.font.size = Pt(11); va.tick_labels.font.color.rgb = MUTED; va.format.line.fill.background()
    ca = ch.category_axis; ca.tick_labels.font.size = Pt(12); ca.tick_labels.font.color.rgb = INK; ca.format.line.color.rgb = LINE
    rows = [["Method", "% of full FT acc.", "Gain over head"], ["Linear probe", "72.8", "—"], ["BitFit", "95.4", "+16.6"], ["LoRA", "98.1", "+18.6"], ["VPT-deep", "94.3", "+15.9"], ["Full FT", "100", "+20.1"]]
    table(s, Inches(8.3), Inches(1.6), Inches(4.4), rows, [1.6, 1.5, 1.3], size=13, row_h=0.42, bold_first_col=True)
    text(s, Inches(8.3), Inches(4.5), Inches(4.4), Inches(2.3), [[("Finding 2. ", {"bold": True, "color": SLATE}), ("The shared head reaches 53.8% on frozen features (VPT reports 51.3 for its linear probe on Stanford Cars). Everything above that is the adaptation mechanism, so the comparison measures the mechanisms, not the classifier.", {})]], size=14)
    notes(s, "This answers the second half of the research question. The linear probe, which is just the shared head on frozen features, gets 53.8. Every PEFT method adds 16 to 20 points on top, so the comparison is genuinely about the mechanisms.")

    # 9 convergence ---------------------------------------------------------------------------
    s = prs.slides.add_slide(blank); n += 1; title(s, "VPT-deep is the only method still learning at epoch 100"); footer(s, n)
    picture(s, FIG / "fig_curves.png", Inches(0.6), Inches(1.5), w=Inches(6.6))
    bullets(s, Inches(7.9), Inches(1.6), Inches(4.8), Inches(3.4), [
        [("BitFit, LoRA, full FT: ", {"bold": True}), ("flat after epoch 80; gain < 0.3 points over the last 20 epochs; 91 to 98% training accuracy.", {})],
        [("VPT-deep: ", {"bold": True}), ("+11 points over the last 20 epochs; 79% training accuracy; limited by optimisation, not capacity.", {})],
        [("Why: ", {"bold": True}), ("random prompts must steer frozen attention through 12 frozen layers, trained with SGD at lr 0.1, hence the oscillations.", {})]], size=15, gap=12)
    rect(s, Inches(7.9), Inches(5.1), Inches(4.8), Inches(1.6), RGBColor(0xEE, 0xF2, 0xFB), line=SLATE2, rounded=True)
    text(s, Inches(8.15), Inches(5.2), Inches(4.3), Inches(1.4), [[("Finding 3. ", {"bold": True, "italic": True, "color": SLATE}), ("VPT-deep's number is a lower bound; its seed std fell from 1.2 to 0.3 when the budget doubled.", {})]], size=14)
    notes(s, "The curves explain the ranking. Three methods have converged. VPT-deep is still rising steeply at epoch 100 and has the lowest training accuracy, so it is under-optimised rather than under-parameterised. I come back to this in the ablations.")

    # 10 cost -----------------------------------------------------------------------------------
    s = prs.slides.add_slide(blank); n += 1; title(s, "PEFT saves memory, not time"); footer(s, n)
    picture(s, FIG / "fig_cost.png", Inches(0.6), Inches(1.5), w=Inches(6.4))
    inf = json.load(open("results/inference_profile.json")); base = inf["plain"]["ms_per_batch"]
    rows = [["Model at inference", "ms / batch", "vs plain", "GB"]] + [[nm, f"{inf[k]['ms_per_batch']:.1f}", f"{100*inf[k]['ms_per_batch']/base-100:+.0f}%", f"{inf[k]['peak_mem_mb']/1024:.2f}"] for nm, k in [("Plain ViT (LP / BitFit / full FT)", "plain"), ("LoRA un-merged", "lora_unmerged"), ("LoRA merged", "lora_merged"), ("VPT-deep p=50", "vpt_deep_p50")]]
    table(s, Inches(7.3), Inches(1.6), Inches(5.4), rows, [2.6, 1.0, 0.9, 0.9], size=12, row_h=0.4, bold_first_col=True)
    text(s, Inches(7.3), Inches(3.9), Inches(5.4), Inches(1.0), "Inference at batch 128, bf16, A100-40GB, after warm-up.", size=12, color=MUTED, italic=True)
    rect(s, Inches(7.3), Inches(4.7), Inches(5.4), Inches(2.0), RGBColor(0xEE, 0xF2, 0xFB), line=SLATE2, rounded=True)
    text(s, Inches(7.55), Inches(4.8), Inches(4.9), Inches(1.8), [[("Finding 4. ", {"bold": True, "italic": True, "color": SLATE}), ("Peak training memory drops 17 to 33%, time per epoch only 2 to 15%: the backward pass still traverses the frozen backbone (cost ≈ 2/3 of full), only weight gradients are skipped. Merged LoRA costs exactly what the plain model costs; VPT pays +27% time and +46% memory at every inference.", {})]], size=13)
    notes(s, "Cost is the other half of the story. Memory savings are real, up to a third. Time savings are small, because gradients must still flow back through every frozen block; my cost model predicts about two thirds of the full backward and that is what we measure. At inference, merged LoRA is identical to the plain model, while VPT's extra tokens cost 27 percent more forever.")

    # 11 budget ---------------------------------------------------------------------------------
    s = prs.slides.add_slide(blank); n += 1; title(s, "The training budget changes the conclusion"); footer(s, n)
    picture(s, FIG / "fig_budget.png", Inches(0.6), Inches(1.5), w=Inches(6.8))
    bullets(s, Inches(7.8), Inches(1.6), Inches(4.9), Inches(3.2), [
        [("First run at 50 epochs ", {"bold": True}), ("(half the VPT paper's budget) to save compute.", {})],
        [("Every backbone-touching method gained 4 to 5 points ", {"bold": True}), ("from 50 to 100 epochs; the linear probe gained 1.", {})],
        [("At 50 epochs LoRA and full FT looked tied ", {"bold": True}), ("(68.5 vs 68.9); at 100, full FT leads by 1.4, outside the noise.", {})]], size=15, gap=12)
    rect(s, Inches(7.8), Inches(5.0), Inches(4.9), Inches(1.7), RGBColor(0xEE, 0xF2, 0xFB), line=SLATE2, rounded=True)
    text(s, Inches(8.05), Inches(5.1), Inches(4.4), Inches(1.5), [[("Finding 5. ", {"bold": True, "italic": True, "color": SLATE}), ("A compute-limited comparison would have reached the wrong conclusion. The budget must be long enough for the slowest method, and checked per method.", {})]], size=14)
    notes(s, "This is the finding I did not plan for. I first ran the grid at 50 epochs. Then I noticed VPT had not converged and reran everything at 100. Every method that touches the backbone moved by 4 to 5 points, and the LoRA-versus-full-fine-tuning verdict flipped from tied to a clear gap.")

    # 12 ablations ------------------------------------------------------------------------------
    s = prs.slides.add_slide(blank); n += 1; title(s, "Ablations: each method is sensitive to a different knob"); footer(s, n)
    picture(s, FIG / "fig_vpt_budget.png", Inches(0.6), Inches(1.5), w=Inches(5.6))
    text(s, Inches(0.6), Inches(5.6), Inches(5.6), Inches(1.2), "VPT-deep at 200 epochs: 71.5 ± 0.6 (vs 69.7), still rising in its last 20 epochs.", size=13, color=MUTED, italic=True)
    rows = [["Ablation (seed 0 unless ±)", "Test acc. (%)", "Params"],
            ["LoRA r = 2 / 8 / 32", "70.2 / 72.0 / 74.9", "74k / 295k / 1.18M"],
            ["VPT p = 10 / 50 / 200", "59.4 / 69.3 / 69.9", "92k / 461k / 1.84M"],
            ["VPT p=50, 200 epochs (3 seeds)", "71.5 ± 0.6", "461k"],
            ["BitFit all biases (3 seeds)", "70.5 ± 0.6", "103k"],
            ["BitFit b_q + b_2 only (3 seeds)", "65.2 ± 0.1", "18k"]]
    table(s, Inches(6.6), Inches(1.5), Inches(6.1), rows, [2.9, 1.8, 1.4], size=12, row_h=0.42, bold_first_col=True)
    bullets(s, Inches(6.6), Inches(4.25), Inches(6.1), Inches(2.6), [
        [("LoRA r=32 passes full FT ", {"bold": True}), ("(74.9 vs 73.9) at 1.4% of the parameters; the paper's r=8 handicaps it here.", {})],
        [("VPT collapses at p=10 ", {"bold": True}), ("and gains nothing from p=200 while doubling cost (12 GB, 1.6× time).", {})],
        [("BitFit's recommended subset does not transfer: ", {"bold": True}), ("b_q + b_2 loses a quarter of the gain on ViT.", {})]], size=13, gap=8)
    notes(s, "Four ablations, each 10 to 25 GPU minutes. Doubling VPT's budget again gives 71.5 and it is still rising. Prompt length is the most sensitive knob in the study. LoRA improves smoothly with rank and at rank 32 passes full fine-tuning. BitFit's two-bias subset, which works on BERT, does not transfer to ViT.")

    # 13 lessons ---------------------------------------------------------------------------------
    s = prs.slides.add_slide(blank); n += 1; title(s, "What I learned"); footer(s, n)
    lessons = [("1", "The protocol is the result", "Budget, normalisation, head init and the evaluation split all had to be fixed before any number meant anything; one of them silently flipped a conclusion.", BLUE),
               ("2", "“From the paper” is under-specified", "Methods that tuned per task leave choices (VPT's p, LoRA's α) that I had to make and document.", ORANGE),
               ("3", "Baselines need the same care", "The linear probe needs an lr 100× larger than the PEFT methods; an untuned baseline would have overstated every gain.", AQUA),
               ("4", "Parameter count ≠ cost", "VPT touches no weights but lengthens every sequence; LoRA's parameters vanish at inference; all PEFT pays most of the backward pass.", YELLOW),
               ("5", "Never select on test", "I caught and discarded one early sweep that did; selection moved to the official val split.", PINK)]
    for i, (num, hd, bd, col) in enumerate(lessons):
        x, y = Inches(0.6) + (i % 3) * Inches(4.15), Inches(1.5) + (i // 3) * Inches(2.7)
        rect(s, x, y, Inches(3.95), Inches(2.45), TINT, rounded=True)
        circ = s.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.25), y + Inches(0.25), Inches(0.5), Inches(0.5)); circ.fill.solid(); circ.fill.fore_color.rgb = col; circ.line.fill.background()
        text(s, x + Inches(0.25), y + Inches(0.25), Inches(0.5), Inches(0.5), num, size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, font=HEAD)
        text(s, x + Inches(0.9), y + Inches(0.22), Inches(2.9), Inches(0.56), hd, size=15, bold=True, color=SLATE, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + Inches(0.25), y + Inches(0.9), Inches(3.45), Inches(1.5), bd, size=13)
    text(s, Inches(8.9), Inches(4.3), Inches(3.8), Inches(2.3), [[("Compute used: ", {"bold": True, "color": SLATE}), ("about 10 GPU-hours on A100 for 30 main runs, 14 ablation runs and profiling, after pre-resizing images once so data loading never bottlenecked.", {})]], size=13, color=INK)
    notes(s, "Five lessons. The biggest one: a fair comparison is mostly about the protocol, and a cheap choice like halving the epochs can flip a conclusion without any warning.")

    # 14 conclusion (dark) ------------------------------------------------------------------------
    s = prs.slides.add_slide(blank); n += 1; rect(s, 0, 0, W, H, SLATE); title(s, "Conclusions and future work", dark=True); footer(s, n, dark=True)
    text(s, Inches(0.6), Inches(1.5), Inches(6.2), Inches(0.5), "Takeaways", size=20, bold=True, color=RGBColor(0xCA, 0xDC, 0xFC), font=HEAD)
    bullets(s, Inches(0.6), Inches(2.1), Inches(6.2), Inches(4.6), [
        "Under one protocol with each paper's own hyper-parameters, full fine-tuning is best (73.9); LoRA recovers 98% at 0.34% of the parameters, BitFit 95%, VPT-deep 94%.",
        "The shared head accounts for 53.8 points; the mechanisms add 16 to 20.",
        "PEFT cuts training memory by up to a third but time by at most 15%; only merged LoRA is free at inference.",
        "The training budget, prompt length and LoRA rank each change the ranking; a comparison must state them."], size=15, color=WHITE, gap=10)
    text(s, Inches(7.3), Inches(1.5), Inches(5.4), Inches(0.5), "Future work", size=20, bold=True, color=RGBColor(0xCA, 0xDC, 0xFC), font=HEAD)
    bullets(s, Inches(7.3), Inches(2.1), Inches(5.4), Inches(3.6), [
        "Smaller domain gaps (CUB, Flowers), where VPT's paper beats full fine-tuning.",
        "Self-supervised backbones (MAE, DINO), where VPT reports a different ordering.",
        "A matched small validation search per method, to separate method from tuning.",
        "Higher resolution and augmentation toward the published 80%+ on Aircraft."], size=15, color=WHITE, gap=10)
    text(s, Inches(7.3), Inches(5.9), Inches(5.4), Inches(0.9), [[("Code: ", {"bold": True, "color": RGBColor(0xCA, 0xDC, 0xFC)}), ("github.com/chenyil6/cs412-peft-vit", {"color": WHITE})], [("Thank you. Questions?", {"italic": True, "color": RGBColor(0x9A, 0xA4, 0xB2)})]], size=14)
    notes(s, "To conclude: full fine-tuning wins, LoRA is the best trade-off, the head is isolated, PEFT saves memory not time, and the budget and structural knobs have to be stated. Future work is listed on the right. Thank you.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True); prs.save(a.out); print(f"wrote {a.out} with {n} slides")


if __name__ == "__main__":
    main()
