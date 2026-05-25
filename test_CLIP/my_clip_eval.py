import clip
import torch
import csv
import os
from PIL import Image
from collections import defaultdict
import numpy as np

# ── 경로 설정 ──────────────────────────────────────────────
IMAGE_DIR = '/mnt/ssd/hj/Robot-Learning/image_data/selected_rooms/val_256'
TEXT_CSV  = '/mnt/ssd/hj/Robot-Learning/text_data/text_eval_600.csv'
OUTPUT    = '/mnt/ssd/hj/Robot-Learning/result/clip_eval_result.txt'
CLASSES   = ['living_room', 'kitchen', 'bedroom', 'exercise_room']

# ── 모델 로드 ──────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
model, preprocess = clip.load('ViT-B/32', device=device)
model.eval()

# ── 이미지 로드 & 인코딩 ───────────────────────────────────
print('이미지 인코딩 중...')
image_features_list = []
image_labels = []

for cls in CLASSES:
    cls_dir = os.path.join(IMAGE_DIR, cls)
    for fname in sorted(os.listdir(cls_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img = preprocess(Image.open(os.path.join(cls_dir, fname)).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        image_features_list.append(feat.cpu())
        image_labels.append(cls)

image_features = torch.cat(image_features_list, dim=0)  # (N_img, D)
print(f'이미지: {len(image_labels)}개')

# ── 텍스트 로드 & 인코딩 ───────────────────────────────────
print('텍스트 인코딩 중...')
text_features_list = []
text_labels = []
text_sentences = []

text_difficulty = []

with open(TEXT_CSV, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tokens = clip.tokenize([row['sentence']], truncate=True).to(device)
        with torch.no_grad():
            feat = model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        text_features_list.append(feat.cpu())
        text_labels.append(row['class'])
        text_sentences.append(row['sentence'])
        text_difficulty.append(row['label'])

text_features = torch.cat(text_features_list, dim=0)  # (N_txt, D)
print(f'텍스트: {len(text_labels)}개')

# ── 유사도 행렬 계산 ───────────────────────────────────────
# sim[i, j] = image i 와 text j 의 cosine similarity
sim = (image_features @ text_features.T).numpy()  # (N_img, N_txt)

cls2idx = {c: i for i, c in enumerate(CLASSES)}

# ── 평가 1: 이미지 → 텍스트 (이미지를 발화로 분류) ─────────
# 각 이미지에 대해 클래스별 평균 유사도 → argmax
img_pred_labels = []
for i in range(len(image_labels)):
    cls_avg = {}
    for cls in CLASSES:
        idxs = [j for j, l in enumerate(text_labels) if l == cls]
        cls_avg[cls] = sim[i, idxs].mean()
    img_pred_labels.append(max(cls_avg, key=cls_avg.get))

img_correct = sum(p == g for p, g in zip(img_pred_labels, image_labels))
img_acc = img_correct / len(image_labels)

# ── 평가 2: 텍스트 → 이미지 (발화를 이미지로 분류) ─────────
txt_pred_labels = []
for j in range(len(text_labels)):
    cls_avg = {}
    for cls in CLASSES:
        idxs = [i for i, l in enumerate(image_labels) if l == cls]
        cls_avg[cls] = sim[idxs, j].mean()
    txt_pred_labels.append(max(cls_avg, key=cls_avg.get))

txt_correct = sum(p == g for p, g in zip(txt_pred_labels, text_labels))
txt_acc = txt_correct / len(text_labels)

# ── Confusion Matrix 계산 ──────────────────────────────────
def confusion_matrix(true_labels, pred_labels, classes):
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    c2i = {c: i for i, c in enumerate(classes)}
    for t, p in zip(true_labels, pred_labels):
        cm[c2i[t]][c2i[p]] += 1
    return cm

img_cm = confusion_matrix(image_labels, img_pred_labels, CLASSES)
txt_cm = confusion_matrix(text_labels, txt_pred_labels, CLASSES)

# ── 결과 출력 & 저장 ───────────────────────────────────────
lines = []
lines.append('=' * 60)
lines.append('CLIP Evaluation Results (ViT-B/32)')
lines.append('=' * 60)
lines.append(f'Image dir : {IMAGE_DIR}')
lines.append(f'Text CSV  : {TEXT_CSV}')
lines.append(f'Images    : {len(image_labels)} (100 per class)')
lines.append(f'Texts     : {len(text_labels)} (150 per class)')
lines.append('')

lines.append('[ 1. Image → Text Classification ]')
lines.append(f'  Overall Accuracy: {img_acc*100:.2f}% ({img_correct}/{len(image_labels)})')
lines.append('  Per-class Accuracy:')
for cls in CLASSES:
    idxs = [i for i, l in enumerate(image_labels) if l == cls]
    acc = sum(img_pred_labels[i] == cls for i in idxs) / len(idxs)
    lines.append(f'    {cls:20s}: {acc*100:.1f}%')
lines.append('')
lines.append('  Confusion Matrix (rows=true, cols=pred):')
lines.append('  ' + ' '.join(f'{c[:10]:>12}' for c in CLASSES))
for i, cls in enumerate(CLASSES):
    lines.append(f'  {cls[:10]:10s}  ' + ' '.join(f'{img_cm[i][j]:12d}' for j in range(len(CLASSES))))
lines.append('')

lines.append('[ 2. Text → Image Classification ]')
lines.append(f'  Overall Accuracy: {txt_acc*100:.2f}% ({txt_correct}/{len(text_labels)})')
lines.append('  Per-class Accuracy:')
for cls in CLASSES:
    idxs = [j for j, l in enumerate(text_labels) if l == cls]
    acc = sum(txt_pred_labels[j] == cls for j in idxs) / len(idxs)
    lines.append(f'    {cls:20s}: {acc*100:.1f}%')
lines.append('')
lines.append('  Confusion Matrix (rows=true, cols=pred):')
lines.append('  ' + ' '.join(f'{c[:10]:>12}' for c in CLASSES))
for i, cls in enumerate(CLASSES):
    lines.append(f'  {cls[:10]:10s}  ' + ' '.join(f'{txt_cm[i][j]:12d}' for j in range(len(CLASSES))))
lines.append('')

lines.append('[ 3. Text → Image: Accuracy by Difficulty (label) ]')
DIFFICULTIES = ['direct', 'indirect', 'ambiguous']
for diff in DIFFICULTIES:
    idxs = [j for j, d in enumerate(text_difficulty) if d == diff]
    if not idxs:
        continue
    correct = sum(txt_pred_labels[j] == text_labels[j] for j in idxs)
    acc = correct / len(idxs)
    lines.append(f'  {diff:12s}: {acc*100:.1f}%  ({correct}/{len(idxs)})')
lines.append('')

lines.append('[ 4. Text → Image: Accuracy by Class × Difficulty ]')
header_row = f'  {"class":20s}' + ''.join(f'{d:>12}' for d in DIFFICULTIES)
lines.append(header_row)
for cls in CLASSES:
    row_str = f'  {cls:20s}'
    for diff in DIFFICULTIES:
        idxs = [j for j, (l, d) in enumerate(zip(text_labels, text_difficulty)) if l == cls and d == diff]
        if idxs:
            acc = sum(txt_pred_labels[j] == cls for j in idxs) / len(idxs)
            row_str += f'{acc*100:>11.1f}%'
        else:
            row_str += f'{"N/A":>12}'
    lines.append(row_str)
lines.append('=' * 60)

result_str = '\n'.join(lines)
print(result_str)

with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write(result_str + '\n')

print(f'\n결과 저장: {OUTPUT}')
