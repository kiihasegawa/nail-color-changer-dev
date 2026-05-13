import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ===== 設定 =====
image_path = "nail french.jpg"
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

# 変更後カラー BGR
target_bgr = np.array([255, 180, 80], dtype=np.uint8)

# HSV許容
h_tolerance = 25
s_tolerance = 70
v_tolerance = 90

# 白フレンチ候補
lower_white = np.array([0, 0, 130], dtype=np.uint8)
upper_white = np.array([180, 130, 255], dtype=np.uint8)

# 色反映
color_strength = 0.85
saturation_strength = 0.75

# ===== 画像読み込み =====
image = cv2.imread(image_path)

if image is None:
    print("画像が見つかりません")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# ===== SAM =====
print("SAM読み込み中...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.88,
    min_mask_region_area=200
)

print("SAMでマスク生成中...")
masks = mask_generator.generate(image_rgb)

print(f"マスク候補数: {len(masks)}")

# ===== 爪全体マスク =====
nail_mask = np.zeros(image.shape[:2], dtype=np.uint8)

for m in masks:

    seg = m["segmentation"]
    area = np.sum(seg)

    if area < 500 or area > 25000:
        continue

    ys, xs = np.where(seg)

    if len(xs) == 0:
        continue

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    w = x_max - x_min + 1
    h = y_max - y_min + 1

    if h == 0:
        continue

    ratio = w / h
    fill_ratio = area / (w * h)

    if ratio < 0.35 or ratio > 2.8:
        continue

    if fill_ratio < 0.35:
        continue

    nail_mask[seg] = 255

kernel = np.ones((5, 5), np.uint8)

nail_mask = cv2.morphologyEx(
    nail_mask,
    cv2.MORPH_CLOSE,
    kernel
)

nail_mask = cv2.erode(
    nail_mask,
    np.ones((3, 3), np.uint8),
    iterations=1
)

cv2.imwrite("nail_mask.jpg", nail_mask)

# ===== 白候補 =====
white_mask = cv2.inRange(hsv, lower_white, upper_white)

white_mask = cv2.bitwise_and(
    white_mask,
    nail_mask
)

white_mask = cv2.morphologyEx(
    white_mask,
    cv2.MORPH_OPEN,
    np.ones((3, 3), np.uint8)
)

white_mask = cv2.morphologyEx(
    white_mask,
    cv2.MORPH_CLOSE,
    np.ones((5, 5), np.uint8)
)

white_mask = cv2.dilate(
    white_mask,
    np.ones((3, 3), np.uint8),
    iterations=1
)

white_mask = cv2.bitwise_and(
    white_mask,
    nail_mask
)

cv2.imwrite("white_candidate_mask.jpg", white_mask)

print("白フレンチ部分をクリックしてください")

# ===== HSV平均 =====
def hsv_mean_from_mask(mask):

    pixels = hsv[mask > 0]

    if len(pixels) == 0:
        return None

    return np.mean(pixels, axis=0)

# ===== HSV類似判定 =====
def is_similar_hsv(mean_hsv, target_hsv):

    if mean_hsv is None or target_hsv is None:
        return False

    h_diff = abs(mean_hsv[0] - target_hsv[0])
    h_diff = min(h_diff, 180 - h_diff)

    s_diff = abs(mean_hsv[1] - target_hsv[1])
    v_diff = abs(mean_hsv[2] - target_hsv[2])

    return (
        h_diff <= h_tolerance and
        s_diff <= s_tolerance and
        v_diff <= v_tolerance
    )

# ===== 色変更 =====
def change_clicked_color(x, y):
    print(f"クリック位置: x={x}, y={y}")
    print("今回はクリック位置に依存せず、white_mask全体からフレンチ候補を検出します")

    target_mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )

    # ===== SAM領域ごとに「白フレンチっぽい部分」を探す =====
    for m in masks:
        seg = m["segmentation"]

        temp_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )

        temp_mask[seg] = 255

        # 爪の中だけ
        temp_mask = cv2.bitwise_and(
            temp_mask,
            nail_mask
        )

        # 白候補の中だけ
        temp_mask = cv2.bitwise_and(
            temp_mask,
            white_mask
        )

        area = np.sum(temp_mask > 0)

        # 小さすぎるノイズ・大きすぎる爪全体を除外
        if area < 100 or area > 12000:
            continue

        ys, xs = np.where(temp_mask > 0)

        if len(xs) == 0:
            continue

        y_min = ys.min()
        y_max = ys.max()
        x_min = xs.min()
        x_max = xs.max()

        w = x_max - x_min + 1
        h = y_max - y_min + 1

        if h == 0:
            continue

        ratio = w / h

        # フレンチ先端は横長〜四角っぽいので、細長すぎるものを除外
        if ratio < 0.6 or ratio > 6.0:
            continue

        # 縦に広がりすぎるものは爪全体の可能性が高いので除外
        if h > image.shape[0] * 0.18:
            continue

        target_mask = cv2.bitwise_or(
            target_mask,
            temp_mask
        )

    selected_area = np.sum(target_mask > 0)
    print(f"選択領域サイズ: {selected_area}")

    if selected_area < 50:
        print("選択領域が小さすぎます")
        return image, target_mask

    # ===== マスク補正 =====
    target_mask = cv2.morphologyEx(
        target_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8)
    )

    target_mask = cv2.dilate(
        target_mask,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    target_mask = cv2.bitwise_and(
        target_mask,
        white_mask
    )

    target_mask = cv2.GaussianBlur(
        target_mask,
        (9, 9),
        0
    )

    # ===== 色変更 =====
    target_hsv = cv2.cvtColor(
        np.uint8([[target_bgr]]),
        cv2.COLOR_BGR2HSV
    )[0][0]

    result_hsv = hsv.copy()
    mask_norm = target_mask / 255.0

    result_hsv[:, :, 0] = (
        (1 - mask_norm * color_strength)
        * result_hsv[:, :, 0]
        + (mask_norm * color_strength)
        * target_hsv[0]
    )

    result_hsv[:, :, 1] = (
        (1 - mask_norm * saturation_strength)
        * result_hsv[:, :, 1]
        + (mask_norm * saturation_strength)
        * target_hsv[1]
    )

    result_hsv = result_hsv.astype(np.uint8)

    result = cv2.cvtColor(
        result_hsv,
        cv2.COLOR_HSV2BGR
    )

    result = cv2.addWeighted(
        result,
        0.88,
        image,
        0.12,
        0
    )

    cv2.imwrite("sam_clicked_mask.jpg", target_mask)
    cv2.imwrite("sam_clicked_result.jpg", result)

    print("保存しました")
    print("sam_clicked_mask.jpg")
    print("sam_clicked_result.jpg")

    return result, target_mask

# ===== クリック =====
def on_mouse(event, x, y, flags, param):

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    result, target_mask = change_clicked_color(x, y)

    cv2.imshow("result", result)
    cv2.imshow("mask", target_mask)

cv2.imshow(
    "click target color",
    image
)

cv2.setMouseCallback(
    "click target color",
    on_mouse
)

cv2.waitKey(0)
cv2.destroyAllWindows()
