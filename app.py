import streamlit as st
import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

st.set_page_config(page_title="French Nail AI", layout="centered")

st.title("French Nail AI 💅")
st.caption("白フレンチ部分だけ色変更するデモ")

# ===== SAM =====
@st.cache_resource
def load_sam():
    sam = sam_model_registry["vit_b"](
        checkpoint="sam_vit_b_01ec64.pth"
    )

    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.88,
        min_mask_region_area=200
    )

    return generator

mask_generator = load_sam()

# ===== UI =====
uploaded_file = st.file_uploader(
    "ネイル画像をアップロード",
    type=["jpg", "jpeg", "png"]
)

target_color = st.color_picker(
    "変更カラー",
    "#55bfff"
)

if uploaded_file is not None:

    image_pil = Image.open(uploaded_file).convert("RGB")
    image = np.array(image_pil)

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV
    )

    st.image(image, caption="元画像")

    with st.spinner("AI解析中..."):

        # ===== SAM =====
        masks = mask_generator.generate(image)

        nail_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )

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

        # ===== 白候補 =====
        lower_white = np.array([0, 0, 180], dtype=np.uint8)
        upper_white = np.array([180, 35, 255], dtype=np.uint8)

        white_mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white
        )

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
            np.ones((7, 7), np.uint8)
        )

        white_mask = cv2.dilate(
            white_mask,
            np.ones((3, 3), np.uint8),
            iterations=1
        )

        # ===== connected components =====
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            white_mask,
            connectivity=8
        )

        french_mask = np.zeros_like(white_mask)

        for i in range(1, num_labels):

            x, y, w, h, area = stats[i]

            if area < 120:
                continue

            if area > 18000:
                continue

            ratio = w / h if h > 0 else 0

            if ratio < 0.25 or ratio > 8.0:
                continue

            thin_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 999

            if thin_ratio > 3.5:
                continue

            if h > w * 2.2:
                continue

            french_mask[labels == i] = 255

        french_mask = cv2.GaussianBlur(
            french_mask,
            (9, 9),
            0
        )

        # ===== 色変更 =====
        target_rgb = tuple(
            int(target_color[i:i+2], 16)
            for i in (1, 3, 5)
        )

        target_bgr = np.array(
            [target_rgb[2], target_rgb[1], target_rgb[0]],
            dtype=np.uint8
        )

        target_hsv = cv2.cvtColor(
            np.uint8([[target_bgr]]),
            cv2.COLOR_BGR2HSV
        )[0][0]

        result_hsv = hsv.copy()

        mask_norm = french_mask / 255.0

        color_strength = 0.85
        saturation_strength = 0.75

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

        result_bgr = cv2.cvtColor(
            result_hsv,
            cv2.COLOR_HSV2BGR
        )

        result_bgr = cv2.addWeighted(
            result_bgr,
            0.9,
            image_bgr,
            0.1,
            0
        )

        result_rgb = cv2.cvtColor(
            result_bgr,
            cv2.COLOR_BGR2RGB
        )

    st.image(
        french_mask,
        caption="検出マスク"
    )

    st.image(
        result_rgb,
        caption="変換結果"
    )
