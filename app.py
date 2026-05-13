import os
import urllib.request

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

st.set_page_config(page_title="French Nail AI", layout="centered")

st.title("French Nail AI 💅")
st.caption("白フレンチ部分だけ色変更するデモ")


@st.cache_resource
def load_sam():
    checkpoint = "sam_vit_b_01ec64.pth"

    if not os.path.exists(checkpoint):
        st.write("SAMモデルをダウンロード中です。初回だけ数分かかります。")
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            checkpoint
        )

    sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.88,
        min_mask_region_area=200
    )

    return generator


def hex_to_bgr(hex_color):
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return np.array([rgb[2], rgb[1], rgb[0]], dtype=np.uint8)


def change_french_color(image_rgb, target_bgr, mask_generator):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    masks = mask_generator.generate(image_rgb)

    nail_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

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

    nail_mask = cv2.morphologyEx(nail_mask, cv2.MORPH_CLOSE, kernel)
    nail_mask = cv2.erode(
        nail_mask,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    lower_white = np.array([0, 0, 130], dtype=np.uint8)
    upper_white = np.array([180, 130, 255], dtype=np.uint8)

    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_mask = cv2.bitwise_and(white_mask, nail_mask)

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

    white_mask = cv2.bitwise_and(white_mask, nail_mask)

    target_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    for m in masks:
        seg = m["segmentation"]

        temp_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        temp_mask[seg] = 255

        temp_mask = cv2.bitwise_and(temp_mask, nail_mask)
        temp_mask = cv2.bitwise_and(temp_mask, white_mask)

        area = np.sum(temp_mask > 0)

        if area < 100 or area > 12000:
            continue

        ys, xs = np.where(temp_mask > 0)
        if len(xs) == 0:
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        w = x_max - x_min + 1
        h = y_max - y_min + 1

        if h == 0:
            continue

        ratio = w / h

        if ratio < 0.6 or ratio > 6.0:
            continue

        if h > image_rgb.shape[0] * 0.18:
            continue

        target_mask = cv2.bitwise_or(target_mask, temp_mask)

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

    target_mask = cv2.bitwise_and(target_mask, white_mask)

    target_mask = cv2.GaussianBlur(
        target_mask,
        (9, 9),
        0
    )

    target_hsv = cv2.cvtColor(
        np.uint8([[target_bgr]]),
        cv2.COLOR_BGR2HSV
    )[0][0]

    result_hsv = hsv.copy()
    mask_norm = target_mask / 255.0

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

    result_bgr = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)

    result_bgr = cv2.addWeighted(
        result_bgr,
        0.88,
        image_bgr,
        0.12,
        0
    )

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return result_rgb, target_mask, white_mask, nail_mask


mask_generator = load_sam()

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
    image_rgb = np.array(image_pil)

    st.image(
        image_rgb,
        caption="元画像",
        use_container_width=True
    )

    if st.button("フレンチ部分の色を変更する"):
        with st.spinner("AI解析中です。少し待ってください..."):
            result_rgb, target_mask, white_mask, nail_mask = change_french_color(
                image_rgb,
                hex_to_bgr(target_color),
                mask_generator
            )

        st.image(
            result_rgb,
            caption="変換結果",
            use_container_width=True
        )

        with st.expander("検出マスクを確認"):
            st.image(
                target_mask,
                caption="最終マスク",
                use_container_width=True
            )

            st.image(
                white_mask,
                caption="白候補マスク",
                use_container_width=True
            )

            st.image(
                nail_mask,
                caption="爪候補マスク",
                use_container_width=True
            )
