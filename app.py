import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="French Nail Color Changer", layout="centered")

st.title("French Nail Color Changer 💅")
st.caption("軽量版：白フレンチ部分の色変更デモ")

uploaded_file = st.file_uploader(
    "ネイル画像をアップロード",
    type=["jpg", "jpeg", "png"]
)

target_color = st.color_picker(
    "変更カラー",
    "#55bfff"
)

show_mask = st.checkbox("検出マスクを表示する", value=True)

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return np.array([b, g, r], dtype=np.uint8)

def change_french_color(image_rgb, target_hex):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 白フレンチ候補
    lower_white = np.array([0, 0, 165], dtype=np.uint8)
    upper_white = np.array([180, 75, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 画像上部の背景・ライト反射を少し除外
    h, w = mask.shape
    mask[: int(h * 0.15), :] = 0

    # ノイズ除去・穴埋め
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8)
    )

    # 細長い反射や大きすぎる領域を除外
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]

        if area < 80:
            continue

        if area > image_rgb.shape[0] * image_rgb.shape[1] * 0.18:
            continue

        ratio = ww / hh if hh > 0 else 0
        thin_ratio = max(ww, hh) / min(ww, hh) if min(ww, hh) > 0 else 999

        # 極端に細長い反射を除外
        if thin_ratio > 5.5:
            continue

        # フレンチ先端は横長〜やや四角寄りを想定
        if ratio < 0.25 or ratio > 8.0:
            continue

        clean_mask[labels == i] = 255

    clean_mask = cv2.dilate(
        clean_mask,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    clean_mask = cv2.GaussianBlur(
        clean_mask,
        (9, 9),
        0
    )

    target_bgr = hex_to_bgr(target_hex)

    target_hsv = cv2.cvtColor(
        np.uint8([[target_bgr]]),
        cv2.COLOR_BGR2HSV
    )[0][0]

    result_hsv = hsv.copy()
    mask_norm = clean_mask / 255.0

    color_strength = 0.85
    saturation_strength = 0.75

    result_hsv[:, :, 0] = (
        (1 - mask_norm * color_strength) * result_hsv[:, :, 0]
        + (mask_norm * color_strength) * target_hsv[0]
    )

    result_hsv[:, :, 1] = (
        (1 - mask_norm * saturation_strength) * result_hsv[:, :, 1]
        + (mask_norm * saturation_strength) * target_hsv[1]
    )

    result_hsv = result_hsv.astype(np.uint8)
    result_bgr = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)

    # ツヤを少し残す
    result_bgr = cv2.addWeighted(result_bgr, 0.88, image_bgr, 0.12, 0)

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return result_rgb, clean_mask

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image_pil)

    st.image(image_rgb, caption="元画像", width="stretch")

    if st.button("フレンチ部分の色を変更する"):
        with st.spinner("変換中です..."):
            result_rgb, mask = change_french_color(image_rgb, target_color)

        st.image(result_rgb, caption="変換結果", width="stretch")

        if show_mask:
            st.image(mask, caption="検出マスク", width="stretch")
