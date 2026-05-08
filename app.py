import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("ネイルカラーチェンジャー 💅　dev")

uploaded_file = st.file_uploader(
    "画像をアップロード",
    type=["jpg", "jpeg", "png"]
)

color = st.color_picker(
    "変えたい色を選んでください",
    "#ff69b4"
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ===== ネイビー部分を検出 =====
    lower_navy = np.array([90, 60, 20])
    upper_navy = np.array([140, 255, 180])
    navy_mask = cv2.inRange(hsv, lower_navy, upper_navy)

    # ===== マスク調整 =====
    nail_mask = navy_mask.copy()

    # 穴を埋める
    kernel = np.ones((5, 5), np.uint8)
    nail_mask = cv2.morphologyEx(nail_mask, cv2.MORPH_CLOSE, kernel)

    # 少し広げる
    kernel = np.ones((5, 5), np.uint8)
    nail_mask = cv2.dilate(nail_mask, kernel, iterations=1)

    # 境界をなめらかに
    nail_mask = cv2.GaussianBlur(nail_mask, (5, 5), 0)

    # ===== 色変換 =====
    hex_color = color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    target_bgr = np.array([b, g, r], dtype=np.uint8)

    target_hsv = cv2.cvtColor(
        np.uint8([[target_bgr]]),
        cv2.COLOR_BGR2HSV
    )[0][0]

    result_hsv = hsv.copy()
    mask_norm = nail_mask / 255.0

    # 色相を変更
    result_hsv[:, :, 0] = (
        (1 - mask_norm) * result_hsv[:, :, 0]
        + mask_norm * target_hsv[0]
    )

    # 彩度を少し上げる
    result_hsv[:, :, 1] = np.clip(
        result_hsv[:, :, 1] * (1 + 0.25 * mask_norm),
        0,
        255
    )

    result_hsv = result_hsv.astype(np.uint8)
    result = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)

    # 元画像を少し混ぜてツヤを残す
    result = cv2.addWeighted(result, 0.9, img, 0.1, 0)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    st.subheader("完成イメージ")
    st.image(result_rgb, use_container_width=True)

    with st.expander("マスク確認"):
        st.image(nail_mask, caption="ネイル検出マスク", use_container_width=True)
