import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="French Nail Color Change", layout="centered")

st.title("French Nail Color Changer 💅")
st.caption("白フレンチ部分だけ色変更する試作版")

uploaded_file = st.file_uploader(
    "画像をアップロード",
    type=["jpg", "jpeg", "png"]
)

target_color = st.color_picker(
    "フレンチ部分の変更後カラー",
    "#55bfff"
)

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return np.array([b, g, r], dtype=np.uint8)

def change_french_white(img_rgb, target_hex):
    image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 白っぽい部分を検出
    lower_white = np.array([0, 0, 165])
    upper_white = np.array([180, 70, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # フレンチ先端っぽい白だけ残す
    tip_mask = np.zeros_like(white_mask)

    contours, _ = cv2.findContours(
        white_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # 小さすぎるノイズ除外
        if area < 120:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # 画像の下側にある白っぽい部分は除外
        if y > image.shape[0] * 0.6:
            continue

        if h == 0:
            continue

        ratio = w / h

        # フレンチ先端は横長〜四角っぽい
        if ratio < 0.8 or ratio > 5.0:
            continue

        # 縦長すぎる白は除外
        if h > 90:
            continue

        cv2.drawContours(tip_mask, [cnt], -1, 255, -1)

    # マスク調整
    kernel = np.ones((5, 5), np.uint8)
    tip_mask = cv2.morphologyEx(tip_mask, cv2.MORPH_CLOSE, kernel)
    tip_mask = cv2.GaussianBlur(tip_mask, (5, 5), 0)

    # 変更後カラー
    target_bgr = hex_to_bgr(target_hex)

    target_hsv = cv2.cvtColor(
        np.uint8([[target_bgr]]),
        cv2.COLOR_BGR2HSV
    )[0][0]

    # 色変更
    result_hsv = hsv.copy()
    mask_norm = tip_mask / 255.0

    result_hsv[:, :, 0] = (
        (1 - mask_norm) * result_hsv[:, :, 0]
        + mask_norm * target_hsv[0]
    )

    result_hsv[:, :, 1] = (
        (1 - mask_norm) * result_hsv[:, :, 1]
        + mask_norm * target_hsv[1]
    )

    result_hsv = result_hsv.astype(np.uint8)
    result = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)

    # ツヤ保持
    result = cv2.addWeighted(result, 0.9, image, 0.1, 0)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return result_rgb, tip_mask

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    result_rgb, mask = change_french_white(img_rgb, target_color)

    st.subheader("Before / After")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img_rgb, caption="Before", use_container_width=True)

    with col2:
        st.image(result_rgb, caption="After", use_container_width=True)

    with st.expander("検出マスクを確認"):
        st.image(mask, caption="French mask", use_container_width=True)

    buffer = BytesIO()
    Image.fromarray(result_rgb).save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        "完成画像を保存",
        data=buffer,
        file_name="french_nail_changed.png",
        mime="image/png"
    )
