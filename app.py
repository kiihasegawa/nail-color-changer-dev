import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="French Nail AI", layout="centered")

st.title("French Nail AI 💅")
st.write("白フレンチ部分だけ色変更するデモ")

uploaded_file = st.file_uploader(
    "ネイル画像をアップロード",
    type=["jpg", "jpeg", "png"]
)

color = st.color_picker("変更色を選択", "#55c8ff")

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="元画像",
        use_container_width=True
    )

    if st.button("フレンチ部分の色を変更する"):

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 60, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)

        h, w = mask.shape

        # 上半分除外
        mask[:h//2, :] = 0

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        result = image.copy()

        hex_color = color.lstrip("#")
        bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

        color_layer = np.zeros_like(image)
        color_layer[:] = bgr

        alpha = 0.7

        result = np.where(
            mask[:, :, np.newaxis] == 255,
            cv2.addWeighted(image, 1-alpha, color_layer, alpha, 0),
            image
        )

        st.image(
            cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
            caption="変換後",
            use_container_width=True
        )
