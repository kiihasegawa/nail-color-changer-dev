import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("AI Nail Color Change")

uploaded_file = st.file_uploader(
    "画像をアップロード",
    type=["jpg", "jpeg", "png"]
)

# カラーパレット
colors = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Pink": (255, 0, 255),
    "Yellow": (0, 255, 255),
    "White": (255, 255, 255),
    "Black": (20, 20, 20),
}

selected_color = st.selectbox(
    "ネイルカラーを選択",
    list(colors.keys())
)

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    # RGB → BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # =========================
    # 爪っぽい領域検出
    # =========================

    # 彩度高め
    s = hsv[:, :, 1]

    # 明るさ
    v = hsv[:, :, 2]

    mask1 = cv2.inRange(s, 50, 255)
    mask2 = cv2.inRange(v, 40, 255)

    mask = cv2.bitwise_and(mask1, mask2)

    # ノイズ除去
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # =========================
    # 輪郭抽出
    # =========================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    nail_mask = np.zeros(mask.shape, dtype=np.uint8)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        # 小さすぎ除外
        if area > 300:

            x, y, w, h = cv2.boundingRect(cnt)

            ratio = w / h

            # 爪っぽい形
            if 0.5 < ratio < 2.0:

                cv2.drawContours(
                    nail_mask,
                    [cnt],
                    -1,
                    255,
                    -1
                )

    # =========================
    # 色変更
    # =========================

    target_bgr = colors[selected_color]

    target = np.uint8([[target_bgr]])
    target_hsv = cv2.cvtColor(
        target,
        cv2.COLOR_BGR2HSV
    )[0][0]

    result_hsv = hsv.copy()

    mask_norm = nail_mask / 255.0

    # Hue変更
    result_hsv[:, :, 0] = (
        (1 - mask_norm) * result_hsv[:, :, 0]
        + mask_norm * target_hsv[0]
    )

    # 彩度少し強める
    result_hsv[:, :, 1] = np.clip(
        result_hsv[:, :, 1] * (1 + 0.3 * mask_norm),
        0,
        255
    )

    result_hsv = result_hsv.astype(np.uint8)

    result = cv2.cvtColor(
        result_hsv,
        cv2.COLOR_HSV2BGR
    )

    # 元画像少し混ぜる
    result = cv2.addWeighted(
        result,
        0.9,
        img,
        0.1,
        0
    )

    # 表示
    st.image(
        cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
        caption="Result"
    )

    st.image(
        nail_mask,
        caption="AI Nail Mask"
    )