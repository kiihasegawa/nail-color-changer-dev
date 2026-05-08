import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="AI Nail Color Change Dev", layout="centered")

st.title("AI Nail Color Change Dev 💅")
st.caption("複数色デザイン対応の試作版")

uploaded_file = st.file_uploader(
    "画像をアップロード",
    type=["jpg", "jpeg", "png"]
)

K = st.slider(
    "検出する色数",
    min_value=3,
    max_value=8,
    value=5
)

target_color = st.color_picker(
    "変更後の色",
    "#55bfff"
)

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return np.array([b, g, r], dtype=np.uint8)

def change_selected_cluster(img_rgb, K, selected_cluster, target_hex):
    image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    image_small = cv2.resize(image, (400, 240))

    pixels = image_small.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    labels_2d = labels.reshape(image_small.shape[:2])

    target_mask_small = np.uint8(labels_2d == selected_cluster) * 255

    target_mask = cv2.resize(
        target_mask_small,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    kernel = np.ones((5, 5), np.uint8)
    target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel)
    target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel)
    target_mask = cv2.GaussianBlur(target_mask, (5, 5), 0)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    target_bgr = hex_to_bgr(target_hex)
    target_hsv = cv2.cvtColor(
        np.uint8([[target_bgr]]),
        cv2.COLOR_BGR2HSV
    )[0][0]

    result_hsv = hsv.copy()
    mask_norm = target_mask / 255.0

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

    result = cv2.addWeighted(result, 0.9, image, 0.1, 0)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return result_rgb, target_mask, centers

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    # まずクラスタだけ作る
    image_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    image_small = cv2.resize(image_bgr, (400, 240))

    pixels = image_small.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)

    st.subheader("検出された色")

    cols = st.columns(K)

    for i, color in enumerate(centers):
        b, g, r = color
        hex_color = f"#{r:02x}{g:02x}{b:02x}"

        with cols[i]:
            st.color_picker(
                f"色 {i}",
                hex_color,
                disabled=True,
                key=f"cluster_color_{i}"
            )

    selected_cluster = st.selectbox(
        "変更したい色番号を選んでください",
        list(range(K))
    )

    result_rgb, target_mask, centers = change_selected_cluster(
        img_rgb,
        K,
        selected_cluster,
        target_color
    )

    st.subheader("Before / After")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img_rgb, caption="Before", use_container_width=True)

    with col2:
        st.image(result_rgb, caption="After", use_container_width=True)

    with st.expander("選択中のマスクを確認"):
        st.image(target_mask, caption="Selected cluster mask", use_container_width=True)

    result_image = Image.fromarray(result_rgb)
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        "完成画像を保存",
        data=buffer,
        file_name="nail_design_changed.png",
        mime="image/png"
    )
