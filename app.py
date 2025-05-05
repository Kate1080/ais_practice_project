import streamlit as st
from PIL import Image
from model import HorseDetect
from visualization import draw_bboxes
import numpy as np
from metrics import calculate_confidence_stats, conf_score_for_hist, box_area_for_hist, boxes_area, calculate_centers, centers_for_graph, density_grid, count_horses_over_time, zone_active
import tempfile
import cv2 as cv
import plotly.express as px
import matplotlib.pyplot as plt
from video_visual import save_annotated_video
import os
import pandas as pd
import base64

@st.cache_resource
def load_model():
    return HorseDetect( model_weights="yolov8n.pt" )

st.set_page_config(
    page_title ="Учет лошадей на ипподроме",
    page_icon="🐎",
    layout="wide"
)

st.markdown("""
<h1 style="text-align: center; 
background: linear-gradient(45deg, #002962, #DCE3EC);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
font-family: 'Arial Black', sans-serif;">
🏇 Horse registration at the hippodrome 🏇
</h1>
""", unsafe_allow_html=True)

st.divider()


with st.sidebar:
    st.title("🏇 Navigation")
    st.markdown("---")

    content_type = st.radio( 'Type of content', ( "📷 Images", "🎥 Video" ) )


if "📷 Images" in content_type:


    uploaded_file = st.file_uploader(
        "Upload horse image",
        type=["jpg", "jpeg", "png"],
        # help="Поддерживаемые форматы: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        try:
            st.subheader("Original image")

            image = Image.open(uploaded_file)
            image_np = np.array(image.convert('RGB'))[:, :, ::-1]

            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                st.image(
                    image,
                    caption="Uploaded image",
                    width=600,
                    use_container_width=False
                )

            st.divider()

            detector = load_model()
            detections = detector.detect( image_np )

            if len(detections) > 0:
                annotated_image = draw_bboxes( image_np, detections )

                annotated_image_rgb = annotated_image[:, :, ::-1]

                st.subheader( "Result" )
                col1, col2, col3 = st.columns([1, 6, 1])
                with col2:
                    st.image(
                        annotated_image_rgb,
                        caption=f"Founded horses: { len( detections ) }",
                        width=600,
                        use_container_width=False
                    )

                stats = calculate_confidence_stats(detections)

                st.subheader("Статистика")

                col1, col2, col3 = st.columns( 3 )
                col1.metric("Average confidence", f"{stats['mean']:.2f}")
                col2.metric("Min confidence", f"{stats['min']:.2f}")
                col3.metric("Max confidence", f"{stats['max']:.2f}")

                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    hist_df = conf_score_for_hist(stats["confidence_scores"])

                    fig_hist = px.bar(
                        hist_df,
                        x="bin_edges",
                        y="count",
                        labels={"bin_edges": "Уверенность", "count": "Количество"},
                        title="Model confidence",
                        color_discrete_sequence=["#636EFA"]
                    )

                    st.plotly_chart(fig_hist)

                col_left, col_center, col_right = st.columns([3, 1, 1])
                area_stats = boxes_area(detections)

                with col_left:
                    hist_df = box_area_for_hist(area_stats["areas"])

                    fig_hist = px.bar(
                        hist_df,
                        x="bin_edges",
                        y="count",
                        labels={"bin_edges": "Площадь (px²)", "count": "Количество"},
                        title="bounding boxes areas",
                        color_discrete_sequence=["#00CC96"]
                    )
                    st.plotly_chart(fig_hist)

                center_stats = calculate_centers(detections)
                col_l, col_c, col_r = st.columns([3, 1, 1])
                with col_l:
                    scatter_df = centers_for_graph(center_stats["centers"])

                    fig_scatter = px.scatter(
                        scatter_df,
                        x="x", y="y",
                        labels={"x": "X", "y": "Y"},
                        title="Objects centers",
                        range_x=[0, 640],
                        range_y=[0, 640],
                        width=600,
                        height=500,
                        color_discrete_sequence=["#AB63FA"]
                    )
                    st.plotly_chart(fig_scatter)

                density_grid = density_grid(center_stats["centers"])

                fig, ax = plt.subplots(figsize=(5, 5))  # <--- Ключевой параметр figsize

                im = ax.imshow(density_grid, cmap='YlOrRd', origin='upper')
                ax.set_title("Objects density")

                fig.colorbar(im, ax=ax, label='Objects amount')

                for i in range(density_grid.shape[0]):
                    for j in range(density_grid.shape[1]):
                        ax.text(j, i, f"{int(density_grid[i, j])}",
                                ha="center", va="center", color="black")
                plt.tight_layout()

                st.pyplot(fig)

            st.divider()

        except Exception as e:
            st.error(f"Ошибка обработки: {str(e)}")

elif "🎥 Video" in content_type:

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_video is not None:
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name

            cap = cv.VideoCapture(video_path)
            ret, frame = cap.read()
            if not ret:
                st.error("Не удалось прочитать видеофайл.")

            else:
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                st.subheader("Preveiw")
                col1, col2, col3 = st.columns([1, 4, 1])
                with col2:
                    st.image(
                        image,
                        width=600,
                        use_container_width=False
                    )

            fps = int(cap.get(cv.CAP_PROP_FPS))
            frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            cap.release()

                # progress_bar = st.progress(0)
                # status_text = st.empty()

            detector = HorseDetect(model_weights="yolov8n.pt")
            detections_by_frame = detector.detect_video(video_path)
            processed_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

            save_annotated_video(
                video_path,
                detections_by_frame,
                processed_video_path,
                fps=fps,
                skip_frames=0
            )

            if os.path.exists(processed_video_path):
                st.write(processed_video_path)
                st.write(f"Размер файла: {os.path.getsize(processed_video_path)} байт")
                st.video(processed_video_path)
            else:
                st.error("Ошибка: размеченное видео не создано.")

            st.subheader("Number of horses in the frame")
            stats = count_horses_over_time(detections_by_frame)

            col1, col2, col3 = st.columns(3)
            col1.metric("Average number", f"{stats['mean_count']:.1f}")
            col2.metric("Max number", stats['max_count'])
            col3.metric("Min number", stats['min_count'])


            df = pd.DataFrame({
                "Frame": range(len(stats['counts'])),
                "Number of horses": stats['counts']
            })

            fig = px.line(
                df,
                x="Frame",
                y="Number of horses",
                title="Number of horses on each frame",
                labels={"Frame": "Frame number", "Number of horses": "Horses on frame"}
            )

            fig.update_layout(
                template="plotly_white",
                xaxis_title="Frame number",
                yaxis_title="Number of horses",
                font=dict(family="Arial", size=12),
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            cap = cv.VideoCapture(video_path)
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            zone_activity = zone_active(detections_by_frame, width, height)

            col1, col2 = st.columns(2)

            df_zones = pd.DataFrame({
                "Zone": zone_activity.keys(),
                "Count": zone_activity.values()
            })

            fig_bar = px.bar(
                df_zones,
                x="Zone",
                y="Count",
                text="Count",
                color="Zone",
                title="Number of horses on each zone"
            )
            st.plotly_chart(fig_bar, use_container_width=True)


        except Exception as e:
            st.error(f"Ошибка при обработке видео: {str(e)}")

# st.write( "there should be a cool project...maybe..one day" )