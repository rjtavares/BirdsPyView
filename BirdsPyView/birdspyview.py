import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from PIL import Image
from helpers import calculate_homography, apply_homography_to_image, line_intersect, get_si_from_coords

st.set_option('deprecation.showfileUploaderEncoding', False)
image_to_open = st.sidebar.file_uploader("Upload Image:", type=["png", "jpg"])

if image_to_open:
    st.title('Pitch lines')
    st.sidebar.write('Draw Penalty Box lines in the order shown below:')
    st.sidebar.image('pitch.png', width=300)
    
    image = Image.open(image_to_open)
    image = image.resize((500, int(500*image.height/image.width)))

    # Create a canvas component to draw pitch lines
    canvas_image = st_canvas(
        fill_color = "rgba(255, 165, 0, 0.3)", 
        stroke_width = 2,
        stroke_color = "#e00",
        background_image=image,
        width = image.width,
        height = image.height,
        drawing_mode= "line",
        key="canvas",
    )

    if canvas_image.json_data["objects"]:
        if len(canvas_image.json_data["objects"])>=4:
            df = pd.json_normalize(canvas_image.json_data["objects"])
            df['y1_line'] = df['top']+df['y1']
            df['y2_line'] = df['top']+df['y2']
            df['x1_line'] = df['left']+df['x1']
            df['x2_line'] = df['left']+df['x2']
            df['slope'], df['intercept'] = get_si_from_coords(df[['x1_line', 'y1_line', 'x2_line', 'y2_line']].values)

            UP_PA = line_intersect(df.loc[0, ['slope', 'intercept']], df.loc[2, ['slope', 'intercept']])
            DP_PA = line_intersect(df.loc[1, ['slope', 'intercept']], df.loc[2, ['slope', 'intercept']])
            UP_G = line_intersect(df.loc[0, ['slope', 'intercept']], df.loc[3, ['slope', 'intercept']])
            DP_G = line_intersect(df.loc[1, ['slope', 'intercept']], df.loc[3, ['slope', 'intercept']])

            pts_src = (UP_PA, DP_PA, UP_G, DP_G)
            pts_dst = ((82.5, 69.2), (82.5, 270.8), (0, 69.2), (0, 270.8))
            
            h,out = calculate_homography(pts_src, pts_dst)
            h_image = apply_homography_to_image(h, image)

            st.title('Players')
            team_color = st.radio("Team color: ", ['red', 'blue'])
            if team_color == 'red':
                stroke_color='#e00'
            else:
                stroke_color='#00e'

            # Create a canvas component to draw pitch lines
            canvas_converted = st_canvas(
                fill_color = "rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width = 5,
                stroke_color = stroke_color,
                background_image=Image.fromarray(h_image),
                drawing_mode= "line",
                height=340,
                width=525,
                key="canvas2",
            )

            if canvas_converted.json_data["objects"]:
                dfCoords = pd.json_normalize(canvas_converted.json_data["objects"])
                dfCoords['y'] = dfCoords['top']/340*100
                dfCoords['x'] = dfCoords['left']/525*100
                dfCoords['team'] = dfCoords.apply(lambda x: 'red' if x['stroke']=='#e00' else 'blue', axis=1)
                if st.button('Preview'):
                    st.dataframe(dfCoords[['team', 'x', 'y']])

                if st.button('Save to disk'):
                    dfCoords[['team', 'x', 'y']].to_csv('output.csv')
                    st.warning('Saved as output.csv')