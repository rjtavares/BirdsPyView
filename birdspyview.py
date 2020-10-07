import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from PIL import Image, ImageDraw
from helpers import Homography, VoronoiPitch, PitchImage, get_polygon
from pitch import FootballPitch

st.set_option('deprecation.showfileUploaderEncoding', False)
image_to_open = st.sidebar.file_uploader("Upload Image:", type=["png", "jpg"])
pitch = FootballPitch()

if image_to_open:
    st.title('Pitch lines')
    st.sidebar.write('Draw Penalty Box lines (options below)')
    st.sidebar.image('pitch.png', width=300)
    
    image = PitchImage(image_to_open, pitch)

    canvas_image = st_canvas(
        fill_color = "rgba(255, 165, 0, 0.3)", 
        stroke_width = 2,
        stroke_color = '#e00',
        background_image=image.im,
        width = image.im.width,
        height = image.im.height,
        drawing_mode= "line",
        key="canvas",
    )

    line_seq = ['UP','DP','RPA', 'RG']
    line_options = pitch.get_lines()

    lines = [st.selectbox(f'Line #{x+1}', line_options, key=f'line {x}', index=line_options.index(line_seq[x]))
             for x in range(4)]

    if canvas_image.json_data["objects"]:
        if len(canvas_image.json_data["objects"])>=4:
            image.set_info(pd.json_normalize(canvas_image.json_data["objects"]), lines)
            st.title('Players')
            team_color = st.selectbox("Team color: ", ['red', 'blue'])
            if team_color == 'red':
                stroke_color='#e00'
            else:
                stroke_color='#00e'

            edit = st.checkbox('Edit mode (move selection boxes)')
            original = True #st.checkbox('Select on original image', value=True)
            update = st.button('Update data')
            image2 = image.im if original else image.conv_im
            height2 = image.im.height if original else image.conv_im.heigth
            width2 = image.im.width if original else image.conv_im.width
            canvas_converted = st_canvas(
                fill_color = "rgba(255, 165, 0, 0.3)",
                stroke_width = 2,
                stroke_color = stroke_color,
                background_image = image2,
                drawing_mode = "transform" if edit else "rect",
                update_streamlit = update,
                height = height2,
                width = width2,
                key="canvas2",
            )

            if canvas_converted.json_data["objects"]:
                if len(canvas_converted.json_data["objects"])>0:
                    dfCoords = pd.json_normalize(canvas_converted.json_data["objects"])
                    if original:
                        dfCoords['x'] = (dfCoords['left']+(dfCoords['width']*dfCoords['scaleX'])/2)
                        dfCoords['y'] = (dfCoords['top']+dfCoords['height']*dfCoords['scaleY'])
                        dfCoords[['x', 'y']] = image.h.apply_to_points(dfCoords[['x', 'y']].values)
                    else:
                        dfCoords['x'] = (dfCoords['left']+dfCoords['width']*dfCoords['scaleX'])
                        dfCoords['y'] = (dfCoords['top']+dfCoords['height']*dfCoords['scaleY'])
                    dfCoords[['x', 'y']] = dfCoords[['x', 'y']]/image.coord_converter
                    dfCoords['team'] = dfCoords.apply(lambda x: 'red' if x['stroke']=='#e00' else 'blue', axis=1)

                st.dataframe(dfCoords[['team', 'x', 'y']])
                show_original = st.checkbox('Show on original image', value=True)
                voronoi = VoronoiPitch(dfCoords)
                final_image = image.im if show_original else image.conv_im
                for region in voronoi.get_regions():
                    polygon = get_polygon(voronoi.get_points_region(region)*image.coord_converter,
                                          image, show_original)
                    draw = ImageDraw.Draw(final_image, mode='RGBA')
                    color = voronoi.get_color_region(region)
                    if color == 'red':
                        fill_color=(255,0,0,30)
                    else:
                        fill_color=(0,0,255,30)
                    draw.polygon(list(tuple(point) for point in polygon.tolist()), fill=fill_color, outline='gray')
                st.image(final_image)

                if st.button('Save to disk'):
                    dfCoords[['team', 'x', 'y']].to_csv('output.csv')
                    st.info('Saved as output.csv')