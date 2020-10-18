import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from PIL import Image, ImageDraw
from helpers import Homography, VoronoiPitch, PitchImage, get_polygon
from pitch import FootballPitch

st.set_option('deprecation.showfileUploaderEncoding', False)
st.beta_set_page_config(page_title='BirdsPyView', layout='wide')
st.title('Upload Image')
image_to_open = st.file_uploader("Select Image file to open:", type=["png", "jpg"])
pitch = FootballPitch()

if image_to_open:
    st.title('Pitch lines')

    lines_expander = st.beta_expander('Draw pitch lines on selected image (2 horizontal lines, then 2 vertical lines)',
                                      expanded=True)
    with lines_expander:
        col1, col2, col_, col3 = st.beta_columns([2,1,0.5,1])
        image = PitchImage(image_to_open, pitch)

        with col1:
            canvas_image = st_canvas(
                fill_color = "rgba(255, 165, 0, 0.3)", 
                stroke_width = 2,
                stroke_color = '#e00',
                background_image = image.im,
                width = image.im.width,
                height = image.im.height,
                drawing_mode = "line",
                key = "canvas",
            )

        with col2:
            line_seq = ['UP','DP','RPA', 'RG']
            h_line_options = list(pitch.horiz_lines.keys())
            v_line_options = list(pitch.vert_lines.keys())

            hlines = [st.selectbox(f'Horizontal Line #{x+1}', h_line_options,
                      key=f'hline {x}', index=h_line_options.index(line_seq[x]))
                     for x in range(2)]
            vlines = [st.selectbox(f'Vertical Line #{x+1}', v_line_options,
                      key=f'vline {x}', index=v_line_options.index(line_seq[x+2]))
                     for x in range(2)]

        with col3: st.image('pitch.png', width=300)

    if canvas_image.json_data["objects"]:
        n_lines = len(canvas_image.json_data["objects"])
        with col3: st.write(f'You have drawn {n_lines} lines')
        if n_lines>=4:
            image.set_info(pd.json_normalize(canvas_image.json_data["objects"]), hlines+vlines)

            with lines_expander:
                st.write('Converted image:')
                st.image(image.conv_im)

            st.title('Players')
            st.write('Draw rectangle over players on image. '+
                     'The player location is assumed to the middle of the base of the rectangle.')

            p_col1, p_col2, p_col_, p_col3 = st.beta_columns([2,1,0.5,1])

            with p_col2:
                team_color = st.selectbox("Team color: ", ['red', 'blue'])
                if team_color == 'red':
                    stroke_color='#e00'
                else:
                    stroke_color='#00e'

                edit = st.checkbox('Edit mode (move selection boxes)')
                update = st.button('Update data')
                original = True #st.checkbox('Select on original image', value=True)

            image2 = image.im if original else image.conv_im
            height2 = image.im.height if original else image.conv_im.heigth
            width2 = image.im.width if original else image.conv_im.width
            with p_col1:
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

            if canvas_converted is not None:
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

                with p_col3:
                    st.write('Player Coordinates:')
                    st.dataframe(dfCoords[['team', 'x', 'y']])

                st.title('Final Output')
                voronoi = VoronoiPitch(dfCoords)
                opacity = int(st.slider('Opacity', 0, 100, value=30)*2.5)
                o_col1, o_col2 = st.beta_columns(2)
                with o_col1: st.image(image.apply_voronoi(voronoi, opacity, True))
                with o_col2: st.image(image.apply_voronoi(voronoi, opacity, False))

                if st.button('Save data to disk'):
                    dfCoords[['team', 'x', 'y']].to_csv('output.csv')
                    st.info('Saved as output.csv')