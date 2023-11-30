import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from helpers import Homography, VoronoiPitch, Play, PitchImage, PitchDraw, get_table_download_link
from pitch import FootballPitch

colors = {'black': '#000000',
          'blue': '#0000ff',
          'brown': '#a52a2a',
          'cyan': '#00ffff',
          'grey': '#808080',
          'green': '#008000',
          'magenta': '#ff00ff',
          'maroon': '#800000',
          'orange': '#ffa500',
          'pink': '#ffc0cb',
          'red': '#ff0000',
          'white': '#ffffff',
          'yellow': '#ffff00'}

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title='BirdsPyView', layout='wide')
st.title('Upload Image or Video')
uploaded_file = st.file_uploader("Select Image file to open:", type=["png", "jpg", "mp4"])
pitch = FootballPitch()

try:
    if uploaded_file:
        if uploaded_file.type == 'video/mp4':
            play = Play(uploaded_file)
            t = st.slider('You have uploaded a video. Choose the frame you want to process:', 0.0, 60.0)
            image = PitchImage(pitch, image=play.get_frame(t))
        else:
            image = PitchImage(pitch, image_bytes=uploaded_file)

        st.title('Pitch lines')

        lines_expander = st.expander('Draw pitch lines on selected image (2 horizontal lines, then 2 vertical lines)',
                                     expanded=True)
        with lines_expander:
            col1, col2, col_, col3 = st.columns([2, 1, 0.5, 1])

            with col1:
                canvas_image = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=2,
                    stroke_color='#e00',
                    background_image=image.im,
                    width=image.im.width,
                    height=image.im.height,
                    drawing_mode="line",
                    key="canvas",
                )

            with col2:
                line_seq = ['UP', 'DP', 'RPA', 'RG']
                h_line_options = list(pitch.horiz_lines.keys())
                v_line_options = list(pitch.vert_lines.keys())

                hlines = [st.selectbox(f'Horizontal Line #{x+1}', h_line_options,
                                       key=f'hline {x}', index=h_line_options.index(line_seq[x]))
                          for x in range(2)]
                vlines = [st.selectbox(f'Vertical Line #{x+1}', v_line_options,
                                       key=f'vline {x}', index=v_line_options.index(line_seq[x + 2]))
                          for x in range(2)]

            with col3:
                st.image('pitch.png', width=300)

        if canvas_image.json_data is not None:
            n_lines = len(canvas_image.json_data["objects"])
            with col3:
                st.write(f'You have drawn {n_lines} lines. Use the Undo button to delete lines.')
            if n_lines >= 4:
                image.set_info(pd.json_normalize(canvas_image.json_data["objects"]), hlines + vlines)

                with lines_expander:
                    st.write('Converted image:')
                    st.image(image.conv_im)

                st.title('Players')
                st.write('Draw rectangle over players on image. ' +
                         'The player location is assumed to be the middle of the base of the rectangle.')

                p_col1, p_col2, p_col_, p_col3 = st.columns([2, 1, 0.5, 1])

                with p_col2:
                    team_color = st.selectbox("Team color: ", list(colors.keys()))
                    stroke_color = colors[team_color]
                    edit = st.checkbox('Edit mode (move selection boxes)')
                    # update = st.button('Update data')
                    original = True  # st.checkbox('Select on original image', value=True)

                image2 = image.get_image(original)
                height2 = image2.height
                width2 = image2.width
                with p_col1:
                    canvas_converted = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)",
                        stroke_width=2,
                        stroke_color=stroke_color,
                        background_image=image2,
                        drawing_mode="transform" if edit else "rect",
                        # update_streamlit=update,
                        height=height2,
                        width=width2,
                        key="canvas2",
                    )

                if canvas_converted.json_data is not None:
                    if len(canvas_converted.json_data["objects"]) > 0:
                        dfCoords = pd.json_normalize(canvas_converted.json_data["objects"])
                        if original:
                            dfCoords['x'] = (dfCoords['left'] + (dfCoords['width'] * dfCoords['scaleX']) / 2)
                            dfCoords['y'] = (dfCoords['top'] + dfCoords['height'] * dfCoords['scaleY'])
                            dfCoords[['x', 'y']] = image.h.apply_to_points(dfCoords[['x', 'y']].values)
                        else:
                            dfCoords['x'] = (dfCoords['left'] + dfCoords['width'] * dfCoords['scaleX'])
                            dfCoords['y'] = (dfCoords['top'] + dfCoords['height'] * dfCoords['scaleY'])
                        dfCoords[['x', 'y']] = dfCoords[['x', 'y']] / image.h.coord_converter
                        dfCoords['team'] = dfCoords.apply(lambda x: {code: color for color, code in colors.items()}.get(x['stroke']),
                                                          axis=1)

                    
                    with p_col3:
                        st.write('Player Coordinates:')
                        df_output = dfCoords[['team', 'x', 'y']]
                        st.dataframe(df_output)

                    st.title('Final Output')
                    voronoi = VoronoiPitch(dfCoords)
                    sensitivity = int(st.slider("Sensitivity (decrease if it is drawing over the players; " +
                            "increase if the areas don't cover the whole pitch)"
                            , 0, 30, value=10) * 2.5)
                    o_col1, o_col2, o_col3 = st.columns((3, 1, 3))
                    with o_col2:
                        show_voronoi = st.checkbox('Show Voronoi', value=True)
                        voronoi_opacity = int(st.slider('Voronoi Opacity', 0, 100, value=20) * 2.5)
                        player_highlights = st.multiselect('Players to highlight', list(dfCoords.index+1))
                        player_size = st.slider('Circle size', 1, 10, value=2)
                        player_opacity = int(st.slider('Circle Opacity', 0, 100, value=50) * 2.5)
                    with o_col1:
                        draw = PitchDraw(image, original=True)
                        if show_voronoi:
                            draw.draw_voronoi(voronoi, image, voronoi_opacity)
                        for pid, coord in dfCoords.iterrows():
                            if pid in player_highlights:
                                draw.draw_circle(coord[['x', 'y']].values, coord['team'], player_size, player_opacity)
                        st.image(draw.compose_image(sensitivity))
                    with o_col3:
                        draw = PitchDraw(image, original=False)
                        for pid, coord in dfCoords.iterrows():
                            draw.draw_circle(coord[['x', 'y']].values, coord['team'], 2, player_opacity)
                            draw.draw_text(coord[['x', 'y']] + 0.5, f"{pid}", coord['team'])
                        st.image(draw.compose_image(sensitivity))

                    st.markdown(get_table_download_link(df_output), unsafe_allow_html=True)

                    # mplsoccer based output viz
                    _, mpl_output, _ = st.columns((1, 2, 1))  # temporary, as fig size doesn't seem to be working
                    from mplsoccer import Pitch
                    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b')
                    fig, ax = pitch.draw()
                    pitch.scatter(dfCoords.x, dfCoords.y, c=dfCoords.team, ax=ax, s=120, edgecolors='white')
                    with mpl_output:
                        st.pyplot(fig)

except NameError as ne:
    st.markdown("")
