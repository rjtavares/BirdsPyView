import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from PIL import Image, ImageDraw
from helpers import *
from pitch import FootballPitch
from itertools import product
from shapely.geometry import Polygon

st.set_option('deprecation.showfileUploaderEncoding', False)
image_to_open = st.sidebar.file_uploader("Upload Image:", type=["png", "jpg"])
pitch = FootballPitch()

if image_to_open:
    st.title('Pitch lines')
    st.sidebar.write('Draw Penalty Box lines (options below)')
    st.sidebar.image('pitch.png', width=300)
    
    image = Image.open(image_to_open)
    image = image.resize((600, int(600*image.height/image.width)))

    canvas_image = st_canvas(
        fill_color = "rgba(255, 165, 0, 0.3)", 
        stroke_width = 2,
        stroke_color = '#e00',
        background_image=image,
        width = image.width,
        height = image.height,
        drawing_mode= "line",
        key="canvas",
    )

    line_seq = ['UP','DP','RPA', 'RG']
    line_options = pitch.get_lines()

    lines = [st.selectbox(f'Line #{x+1}', line_options, key=f'line {x}', index=line_options.index(line_seq[x]))
             for x in range(4)]

    if canvas_image.json_data["objects"]:
        if len(canvas_image.json_data["objects"])>=4:
            df = pd.json_normalize(canvas_image.json_data["objects"])
            df['line'] = lines
            df['y1_line'] = df['top']+df['y1']
            df['y2_line'] = df['top']+df['y2']
            df['x1_line'] = df['left']+df['x1']
            df['x2_line'] = df['left']+df['x2']
            df['slope'], df['intercept'] = get_si_from_coords(df[['x1_line', 'y1_line', 'x2_line', 'y2_line']].values)
            df = df.set_index('line')

            vertical_lines = [x for x in lines if x in pitch.vert_lines]
            horizontal_lines = [x for x in lines if x in pitch.horiz_lines]
            intersections = {'_'.join([v, h]): line_intersect(df.loc[v, ['slope', 'intercept']], df.loc[h, ['slope', 'intercept']])
                             for v,h in product(vertical_lines, horizontal_lines)}

            pts_src = list(intersections.values())
            pts_dst = [pitch.get_intersections()[x] for x in intersections]

            h = Homography(pts_src, pts_dst)
            h_image = h.apply_to_image(image)

            st.title('Players')
            team_color = st.selectbox("Team color: ", ['red', 'blue'])
            if team_color == 'red':
                stroke_color='#e00'
            else:
                stroke_color='#00e'

            edit = st.checkbox('Edit mode (move selection boxes)')
            original = True #st.checkbox('Select on original image', value=True)
            update = st.button('Update data')
            image2 = image if original else Image.fromarray(h_image)
            height2 = image.height if original else 340
            width2 = image.width if original else 525
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
                        dfCoords['y'] = (dfCoords['top']+dfCoords['height']*dfCoords['scaleY'])
                        dfCoords['x'] = (dfCoords['left']+(dfCoords['width']*dfCoords['scaleX'])/2)
                        dfCoords[['x', 'y']] = h.apply_to_points(dfCoords[['x', 'y']].values)*np.array([100/525, 100/340])
                    else:
                        dfCoords['y'] = (dfCoords['top']+dfCoords['height']*dfCoords['scaleY'])/340*100
                        dfCoords['x'] = (dfCoords['left']+dfCoords['width']*dfCoords['scaleX'])/525*100
                    dfCoords['team'] = dfCoords.apply(lambda x: 'red' if x['stroke']=='#e00' else 'blue', axis=1)

                st.dataframe(dfCoords[['team', 'x', 'y']])
                pitch_polygon = Polygon(((0,0), (0,340), (525,340), (525,0)))
                camera_polygon = Polygon(h.apply_to_points(((0,0), (0,image.height), (image.width, image.height), (image.width,0))))
                
                show_original = st.checkbox('Show on original image', value=True)
                vor, dfVor = calculate_voronoi(dfCoords[['team', 'x', 'y']])
                final_image = image if show_original else Image.fromarray(h_image)
                for index, region in enumerate(vor.regions):
                    if not -1 in region:
                        if len(region)>0:
                            base_polygon = Polygon((np.vstack([vor.vertices[i] for i in region])/np.array([100/525, 100/340])).tolist())
                            polygon = camera_polygon.intersection(pitch_polygon).intersection(base_polygon)
                            if polygon.area>0:
                                if show_original:
                                    polygon = h.apply_to_points(np.vstack(polygon.exterior.xy).T, inverse=True)
                                else:
                                    polygon = np.vstack(polygon.exterior.xy).T
                                draw = ImageDraw.Draw(final_image, mode='RGBA')
                                color = dfVor[dfVor['region']==index]['team'].values[0]
                                if color == 'red':
                                    fill_color=(255,0,0,30)
                                else:
                                    fill_color=(0,0,255,30)
                                draw.polygon(list(tuple(point) for point in polygon.tolist()), fill=fill_color, outline='gray')
                            else:
                                st.write(polygon)
                st.image(final_image)

                if st.button('Save to disk'):
                    dfCoords[['team', 'x', 'y']].to_csv('output.csv')
                    st.info('Saved as output.csv')