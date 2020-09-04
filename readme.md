First alpha version of BirdsPyView, a Streamlit app to transform perspective of an image to a top-down view by identifying a rectangle on the ground, built to collect data on football matches.

# Installation

First install **Streamlit** and **Streamlit Drawable Canvas**:

    pip install streamlit
    pip install streamlit-drawable-canvas

Then clone the repo and run:

    streamlit run birdspyview.py

# Getting started

1. Upload an image. All the lines from the Penalty Box must be visible, but the whole line is not necessary.

2. Draw a line over each of the Penalty Box lines in the order shown below. You don't need to cover the whole line.

![](BirdsPyView/pitch.png?raw=true)

3. If the converted image is clear, draw lines over each player feet. The output coordinate is the average position of the line.

4. If you need to start over, refresh the page.

# Demo

![](demo.gif?raw=true)