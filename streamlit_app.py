import streamlit as st
import pandas as pd
import PIL.Image
import snowflake.connector

st.set_page_config(page_title="SnowTruck", page_icon=":truck:")

#st.image(snowtruck_logo)
title_container = st.beta_container()
fp = open("images/Snowtruck_Icon.jpeg","rb")
snowtruck_logo = PIL.Image.open(fp)

col1, col2 = st.beta_columns([1, 20])
with title_container:
    with col1:
        st.image(snowtruck_logo, width=64)
    with col2:
        st.markdown('<h1 style="color: blue;">SnowTruck</h1>',
                    unsafe_allow_html=True)
#st.title("SnowTruck:minibus:")

# connects to the snowflake account 
# if need to use the data there
conn = snowflake.connector.connect(**st.secrets["snowflake"])

# tabs
tab1,tab2,tab3,tab4,tab5 = st.tabs(["tab1","tab2","tab3","tab4","tab5"])

with tab1:
  st.write()

with tab2:
  st.write()

with tab3:
  st.write()

with tab4:
  st.write()

with tab5:
  st.write()
