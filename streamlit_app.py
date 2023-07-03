import streamlit as st
import pandas as pd
import snowflake.connector

st.set_page_config(page_title="SnowTruck", page_icon=":truck:")

st.title("SnowTruck")

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
