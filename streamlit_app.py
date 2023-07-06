import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image
import snowflake.connector
import pydeck as pdk
import pickle

st.set_page_config(page_title="SnowTruck", page_icon=":truck:")

#st.image(snowtruck_logo)
fp = open("images/Snowtruck_ProductBox.jpeg","rb")
snowtruck_logo = PIL.Image.open(fp)

st.image(snowtruck_logo,width=64)
st.title("SnowTruck:minibus:")

# connects to the snowflake account 
# if need to use the data there
conn = snowflake.connector.connect(**st.secrets["snowflake"])

# tabs
tab1,tab2,tab3,tab4,tab5 = st.tabs(["tab1","tab2","tab3","tab4","tab5"])

with tab1:
  st.write()

with tab2:

  # Define the app title and favicon
    st.title('What are the top-selling food items by menu?') 
    st.markdown("This tab allows you to make predictions on the price of a listing based on the neighbourhood and room type. The model used is a Random Forest Regressor trained on the Airbnb Singapore listings dataset.")
    st.write('Choose a neighborhood group, neighborhood, and room type to get the predicted average price.')
    
    with open('xgb_xinle.pkl', 'rb') as file:
        xgb = pickle.load(file)

    # Load the cleaned and transformed dataset
    df = pd.read_csv('df_xinle.csv')
    price = df[['TOTAL_QUANTITY']] # extract price column from listings_new2.csv

    dowmappings={'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
    dow_reverse_mapping = {v: k for k, v in dowmappings.items()}
    dow_labels = list(dowmappings.keys())
    

with tab3:
  st.write()

with tab4:
  st.write()

with tab5:
  st.write()
