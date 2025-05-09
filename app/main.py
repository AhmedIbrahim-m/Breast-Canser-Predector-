import streamlit as st
import pickle as pickle 
import pandas as pd 
import plotly.graph_objects as go
import numpy as np 

def get_scaled_values(input_dict):
  data = get_clean()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict
  
def  get_clean():
    data = pd.read_csv("data.csv")
  
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1 , 'B': 0})
    return data 
   
def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig

  

def sidebar():
    st.sidebar.header("cell Nuclei det")
    data = get_clean()
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    
    sample={}
    for label , key in slider_labels:
       sample[key] = st.sidebar.slider(
            label= label ,
            min_value=float(0)  ,
            max_value= float(data[key].max()) ,
            value=  float(data[key].mean())
            
        )
    return sample


def  add_pred(input):
    model = pickle.load(open("canser/model.pkl", "rb"))
    scaler = pickle.load(open("canser/scaler.pkl", "rb"))
    input_arr = np.array(list(input.values())).reshape(1,-1)
    input_arr_scale = scaler.transform(input_arr)
    
    
    pred = model.predict(input_arr_scale)
    if pred[0] == 0 :
        st.write("Benign".upper())
    else :
        st.write("Malicious".upper())
        
    
    st.write("The % of the Benign " , model.predict_proba(input_arr_scale)[0][0])
    st.write("The % of the Malicious " , model.predict_proba(input_arr_scale)[0][1])
        
    
    
def main():
    
    st.set_page_config(
        page_title= "Breast Canser Predictor",
        page_icon=":female-doctor:",
        layout="wide", 
        initial_sidebar_state="expanded"
)
    
    input =  sidebar()
   
    with st.container():
        st.title("Breast Canser Predictor")
        st.write("This web application uses Logistic Regression to predict whether a breast tumor is malignant (cancerous) or benign (non-cancerous) based on key features extracted from cell nuclei in breast mass")
    
    col1 , col2 , = st.columns([4,1]) 
    
    with col1:
         radar_chart = get_radar_chart(input)
         st.plotly_chart(radar_chart)
    
    with col2 :
       add_pred(input)
       
       
       
       
if __name__ == '__main__':
    main() 
