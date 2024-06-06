# module_name, package_name, ClassName, method_name, 
# ExceptionName, function_name, GLOBAL_CONSTANT_NAME, 
# global_var_name, instance_var_name, function_parameter_name, local_var_name.

import streamlit as st
import numpy as np
import json
import keras
from glob import glob
from random import choice
from mtcnn import MTCNN
import tensorflow as tf
import cv2

st.title('Get haircut recommendations')
top = st.container()
left_column, right_column = st.columns(2)
bottom = st.container()

def preprocess_image(image, img_size = (150, 150)):
        # image = image.astype('float32')
        detector = MTCNN()
        min_conf = 0.9
        offset = 20
        new_batch = []

        h,w,ch = image.shape
        area = 0
        final_face = None
        detections = detector.detect_faces(image)

        # transform only face with the biggest area 
        for det in detections:
            if det['confidence'] >= min_conf:
                x, y, width, height = det['box']
                object = image[max(y-offset,0):min(y+height+offset,h), max(0,x-offset):min(w,x+width+offset), :]
                object_area = object.shape[0]*object.shape[1]
                if (object_area > area):
                    area = object_area
                    final_face = object
        final_face = cv2.resize(final_face, img_size)
        new_batch.append(final_face.astype(int))
        results_tensor = tf.stack(new_batch)
        return results_tensor

def get_face_shape(batched_img):
    
    @st.cache_resource
    def load_model():
        return keras.saving.load_model("face_shape/models_weights/fine_tune_block6_aug.keras")

    model = load_model()
    class_names = ['heart', 'oblong', 'oval', 'round', 'square']

    predicted_batch = model.predict(batched_img)
    predicted_id = np.argmax(predicted_batch, axis=1)

    return class_names[predicted_id[0]]

@st.cache_resource
def load_recommendations():
        try:
            with open("hair_cut/recommendationPrompts.json") as stream:
                try:
                    return json.load(stream)
                except ValueError:  # includes simplejson.decoder.JSONDecodeError
                    st.text('Decoding JSON has failed')
        except FileNotFoundError:
            st.text('This file does not exist, try again!')

def recommend(face_img):
        processed_face = preprocess_image(face_img)
        face_shape = get_face_shape(processed_face)
        recommendations = load_recommendations()
        if recommendations is not None:
            return recommendations[face_shape]
        else:
            return None


def main():
  with top:
    if top.button("Get Recommendations", type="primary"):
        recommendations = None
        random_file = None
        with st.spinner('Your faceshape is analysed...'):
            random_file = choice(glob(f'face_shape/shapeofyou-2/test/**/*.jpg'))
            face_img = cv2.cvtColor(cv2.imread(random_file), cv2.COLOR_BGR2RGB)
            left_column.image(face_img, caption='For test purposes:  '+random_file.split('/')[-2])
            recommendations = recommend(face_img)
        if recommendations is not None:
            top.subheader(f"Congratulations! You have a {recommendations['faceShape']} shape!", divider='rainbow')
            # congrats_text = f"Congratulations! You have a {recommendations['faceShape']} shape!"
            does = '#### Do\'s\n\n'+('\n\n').join(recommendations['does'])
            right_column.success(does)
            donts = '#### Don\'ts\n\n'+('\n\n').join(recommendations['donts'])
            right_column.error(donts)
            for length, cuts in recommendations['haircut'].items():
              bottom.subheader(length)
              for cut in cuts:
                bottom.text(cut)

if __name__ == "__main__":
    main()