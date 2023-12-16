import gradio as gr
from fastai.vision.all import *

# Load the model downloaded from Kaggle.
learn = load_learner('model.pkl')

# Define a prediction function for the model.
labels = ('Polar', 'Temperate', 'Tropical')


def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    return dict(zip(labels, map(float, probs)))

# Define gradio inputs and outputs.
image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()

iface = gr.Interface(fn=predict, inputs=image, outputs=label)
iface.launch(share=True)
