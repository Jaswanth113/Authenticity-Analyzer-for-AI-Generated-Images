from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Load the saved model from the .h5 file
model = load_model('model_casia_run.h5')

# Generate the architecture diagram
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True, expand_nested=True)
