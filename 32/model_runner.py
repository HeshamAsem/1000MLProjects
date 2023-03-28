# %%
from functions_mo import augment_images, load_data, encode_labels, split_data, scale_data, create_model, train_model

# Set the root directory for your data
root_dir =  '/home/mohammad/Documents/classification/cars classification/dataset/DATA/'
scaler_filename = "scaler.save"
string_to_delete = 'aug'
width = 224
height = 224

# Augment the images in the directory
augment_images(root_dir, width=224, height=224)

# Load the augmented images and their labels
X, y = load_data(root_dir)

# Encode the labels using one-hot encoding
y = encode_labels(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y)

# Scale the data using standard scaling
X_train, X_test = scale_data(X_train, X_test)

# create the model
model = create_model()

# train the model
train_model(model, X_train, y_train, X_test, y_test)


# %%



