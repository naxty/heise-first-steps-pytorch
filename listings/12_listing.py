from PIL import Image
def predict(model, transform_function, image_path):
    image = Image.open(image_path)
    transformed = transform_function(image).float().unsqueeze(0)
    predicted = model(transformed)
    probabilities = F.softmax(predicted, dim=1)
    return probabilities
predict(fruit_model, data_transform, "data/evaluation/apple.jpg")