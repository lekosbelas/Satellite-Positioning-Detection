from data_loading import load_annotations
from model import SimpleCNN
annotations, images, categories = load_annotations('/Users/mr.nerdhuge/Downloads/sayzek_project/satelite/annotations/train.json')

num_classes = len(categories)

model = SimpleCNN(num_classes)