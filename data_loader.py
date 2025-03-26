"""
from dataset import CustomDataset
from data_loading import load_annotations

# Verilerin yüklenmesi
annotations, images, categories = load_annotations('/Users/mr.nerdhuge/Downloads/sayzek_project/satelite/annotations/train.json')
image_folder = '/Users/mr.nerdhuge/Downloads/sayzek_project/satelite/images/'


# Dataset ve DataLoader'ı test etmek için
train_dataset = CustomDataset(images, annotations, image_folder)
print(f"Dataset uzunluğu: {len(train_dataset)}")

# İlk veriyi çekip test etme
first_image, first_target = train_dataset[0]
print("İlk görüntü boyutu:", first_image.shape)
print("İlk hedef:", first_target)

image, target = train_dataset[0]
print("Image shape:", image.shape)
print("Target:", target)

"""
