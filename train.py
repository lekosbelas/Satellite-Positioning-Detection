import torch
from torch.optim import Adam
import traceback

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    print(f"Training will run for {num_epochs} epochs.\nModel is using {device}.\n")

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}] starting...")

        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader, 1):
            print(f"\nProcessing Batch {batch_idx}...")
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            try:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                running_loss += losses.item()

                print(f"Batch {batch_idx}: Loss: {losses.item():.4f}")

            except Exception as e:
                print(f"An error occurred during batch {batch_idx}: {e}")
                traceback.print_exc()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {epoch_loss:.4f}\n")

    print("Training completed.\n")
