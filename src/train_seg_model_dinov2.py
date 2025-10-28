import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from datetime import datetime



# --- 1. Load DINO-v2 Backbone ---

def get_dino_v2_backbone():
    # This returns a model directly, not a state_dict
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")  # ✅ this is already a model
    return backbone


class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        features = self.backbone.get_intermediate_layers(x, n=1)[0]  # (B, N, 768)
        feat_size = int(features.shape[1] ** 0.5)
        features = features.permute(0, 2, 1).reshape(B, 768, feat_size, feat_size)  # (B, 768, h, w)
        out = self.decoder(features)  # (B, num_classes, h, w)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)  # upscale to 1022x1022
        return out


# --- 3. Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transforms.Compose([
            transforms.Resize((1022, 1022)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        self.mask_transform = transforms.Resize((1022, 1022), interpolation=Image.NEAREST)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])
        img = self.transform(img)
        mask = self.mask_transform(mask)
        mask1 = np.array(mask)
        #print(np.unique(mask1))
        mask = torch.tensor(np.array(mask)//100, dtype=torch.long)
        return img, mask
    
    def __len__(self):
        return len(self.image_paths)

# --- 4. Training Loop ---
def train(model, dataloader,val_loader, test_loader, output_dir, device, num_epochs):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

   
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
        avg_train_loss = total_loss / len(dataloader)

        # ---- Validation ----
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")

        avg_loss, accuracy, avg_acc =  test(model, test_loader, device)
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_train_loss,
        }, os.path.join(output_dir,"checkpoint.pth"))
        
        torch.save(model.state_dict(), "dino_v2_segmentation_oct17.pth")
        


def test(model, test_loader, device):
    """Evaluate model on test data and return average loss + accuracy."""
    #model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0
    
    avg_acc = 0
    cnt = 0 

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            # Predictions
            nonzero = masks != 0  
            # count only nonzero regions
            correct += ((preds == masks) & nonzero).sum().item()
            total   += nonzero.sum().item()
            total += masks.numel()
            
            for cls in [1, 2, 3]:
                cnt =  cnt+1
                cls_mask = masks == cls
                correct_cls = ((preds == masks) & cls_mask).sum().item()
                total_cls   = cls_mask.sum().item()
                acc_cls     = correct_cls / total_cls if total_cls > 0 else 0
                avg_acc = avg_acc + acc_cls
                #print(f"Class {cls} accuracy: {acc_cls:.4f}")
                
    avg_acc = avg_acc/cnt
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy*100:.2f}%,  Test avg class Accuracy: {avg_acc*100:.2f}%")
    
    return avg_loss, accuracy, avg_acc


if __name__ == "__main__":   
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    num_epochs=	75
    
    output_dir = os.path.join("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/ALS/seg_runs",str(timestamp))
    os.makedirs(output_dir)
    # Replace these with your actual paths
    image_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/ALS/training_dataset/train_filtered/images"
    mask_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/ALS/training_dataset/train_filtered/labels"
    
    val_image_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/ALS/training_dataset/val_filtered/images"
    val_mask_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/ALS/training_dataset/val_filtered/labels"
    
    test_image_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/ALS/training_dataset/val_filtered/images"
    test_mask_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/ALS/training_dataset/val_filtered/labels"
    
    
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
    
    val_image_paths = sorted([os.path.join(val_image_dir, f) for f in os.listdir(val_image_dir)])
    val_mask_paths = sorted([os.path.join(val_mask_dir, f) for f in os.listdir(val_mask_dir)])
    
    test_image_paths = sorted([os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir)])
    test_mask_paths = sorted([os.path.join(test_mask_dir, f) for f in os.listdir(test_mask_dir)])

    train_dataset = SegmentationDataset(image_paths, mask_paths)
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    
    val_dataset = SegmentationDataset(val_image_paths, val_mask_paths)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=4)
    
    
    test_dataset = SegmentationDataset(test_image_paths, test_mask_paths)
    test_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    backbone = get_dino_v2_backbone()
   
    print(backbone)
   
    model = DINOv2SegmentationModel(backbone, num_classes=3)  # Update `num_classes` as needed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    train(model, dataloader, val_dataloader, test_dataloader, output_dir, device, num_epochs)

    
