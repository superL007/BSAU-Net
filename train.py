import os
import matplotlib.pyplot as plt
import numpy as np

from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image
import cv2  # Used for generating heatmaps

# Check if GPU is available, use GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths for model weights, data, and saving results
weight_path = ''
data_path = r''
save_path = ''
losses_file = ''  # Path to the file for recording loss values

# Define Dice loss function
def dice_loss(inputs, targets, smooth=1):
    intersection = torch.sum(inputs * targets)
    dice_coefficient = (2. * intersection + smooth) / (torch.sum(inputs) + torch.sum(targets) + smooth)
    loss = 1 - dice_coefficient
    return loss

# Define PPV (Positive Predictive Value) calculation
def ppv(inputs, targets, smooth=1):
    true_positive = torch.sum(inputs * targets)
    predicted_positive = torch.sum(inputs)
    ppv = (true_positive + smooth) / (predicted_positive + smooth)
    return ppv.item()

# Define Dice coefficient calculation
def dice_coefficient(inputs, targets, smooth=1):
    intersection = torch.sum(inputs * targets)
    dice = (2. * intersection + smooth) / (torch.sum(inputs) + torch.sum(targets) + smooth)
    return dice.item()

# Define function to calculate sensitivity
def sensitivity(inputs, targets, smooth=1):
    true_positive = torch.sum(inputs * targets)
    actual_positive = torch.sum(targets)
    sensitivity = (true_positive + smooth) / (actual_positive + smooth)
    return sensitivity.item()

# Hook function to capture feature maps from intermediate layers
def hook_fn(module, input, output):
    # Store feature maps from the selected layer
    features.append(output)

if __name__ == '__main__':
    # Create data loader using a custom dataset class to load data
    data_loader = DataLoader(MyDataset(data_path), batch_size=1, shuffle=True)
    # Create an instance of the UNet model and move it to the device (GPU or CPU) for computation
    net = BSAUnet().to(device)

    # Load pre-trained weights if they exist
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('Weights loaded successfully!')
    else:
        print('Failed to load weights')

    # Register a hook to capture feature maps from a specific layer in the model
    features = []
    hook = net.encoder.layer4[1].register_forward_hook(hook_fn)  # Example: Capture feature maps from layer4

    # Define optimizer and loss function
    opt = optim.Adam(net.parameters())
    loss_fun = dice_loss

    epochs = 100
    losses = []
    dice_coefficients = []
    ppv_values = []  # Store PPV values for each epoch
    sensitivities = []

    # Training process
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        intersection = 0
        union = 0
        total_ppv = 0  # Accumulate PPV values for each epoch
        # Calculate TP (True Positive) and AP (Actual Positive) at the end of each epoch
        total_true_positive = 0
        total_actual_positive = 0

        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)
            train_loss = dice_loss(out_image, segment_image)

            # Zero gradients, perform backpropagation, and update parameters
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            # Calculate numerator and denominator for Dice coefficient
            intersection += torch.sum(out_image * segment_image)
            union += torch.sum(out_image) + torch.sum(segment_image)

            # Calculate and record PPV value for each batch
            ppv_value = ppv(out_image, segment_image)
            total_ppv += ppv_value
            ppv_values.append(ppv_value)

            # Calculate and record true positive and actual positive for each batch
            true_positive = torch.sum(out_image * segment_image)
            actual_positive = torch.sum(segment_image)
            total_true_positive += true_positive.item()
            total_actual_positive += actual_positive.item()

            if i % 2 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            # Save model weights every 50 steps
            if i % 20 == 0:
                torch.save(net.state_dict(), weight_path)

            # Generate heatmap from captured feature maps
            if len(features) > 0:
                feature_map = features[-1][0]  # Extract feature map saved in the hook
                feature_map = feature_map.cpu().detach().numpy()

                # Convert feature map to heatmap (adjust as needed)
                heatmap = np.mean(feature_map, axis=0)  # Use mean across all channels
                heatmap = np.maximum(heatmap, 0)  # Remove negative values
                heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[3]))  # Resize to match original image size
                heatmap = heatmap / np.max(heatmap)  # Normalize to [0, 1]

                # Convert heatmap to RGB
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                # Save heatmap
                plt.imshow(heatmap)
                plt.axis('off')
                plt.savefig(f'{save_path}/heatmap_epoch{epoch}_batch{i}.png', bbox_inches='tight')
                plt.close()

            # Save input image, target image, and output image for visualization
            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]

            img = torch.stack([_out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')

            epoch_loss += train_loss.item()

        epoch_loss /= len(data_loader)
        losses.append(epoch_loss)

        # Calculate average Dice coefficient
        dice = (2.0 * intersection + 1e-7) / (union + 1e-7)  # Add a small value to avoid division by zero
        average_dice = dice.item()
        dice_coefficients.append(average_dice)

        # Calculate average PPV
        average_ppv = total_ppv / len(data_loader)

        # Calculate sensitivity for each epoch
        sensitivity_value = total_true_positive / (total_actual_positive + 1e-7)  # Add a small value to avoid division by zero
        sensitivities.append(sensitivity_value)

        print(
            f'Epoch {epoch} - Average loss: {epoch_loss}, Average Dice coefficient: {average_dice}, Average PPV: {average_ppv}, Average Sensitivity: {sensitivity_value}')

    # Plot the loss curve, Dice coefficient curve, and PPV curve during training
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Training')

    plt.subplot(1, 3, 2)
    plt.plot(dice_coefficients)
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient Training')

    plt.subplot(1, 3, 3)
    plt.plot(sensitivities)
    plt.xlabel('Epoch')
    plt.ylabel('Sensitivity')
    plt.title('Sensitivity Curve')

    plt.tight_layout()
    plt.show()

    # Save loss values to a file
    with open('your file', 'w') as file:
        for loss_value in losses:
            file.write(str(loss_value) + '\n')
