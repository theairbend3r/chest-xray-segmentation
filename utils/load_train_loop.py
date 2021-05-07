from tqdm import tqdm

import torch

from .load_extra import plot_n_imgs
from .load_loss import jaccard_index, dice_coef


def train_loop(
    epochs,
    model,
    optimizer,
    scheduler,
    criterion,
    writer,
    data_name,
    data_subset,
    model_name,
    train_loader,
    val_loader,
    model_save_path,
    device,
):

    for e in tqdm(range(1, epochs + 1)):

        # TRAINING
        train_epoch_loss = 0
        model.train()

        # Load train batch
        for x_train_batch, y_train_batch in train_loader:
            x_train_batch, y_train_batch = (
                x_train_batch.to(device),
                y_train_batch.to(device),
            )

            optimizer.zero_grad()

            y_train_pred = model(x_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()

        # VALIDATION
        with torch.no_grad():
            model.eval()

            val_epoch_loss = 0
            val_epoch_iou = 0
            val_epoch_dice = 0

            val_true_image_list = []
            val_true_mask_list = []
            val_pred_mask_list = []

            for x_val_batch, y_val_batch in val_loader:
                x_val_batch, y_val_batch = (
                    x_val_batch.to(device),
                    y_val_batch.to(device),
                )

                y_val_pred = model(x_val_batch)
                val_loss = criterion(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()

                y_val_pred = torch.log_softmax(y_val_pred, dim=1)
                _, y_val_pred_tag = torch.max(y_val_pred, dim=1)

                val_epoch_iou += jaccard_index(
                    y_val_pred_tag.squeeze(), y_val_batch.squeeze(), num_classes=2,
                )[0]

                val_epoch_dice += dice_coef(
                    y_val_pred_tag.squeeze(), y_val_batch.squeeze(), num_classes=2,
                )

                # Save val images every 10 epochs to display on Tensorboard
                if e % 10 == 0:
                    val_true_image_list.append(x_val_batch.squeeze().cpu().numpy())
                    val_true_mask_list.append(y_val_batch.squeeze().cpu().numpy())
                    val_pred_mask_list.append(y_val_pred_tag.squeeze().cpu().numpy())

        scheduler.step()

        # Save model every 2 epochs
        if e % 2 == 0:
            model_path = f"{model_save_path}/{data_name}/{data_name}_{data_subset}_{model_name}_epoch_{e}_loss_{val_epoch_loss/len(val_loader):.3f}.pt"
            torch.save(model.state_dict(), model_path)

        # Dsiplay 5 val images every 10 epochs on Tensorboard
        if e % 10 == 0:
            predictions_on_random_samples = plot_n_imgs(
                val_true_image_list, val_true_mask_list, val_pred_mask_list, n=5
            )
            writer.add_figure(
                "predictions_on_random_samples", predictions_on_random_samples, e
            )

        train_loss = train_epoch_loss / len(train_loader)
        val_loss = val_epoch_loss / len(val_loader)
        val_iou = val_epoch_iou / len(val_loader)
        val_dice = val_epoch_dice / len(val_loader)

        writer.add_scalar("Train/Loss (CE)", train_loss, e)
        writer.add_scalar("Val/Loss (CE)", val_loss, e)
        writer.add_scalar("Val/Acc (IoU)", 100 * val_iou, e)
        writer.add_scalar("Val/Acc (Dice)", 100 * val_dice, e)

        writer.flush()

        print("=" * 50)
        print(
            f"Epoch {e+0:02}: \n\tTrain Loss: {train_loss:.5f} \n\tVal Loss: {val_loss:.2f} \n\tVal IoU: {val_iou:.4f} \n\tVal DICE: {val_dice:.4f}"
        )
