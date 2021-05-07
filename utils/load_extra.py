import random
import matplotlib.pyplot as plt


def plot_n_imgs(input_image_list, true_mask_list, pred_mask_list, n):
    if not len(input_image_list) == len(true_mask_list) == len(pred_mask_list):
        raise ValueError(
            "The image, true mask, and pred mask lists have to be the same length."
        )

    zipped_list = list(zip(input_image_list, true_mask_list, pred_mask_list))

    random.shuffle(zipped_list)

    shuffled_input_image_list, shuffled_true_mask_list, shuffled_pred_mask_list = zip(
        *zipped_list
    )

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(n * 2, n * 3))

    for i in range(n):
        axes[i, 0].imshow(shuffled_input_image_list[i])
        axes[i, 0].set_title("Input Image")

        axes[i, 1].imshow(shuffled_true_mask_list[i])
        axes[i, 1].set_title("True Mask")

        axes[i, 2].imshow(shuffled_pred_mask_list[i])
        axes[i, 2].set_title("Pred Mask")

    fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)

    return fig
