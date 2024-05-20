import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_dataloader_sample(
        batch, sample_size: int = 4, output_path: str = None
    ):
    plt.clf()
    fig, ax = plt.subplots(1, sample_size)

    imgs = batch['image'][:sample_size]
    masks = batch['label'][:sample_size]

    for i in range(imgs.shape[0]):
        ax[i].imshow(imgs[i][0])
        ax[i].imshow(masks[i][0], cmap='gnuplot', alpha=0.5)
        ax[i].set_xticks([])
        ax[i].set_yticks([])

        if ('point_coords' in batch and batch['point_coords'] is not None):
            for pt in batch['point_coords'][i]:
                ax[i].plot(pt[0], pt[1], 'o', color='r', markersize=2)
        if 'boxes' in batch and batch['boxes'] is not None:
            min_x, min_y, max_x, max_y = batch['boxes'][i]
            ax[i].add_patch(
                patches.Rectangle(
                    xy=(min_x, min_y),
                    width=max_x - min_x,
                    height=max_y - min_y,
                    linewidth=1,
                    color='red', fill=False))

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    if output_path is not None:
        plt.savefig(output_path)

    return fig