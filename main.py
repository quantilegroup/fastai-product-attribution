from fastai.vision.all import *
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt

# NOTE this script is meant to be a tutorial, with extra comments for
# illustrative purposes. The hardest part of running this script is getting fast.ai's DataLoader to
# work for your use case. We recommend reading their docs here:
#   - docs: https://docs.fast.ai/data.load
#   - tutorial: https://github.com/fastai/fastbook/blob/master/06_multicat.ipynb


def duplicate_unique_rows(df, classes):
    """
    Oversample classes with only one observation to avoid exceptions when splitting into test and training sets

    TODO write a new multi-label stratification function that stratifies 
    individual attributes, not concatenated attributes 
    """
    unique_index = ~df.duplicated(subset=[classes])

    output_df = df.copy().append(df[unique_index])

    return output_df


def plot_accuracy_thresholds(learner):
    """
    Plot accuracy thresholds (x-axis) against accuracy out-of-sample accuracy metrics (y-axis)
    """

    preds, targets = learner.get_preds()

    thresholds = torch.linspace(0.05, 0.95, 29)

    accuracies = [
        accuracy_multi(preds, targets, thresh=i, sigmoid=False) for i in thresholds
    ]

    return plt.plot(thresholds, accuracies)


def splitter(df):
    """
    Tell FastAI how to split the data into test and training sets (in our case, stratified sampling using the attribute column)
    """
    train, test = train_test_split(
        df, test_size=0.2, random_state=7, stratify=df["ClassId"]
    )
    return train.index, test.index


path = "./data"

# Load-in csv that ties each image to its attributes
raw_df = pd.read_csv(f"{path}/train.csv")[["ImageId", "ClassId"]]

# Oversample unique attributes so that they can be split into training / testing
train_df = duplicate_unique_rows(raw_df, "ClassId").reset_index(drop=True)

# Load images and attributes (ImageBlock and MultiCategoryBlock, respectively) into a DataBlock for processing (see README for docs)
# We'll also augment, downsize, and normalize our images to improve runtime. We can easily revert back to our originals.
dls = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    get_x=ColReader("ImageId", pref=path + "/train/"),
    get_y=ColReader("ClassId", label_delim="_"),
    splitter=splitter,
    item_tfms=Resize(128, method="crop"),
    batch_tfms=[*aug_transforms(size=256), Normalize.from_stats(*imagenet_stats)],
).dataloaders(train_df, bs=16, num_workers=0)

# Visually validate a subset of our images
dls.show_batch()


# Create custom error metrics
acc_metric = partial(accuracy_multi, thresh=0.2)
f_score_metric = partial(F1ScoreMulti, thresh=0.2)

# Download standard ResNet50 to use as our base weights. The last layer will be removed and initialized again with random weights
learn = cnn_learner(dls, resnet50, metrics=[acc_metric, f_score_metric])

# Find the best pick for our initial learning rates
learn.lr_find()


# Train our learner using FastAI's one-cycle policy, where we gradually increase and then decrease our learning rate over four epochs
# Most of our ResNet50 layers are still frozen post-download; we're only really updating the last layer
lr = 0.01
learn.fit_one_cycle(4, lr)

# Save our model in the default path "./models/"
learn.save("fashion_weights_stage_1")

# Show results
learn.show_results()

# Now that we've refined the last layer of our NN, we can efficiently retrain the others layers of our network using .unfreeze()
learn.unfreeze()

lrs = slice(lr / 400, lr / 4)
learn.fit_one_cycle(4, lrs, pct_start=0.8)

learn.save("fashion_weights_stage_2")

learn.show_results()

# Final note: we could have simulated the freeze/unfreeze training process using the learn.fine_tune() call below
# learn.fine_tune(4, base_lr=lr, freeze_epochs=4)
