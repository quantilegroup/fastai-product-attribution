# Automating your product attribute tagging with FastAI and PyTorch 

This repo contains a quick, one-script tutorial for training an image classifier using PyTorch and fastai. 


## The business case for automated product attribution using computer vision

New products are a high-risk, high-rewards proposition: [more than 25% of annual profits come from the launch of new products](https://www.mckinsey.com/business-functions/marketing-and-sales/our-insights/how-to-make-sure-your-next-product-or-service-launch-drives-growth#:~:text=More%20than%2025%20percent%20of,McKinsey%20survey%20(Exhibit%201).), but [40-60% of new product launches fail](https://newproductsuccess.org/new-product-failure-rates-2013-jpim-30-pp-976-979/) within the first year (and often in as little as ten weeks). A successful launch often hinges on just one calculation: the forecasted sales for the first few months of the product's life. These forecasts are critical to making a variety of decisions across the value chain, including:

![List of new product launch decisions influenced by forecasting](/static/forecasting_decisions.svg)

If you're going to spend millions of dollars to bring a new product to market, you want to be sure you're doing everything you can to nail your initial forecasts. Unfortunately, new product forecasting is notoriously difficult for nearly every organization due to heterogeneous customer tastes, a lack of historical data, and the need to sell-in your forecasts to a cross-functional group of stakeholders.

Though every product launch is different, we've found four keys to success while advising previous clients on their launches over the years:

1. **Pool variance between similar products by leveraging shared product attributes in your modeline pipelines** (our focus for this post)
2. Predict confidence intervals, not point estimates, to help your team think-through different scenarios scenarios
3. Rely on generative models to produce stable results with small sample sizes
4. Build an automated pipeline that you can easily review, understand, and update as you collect new information

**In this post, we'll share how you can use computer vision to quickly identify shared attributes between products**, even for products that your business has never offered before. A sample of our training code is available [on GitHub](https://github.com/quantilegroup/computer_vision_for_product_attribution) if you'd like to follow along. The pipeline we'd use to train and serve image classifications depends on your infrastructure, but might look something like the following: 

![Illustration of training and serving process](/static/train.svg)


## Aside: keeping up with computer vision technology

Computer vision has gone from technical marvel to commoditized technology in less than a decade. Convolutional neural networks (the family of deep learning algorithms commonly used for computer vision tasks) have become exponentially better, faster, and cheaper due to advancements in a number of related disciplines:

- **Hardware**: CPUs and GPUs have improved tremendously over time, but cloud is the real game-changer: anyone can access powerful, reliable hardware from the convenience of their laptop 
- **Low-level computational software**: Open-source frameworks like [PyTorch](https://pytorch.org/) (Facebook), [TensorFlow](https://www.tensorflow.org/) (Google), and [CUDA](https://developer.nvidia.com/cuda-zone) (NVIDIA) have significantly reduced the technical expertise required to run high-performance calculuations on GPUs and tensors
- **Open-source libraries**: With better hardware and software at their disposal, the open-source community has rallied around [fastai](https://www.fast.ai/about/) and [OpenCV](https://opencv.org/) to build consumer-friendly technology that is powerful enough for commercial use

fastai, in particular, has paved the way for non-specialists like us to get up and running in computer vision through their [intuitive trainings](https://course.fast.ai/) and [Python library built on PyTorch](https://docs.fast.ai/). fastai is far from perfect (see "Challenges" at the end), but it's cutting-edge modeling components make it our framework of choice for computer vision. 

## Gathering image-attribute data

Before you start to code, you're going to need hundreds of image-attribute pairings to both train your classifier and validate its results. There isn't a "minimum number" of images you should be aiming for; people used to say you needed at least 1,000 images per attribute, but the invention of transfer learning and our need for multi-class predictions (both described later) suggest that you can likely get away with a few hundred images per class. The more images you can provide upfront, the better your performance will be.

You'll want to choose images and attributes that match what you'd like to predict in the real world. If all your suppliers send you images of clothing on white backgrounds, then you should build a training set that mostly includes white-background studio photographs. Mixing different types of photos (e.g., including lifestyle photographs in with your studio photographs) can improve classifier performance, but adding-in unnecessary attributes will likely hurt your predictions. 

The attributes you choose should match the metadata you (hopefully) already collect from your suppliers. A few attributes we commonly see are:
- Categories (e.g., shirt, blouse, cardigan)
- Subcategories (e.g., band shirts, button-ups, pumps)
- Collection (e.g., spring, winter)
- Material (e.g., sequin, snakeskin, seersucker)
- Fit (e.g., above-the-hip, midi, double-breasted)
- Patterns (e.g., flowers, Hawaiian, herringbone)
- Embellishments (e.g., tassels, rivets, ruffles)
- Base and secondary colors

When picking attributes, the core question we're trying to answer is: "Could the presense or absence of this attribute cause a consumer to purchase an item?". If the answer is yes, and we can find and label enough images to train on the attribute, then we should include it in our training data. Some attributes will certainly be more predictive than others, but we want to include as many of them in our modeling pipeline as possible and let our algorithms sort them out.

To connect each image with its product attributes, we recommend creating a simple .csv with one column containing the image name and another column which lists its attributes. We want each image to be tied to multiple attributes, so we'll tell fastai that we want to run multi-class predictions by a) including multiple attributes in our "AttributeID" column, delimited by underscores and b) passing a `MultiCategoryBlock` to the `DataBlock` (see the code snippet below). Here's an illustration of what we mean, with each attribute encoded as a unique ID for our algorithm to learn from:

!["Example of train.csv and labels.json"](/static/example.svg)

fastai provides many different ways you can organize your images, labels, and folders using their [DataLoaders API](https://docs.fast.ai/vision.data), so you don't have to reorganize your whole data warehouse to test this technology. 

If you want to experiment with this technology using open-source images, there are several great sets to choose from:

- [DeepFashion (280k images)](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data)
- [Kaggle's iMaterialist Fashion Challenge (20GB)](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data)
- [Stanford's Clothing Attributes Dataset (1856 images)](https://exhibits.stanford.edu/data/catalog/tb980qz1002)

Note that these images may have licensing restrictions that make them unfit for commercial use.

## Preprocess your images

fastai's latest library uses a [`DataBlock`](https://github.com/fastai/fastai2/blob/master/nbs/50_tutorial.datablock.ipynb/) class to make it easy to bulk important image and attributes according to the schema we laid-out in our csv. The exact steps you use when preprocessing can vary by use case, but here's where to start.

### Split images into training, test, and validation sets

First, we split our data into three sets of images so that fastai can measure out-of-sample error as it trains our convolutional neural network. There's a lot to be said about [picking a strong validation strategy](https://www.fast.ai/2017/11/13/validation-sets/), but we'll leave that discussion for another post. One important note: fastai doesn't support cross-validation out-of-the-box, but you can build your own CV pipeline by passing a custom splitter function into your `DataBlock`.

```python
def splitter(df):
    """
    Tell FastAI how to split the data into test and training sets 
    (in our case, stratified sampling using the attribute column)
    """
    train, test = train_test_split(
        df, test_size=0.2, random_state=7, stratify=df["ClassId"]
    )
    return train.index, test.index
```

### Augment your images
An old trick for training computer vision models is to pass-in altered copies of the same image so that your learner adapts to various angles, sizes, and lighting conditions. Image augmentation also increases your training sample size, which can further improve your classifier's performance. 

fastai makes data augmentation easy with their [`vision.augment`](https://docs.fast.ai/vision.augment) API. In the arguments to our `DataBlock` below, we've resized, cropped, and normalized each batch of images prior to training. We've also told fastai's API where to find our data, how to split it, and how to separate each label we've laid-out in our .csv file. If you wish to go deeper into this API, we recommend reading the docs above or following-along with [fastai's tutorials](https://github.com/fastai/fastbook/blob/master/06_multicat.ipynb).

```python
dls = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    get_x=ColReader("ImageId", pref=path + "/train/"),
    get_y=ColReader("ClassId", label_delim="_"),
    splitter=splitter,
    item_tfms=Resize(128, method="crop"),
    batch_tfms=[*aug_transforms(size=256), Normalize.from_stats(*imagenet_stats)],
).dataloaders(train_df, bs=16, num_workers=0)

```


## Building your classifier

Labeling your images and getting them into fastai's `DataBlock` is the hardest part of this process. Once your data is loaded into their library, fastai's modeling capabilities make the modeling tuning incredibly easy.

### Update a pre-trained model

First, we tell fastai which model to use as the 'base' architecture for training. fastai leverages a technique called [transfer learning](http://scholar.google.com/scholar_url?url=https://www.mdpi.com/2078-2489/11/2/108/pdf&hl=en&sa=X&ei=1o2fX6qCDNe2yATS17rYCQ&scisig=AAGBfm3uWKmBPXIEiKJ03hn3ylcTujAF3A&nossl=1&oi=scholarr) where, instead of training a convolutional neural network from scratch, we instead make updates to an existing architecture that's already been trained by other researchers. 'Borrowing' a pre-built architecture improves both speed and accuracy by leveraging pretrained weights and sidestepping the [cold start problem](https://forums.fast.ai/t/the-cold-start-problem/30333).

If this sounds confusing, fastai makes it easy: the following line of code downloads ResNet152 to use as our base, removes the last layer, and initializes the last layer again with random weights for us to tune.

```py
learn = cnn_learner(dls, resnet152, metrics=[acc_metric, f_score_metric])
```

With our base architecture in place, we can easily tune the last layer of our ResNet152 architecture. `.lr_find()` provides a helpful diagnostic for finding a reasonable learning rate, while `.fit_one_cycle()` does the actual tuning. Behind the scenes, fastai's [1cycle](https://sgugger.github.io/the-1cycle-policy.html) policy uses [cyclical learning rates](https://iconof.com/1cycle-learning-rate-policy/) and momentum to both improve performance and reducing training times.

```py
# Find the best pick for our initial learning rates
learn.lr_find()

# Train our learner using FastAI's one-cycle policy, where we gradually increase and then decrease our learning rate over four epochs
# Most of our ResNet152 layers are still frozen post-download; we're only really updating the last layer
lr = 0.01
learn.fit_one_cycle(4, lr)
```

With our last layer trained, we can unfreeze the earlier layers in our model to fine-tune with a much smaller learning rate.

```py
learn.unfreeze()

lrs = slice(lr / 400, lr / 4)
learn.fit_one_cycle(4, lrs, pct_start=0.8)
```

As we tune, we can check how our model is performing with the `.show_results()` method. When we're finished, we can use `.save()` to save our ResNet's weights as a `.pth` file for later use. Your model can then be used to quickly classify new images using [PyTorch's TorchServe](https://pytorch.org/serve/) or [fastai-serving](https://github.com/developmentseed/fastai-serving).

Easy, right? We can do in a day what previously took weeks of effort from highly-qualified computer science PhDs.

## Challenges you'll face

fastai is an incredible tool, but it's still relatively young. The latest version (v2) included some welcome improvements, but also introduced new challenges:

- The new syntax and `DataBlock` APIs are not backward-compatible, which means that all of the great community content that exists out on the web is no longer useful. When you run into issues, you'll often have to dig into source code and the official docs to find your answer.
- The fastai team is hard at work updating their documentation, but the current docs are not new-user friendly
- fastai's methods are still very much in-flux, which means there's a risk that you'll have to make minor syntax updates should you update to a new version

If you're worried that fastai might be too technical for you to debug, your best alternative would be to use a paid classification API like [Ximiliar](https://demo.ximilar.com/fashion/fashion-tagging). These services aren't expensive, but may be harder to serve in your production environment due to security restrictions.


## Ways to extend this framework
- **Nested attributes**: Sometimes, we want our attributes to fall into a nested hierarchy (e.g., some attributes only apply to specific categories, but we want to predict categories and attributes at runtime). The framework we've layed above can easily handle this use case by chaining classifiers together and tuning different model weights for each category 
- **Image embeddings**: Tying similar products together using similar attributes is an intuitive way to cluster products together, but we could also use image embeddings to directly identify similar images without first tying them to pre-determined attributes
- **No-code alternatives**: Microsoft recently released [Lobe.ai](https://lobe.ai/), a new computer vision webapp that looks like it could solve this exact use case without need for code. We haven't tried this project out yet, but we're excited to see where it goes
- **Solving other use cases**: The process layed-out above can be used in a variety of other use cases, including:
  - **Search optimization**: improving your customer's search experiences through granular attribution tagging
  - **Recommendation & personalization engines**: recommending other products that customers might enjoy based on similiar attributes
  - **Metadata audits**: correcting incorrect or vague product attributes within your current database