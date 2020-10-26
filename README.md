Sources: 
- https://www.lilystyle.ai/old-home: types of models, statistics
- https://www.semantics3.com/blog/e-commerce-image-attribute-extraction-using-machine-learning/: great overview on the use case
- https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf: Research paper which includes references to datasets like DeepFashion2, DARN
- [Paper which lists a few other free datasets](https://www.researchgate.net/figure/Comparison-of-clothing-datasets_tbl1_303901989):
  - [Clothing Attribute](https://exhibits.stanford.edu/data/catalog/tb980qz1002)
  - Apparel Style
  - Colorful-Fashion
  - MVC
- Guide on FastAI applied to DeepFashion2
  - https://medium.com/@pankajmathur/clothing-categories-classification-using-fast-ai-v1-0-in-10-lines-of-code-4e848797721
- [iMaterialist Kaggle Competition](https://www.kaggle.com/hyeonho/imaterialist-fashion-2019-at-fgvc6-eda)
- [Getting similiar vectors using Spotify's Annoy](https://towardsdatascience.com/similar-images-recommendations-using-fastai-and-annoy-16d6ceb3b809)
- [Interesting use case overview](https://www.linkedin.com/pulse/building-personalized-real-time-fashion-collection-recommender-thia/)

Use Cases:
  - Recommendation Engines: helping customers find related products ("products like this")
  - Image and text search optimization: Improving search results (e.g., floral dress example)
  - Tagging: extracting product features for use in forecasting and categorization
  - Inventory audits: identifying incorrect product categories
  - PyTorch models: https://www.kaggle.com/jpraveenkanna/fashion-classification

Guides:
- Kaggle Fashion
  - Data
    - https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset
    - https://www.kaggle.com/paramaggarwal/fashion-product-images-small
  - Notebooks
    - Similiar products (keras): https://www.kaggle.com/niharika41298/product-recommendations-plotly-dash-interactive-ui
    - Pre-transformations: https://www.kaggle.com/marlesson/building-a-recommendation-system-using-cnn-v2
  - 12 free retail image datasets
    - https://lionbridge.ai/datasets/12-free-retail-image-datasets-for-computer-vision/
- (MMFashion)[https://github.com/open-mmlab/mmfashion]: Prebuilt pytorch library to do product attribution
  - Data
    - DeepFashion2 dataset
    - Fash

Approaches to build:
    - Custom PyTorch
    - FastAI
    - MMFashion
    - Free Ximilar API

Data: 
    - MMFashion (built-in)
    - [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)
    - Kaggle (44k images)

Companies in this space:
- Semantics3
- Lily AI
- [Ximilar](https://www.ximilar.com/) (have a very cool API / demo)

Keywords:
- personalization
- shopper experiences
- image recognition
- fashion

TODOs:
- Delete product data
- Use a mix of CV and NLP
- Apply image transformations