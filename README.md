Environment: 
install pytorch
install sentence transformer: pip install -U sentence-transformers
pip install torch_geometric
pip install torch-scatter

File Description
There are two folders where ./datasets is everything about the dataset, and ./rec is code related to recommendation.
Specifically,  the description of documents in ./datasets are as follows:
    python files
    - preprocess_kcore.py: preprocess the raw data downloaded from amazon website into structured data
    - obtain_embeddings_st.py: obtain the item embeddings based on sentence transformer
    - augment_dataset.py: generate augmented interactions based on the given strategies including 'random', 'sentence_transformer_6'
    folders: three folders src/preprocessed/aug related to the raw downloaded data, preprocessed data, and augmented interactions
The description of documents in ./rec are as follows:
    python files
    - parse.py: define the input parameters such as the name of dataset, learning rate
    - main.py: the main training framework
    - dataprocess.py: preprocess the data for training from the structured data in ./datasets/{dataset_name}/preprocessed
    - model.py: the recommendation models such as LightGCN and MF
    - evaluation.py: define the evaluation metrics
    - test.py: evaluate the trained model
    - utils.py: utility functions
    folders:
    - trained_models: save the trained models
    - results_rec_list: save the recommendation list after executing save_rec.py, used for rec-based augmentation
    - results_logs: save the results in terms of utility and fairness performance, each line in the file corresponds to one seed
    - results: intermediate files, can ignore
    
How to run a new dataset?
[1] Dataset Preparation
Download meta data from https://amazon-reviews-2023.github.io/ and interaction data from https://amazon-reviews-2023.github.io/data_processing/0core.html
Save two downloaded files into ./datasets/{dataset_name}/src
Under the folder of ./datasets:
[Preprocess dataset] Run preprocess using python preprocess_kcore.py, the preprocessed data will be automatically saved into ./datasets/{dataset_name}/preprocessed
    The preprocessed files include the following:
    - edges.pkl: (train_edges, val_edges, test_edges)
    - id_to_idx.pkl: (user_id_to_idx, item_id_to_idx) where id is the original id used in amazon dataset and idx is the index
    - idx_to_title.pkl: a dictionary where key is the item idx and value is the title of this item
    - item_popularity_label.pkl: popular item set, unpopular item set
    - user_activeness_label.pkl: active user set, inactive user set
[Obtain item textual embedding] Run python obtain_embeddings_st.py and a embedding document will be saved into the preprocessed directory
[Generate augmented interactions] Run python augment_dataset.py and the augmented interactions will be saved into ./datasets/{dataset_name}/aug/
We provide the detailed command in run.sh for the above three stages using the example of one dataset.

[2] Recommendation Training
Under the folder of ./rec:
[Train on the original dataset]
[Train on the augmented dataset]

[3] Recommendation Evaluation
Under the folder of ./rec
The evaluation results will be automatically saved to files

The command examples are in run.sh.