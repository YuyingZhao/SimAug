for dataset in "Baby_Products"
do
    python preprocess_kcore.py --dataset_name=$dataset --kcore_num=5
    python obtain_embeddings_st.py --dataset_name=$dataset --flag='6'
    python augment_dataset.py --dataset_name=$dataset --aug_num=2 --strategy='sentence_transformer_6'
done

