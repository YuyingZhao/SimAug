# original dataset
for seed in 1 2 3
do
    python main.py --lr=1e-4 --n_hops=2 --model='LightGCN' --dataset="Baby_Products" --seed=$seed
done

for seed in 1 2 3
do
    python test.py --lr=1e-4 --n_hops=2 --model='LightGCN' --dataset="Baby_Products" --seed=$seed
done

# augmented dataset
for seed in 1 2 3
do
    python main.py --lr=1e-4 --n_hops=2 --model='LightGCN' --dataset="Baby_Products" --seed=$seed --flag=1 --aug_num=2 --strategy='sentence_transformer_6'
done

for seed in 1 2 3
do
    python test.py --lr=1e-4 --n_hops=2 --model='LightGCN' --dataset="Baby_Products" --seed=$seed --flag=1 --aug_num=2 --strategy='sentence_transformer_6'
done

