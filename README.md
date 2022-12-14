# GCNext

Unsupervised Graph Embeddings for Session-based Recommendation with Item Features.

Use `pipenv` to create virtual environment. 

Diginetica & Tmall datasets can be found under: https://drive.google.com/drive/folders/1ritDnO_Zc6DFEU6UND9C8VCisT0ETVp5

Copy raw datasets to `data` folder. 

## Usage
Tmall dataset as example:
- Preprocessing: in `dataset` folder, run:
    ```
    python tmall.py
    ```

- Generate GNN embeddings: in `gnn` folder, run:
    ```
    python bgrl.py --config ./config/tmall/bgrl.yaml 
    ```
- Run sequential model training: in `seq_rec` folder, run:
    ```
    python run.py --config config/tmall/gru4recgnn.yaml
    ```

Change config paths for different datasets and models.

## Requirements
See `Pipfile`.

## Citation
```
@inproceedings{peintner:cars:2022,
 author = {Andreas Peintner, Marta Moscati, Emilia Parada-Cabaleiro, Markus Schedl, Eva Zangerle},
 booktitle = {CARS: Workshop on Context-Aware Recommender Systems (RecSys ’22)},
 title = {Unsupervised Graph Embeddings for Session-based Recommendation with Item Features},
 year = {2022}
}
```

## References
- https://github.com/nerdslab/bgrl
- https://github.com/rn5l/session-rec
- https://github.com/RUCAIBox/RecBole

