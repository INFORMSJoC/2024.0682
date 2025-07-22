# Eliminating Social Popularity Bias in Recommendation: Causal Inference-Based Social Graph Neural Networks

This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the MIT License.

The software and data in this repository are a snapshot of the software and data that were used in the research reported on in the paper:

["Eliminating Social Popularity Bias in Recommendation: Causal Inference-Based Social Graph Neural Networks"](https://doi.org/10.1287/ijoc.2024.0682)

---

## Cite
To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

-[https://doi.org/10.1287/ijoc.2024.0682](https://doi.org/10.1287/ijoc.2024.0682)

-[https://doi.org/10.1287/ijoc.2024.0682.cd](https://doi.org/10.1287/ijoc.2024.0682.cd)

Below is the BibTex for citing this snapshot of the repository:

```bibtex
@article{CISGNN,
  author =        {H.L. Xu, R.N. Yang, and R.B. Geng},
  publisher =     {INFORMS Journal on Computing},
  title =         {Eliminating Social Popularity Bias in Recommendation: Causal Inference-Based Social Graph Neural Networks},
  year =          {2024},
  doi =           {10.1287/ijoc.2024.0682,cd},
  note =          {Available for download at https://github.com/INFORMSJoC/2024.0682},
}  
```

---

## Description
Social recommender models not only exhibit a well-known bias toward popular items but also have a social popularity bias that 
is often overlooked in existing research. Both biases can lead the model to learn inaccurate user representations, 
ultimately compromising the diversity and accuracy of recommendations. Using the backdoor adjustment operator and the 
counterfactual reasoning strategy as key components, a causal inference-based 
social graph neural network (CISGNN) is proposed.

The code in `src/run_dec_model.py` implements the main training and evaluation pipeline for CISGNN. 
The model and its components are defined in `src/models/`. 
The pipeline includes data loading, model training, validation, early stopping, and final testing.

---

## Data and Instructions to Run CISGNN

We use four real-world datasets: Ciao, 
Epinions, Yelp(Philadelphia), and Yelp(Tucson). Each dataset contains user-item ratings and social trust relationships. The datasets are located in the `data/raw/` directory:

- `data/raw/Ciao/`
- `data/raw/Epinions/`

Each folder contains rating and trust files. `src/dataloader.py` will automatically process and use these datasets. The processed data sets are stored in the `data/processed/` directory.
You can also obtain the original dataset by clicking on the following hyperlink:[Ciao](https://www.cse.msu.edu/~tangjili/trust.html), [Epinions](https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.html), [Yelp](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset). Becides, since the Yelp data set is too large, you can download the corresponding data through the link and use the `data/proc_yelp.py` to extract the data of the corresponding city.

### Running the Code

To run the CISGNN model, use the following command from the project root directory:

```bash
python src/run_dec_model.py
```

The results and trained models will be saved in the `results/` and `saved_models/` directories, respectively.

---

## Prerequisites

Please install the following packages before running the CISGNN model:

- python >= 3.6
- torch >= 1.7
- pandas >= 1.1

You can install the required packages with:

```bash
pip install torch pandas
```

---

## Project Structure

- `src/run_dec_model.py` : Main script to train and evaluate CISGNN
- `src/models/` : Model definitions
- `src/dataloader.py` : Data loading utilities
- `src/Procedure.py` : Training and evaluation procedures
- `src/utils.py` : Utility functions
- `data/raw/` : Raw datasets
- `results/` : Output results
- `saved_models/` : Saved model checkpoints

 

---

## Contact

For questions or issues, please contact:
- Huilin Xu: xuhuilin@stu.xjtu.edu.cn
- Ruina Yang: rnyang@mail.xjtu.edu.cn
- Ruibin Geng: rbgeng@nwpu.edu.cn

