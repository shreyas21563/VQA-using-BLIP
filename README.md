# Visual Question Answering Using BLIP

## Project Overview

This project evaluates the performance of the BLIP (Bootstrapping Language-Image Pre-training) model for the Visual Question Answering (VQA) task. We performed inference using a pretrained BLIP model on three datasets: VQA v2.0 training, VQA v2.0 validation, and DAQUAR. The model was pretrained on both the VQA v2.0 training and validation datasets but not on the DAQUAR dataset. We analyzed the model's performance using various evaluation metrics and documented our findings in this project.

## Project Structure

The project directory contains the following files and folders:

- `.git/` - Git repository metadata.
- `DAQUAR/` - Contains the DAQUAR dataset.
- `Evaluation.ipynb` - Jupyter notebook for evaluating the model.
- `LatexCode/` - Contains LaTeX code used for the report.
- `README.md` - This README file.
- `Report.pdf` - Detailed project report.
- `Slides.pptx` - Presentation slides summarizing the project.
- `Visualization.ipynb` - Jupyter notebook for visualizing the datasets.
- `VQA_v2_Training/` - Contains the VQA training dataset.
- `VQA_v2_Val/` - Contains the VQA validation dataset.

## Datasets

We utilized the following datasets for evaluating the BLIP model:

| Dataset             | # Images | # Questions | # Answers     |
|---------------------|----------|-------------|---------------|
| **VQA v2.0 Training**   | 82,783   | 443,757     | 4,437,570     |
| **VQA v2.0 Validation** | 40,504   | 214,354     | 2,143,540     |
| **DAQUAR**              | 1,449    | 5,674       | 5,674         |

## Evaluation Metrics

We used a diverse set of evaluation metrics to assess the model's performance, each offering unique insights into its capabilities. Below are the metrics used:

1. **Accuracy**: The proportion of correctly predicted instances out of all instances.
2. **BLEU Score**: Measures the geometric average of the modified n-gram precisions, adjusted for brevity.
3. **BERT Score**: Uses pre-trained contextual embeddings from BERT to match words in candidate and reference sentences by cosine similarity.
4. **WUPS Score**: Calculates the similarity between two words based on their longest common subsequence in the taxonomy tree.
5. **VQA Score**: Assesses the correspondence between the model’s response and all potential answers provided for a given question.

## Results

### Accuracy

| Dataset             | Accuracy |
|---------------------|----------|
| VQA v2.0 Training   | 0.769    |
| VQA v2.0 Validation | 0.766    |
| DAQUAR              | 0.230    |

### BLEU Scores

| Dataset             | BLEU1 | BLEU2 | BLEU3 | BLEU4 |
|---------------------|-------|-------|-------|-------|
| VQA v2.0 Training   | 0.763 | 0.552 | 0.438 | 0.349 |
| VQA v2.0 Validation | 0.760 | 0.551 | 0.438 | 0.354 |
| DAQUAR              | 0.183 | 0.081 | 0.037 | 0.0   |

### BERT Scores

| Dataset             | BERT Precision | BERT Recall | BERT F1 |
|---------------------|----------------|-------------|---------|
| VQA v2.0 Training   | 0.985          | 0.986       | 0.985   |
| VQA v2.0 Validation | 0.985          | 0.985       | 0.985   |
| DAQUAR              | 0.945          | 0.935       | 0.939   |

### WUPS Scores

| Dataset             | WUPS 0.0 | WUPS 0.9 |
|---------------------|----------|----------|
| VQA v2.0 Training   | 86.573   | 79.484   |
| VQA v2.0 Validation | 86.223   | 79.203   |
| DAQUAR              | 58.122   | 30.680   |

### VQA Scores

| Dataset             | VQA Score |
|---------------------|-----------|
| VQA v2.0 Training   | 84.89     |
| VQA v2.0 Validation | 84.73     |
| DAQUAR              | -         |

## Experimental Setup

The VQA v2.0 datasets was divided into parts to handle its large size. The training set was divided into 10 parts, and the validation set into 2 parts, to facilitate parallel processing. The DAQUAR dataset was small enough to process as a whole.

## Conclusion

The BLIP model shows high performance on the VQA v2 datasets, with good precision and recall metrics. However, the model struggles with the DAQUAR dataset, as evidenced by lower accuracy and BLEU scores. Despite this, high BERT scores indicate strong semantic similarity in the model's answers across all datasets. This project highlights the importance of using multiple evaluation metrics to gain a comprehensive understanding of model performance.

## How to Use

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- Jupyter Notebook
- Required Python packages (can be found in `requirements.txt` if available)

### Running the Evaluation

1. Clone the repository:
    ```bash
    git clone https://github.com/shreyas21563/VQA-using-BLIP
    ```

2. Navigate to the project directory:
    ```bash
    cd VQA-using-BLIP
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Jupyter notebooks:
    ```bash
    jupyter notebook
    ```

5. Open `Evaluation.ipynb` and `Visualization.ipynb` to run the evaluation and visualization code.

## Authors

- Shreyas Kabra
- Ritwik Harit
- Vasan Vohra

## References
1. Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip:
 Bootstrapping language-image pre-training for unified vision
language understanding and generation. In International con
ference on machine learning, pages 12888–12900. PMLR,
 2022.
2. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
 Toutanova.
 Bert:
 Pre-training of deep bidirectional
 transformers for language understanding. arXiv preprint
 arXiv:1810.04805, 2018.
3. Mateusz Malinowski and Mario Fritz. A multi-world ap
proach to question answering about real-world scenes based
 on uncertain input. Advances in neural information process
ing systems, 27, 2014.
4. George A Miller. Wordnet: a lexical database for english.
 Communications of the ACM, 38(11):39–41, 1995
