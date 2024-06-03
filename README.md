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

1. **VQA v2.0 Training**: 
   - **# Images**: 82,783
   - **# Questions**: 443,757
   - **# Answers**: 4,437,570

2. **VQA v2.0 Validation**:
   - **# Images**: 40,504
   - **# Questions**: 214,354
   - **# Answers**: 2,143,540

3. **DAQUAR**:
   - **# Images**: 1,449
   - **# Questions**: 5,674
   - **# Answers**: 5,674

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

## Exploratory Data Analysis

### Question Length Distribution
- **VQA v2**: Majority of questions consist of 4 to 7 words.
- **DAQUAR**: Questions are generally longer, ranging from 6 to 11 words.

### Answer Length Distribution
- Majority of answers across all datasets consist of 1 or 2 words.

### Question Type Distribution
- **VQA v2**: Predominantly questions starting with "how many," "is the," and "what".
- **DAQUAR**: Frequently begins with "what is on the," "what is the," and "what is".

## Experimental Setup

The VQA v2.0 dataset was divided into parts to handle its large size. The training set was divided into 10 parts, and the validation set into 2 parts, to facilitate parallel processing. The DAQUAR dataset was small enough to process as a whole.

## Analysis

### Accuracy
- **Case 1**: Predicted "trash can" vs. ground truth "garbage bin" - 0 accuracy despite semantic similarity.
- **Case 2**: Predicted "blue chair" vs. ground truth "chair" - 0 accuracy despite more detailed answer.

### BLEU Score
- **Case 1**: 0 BLEU score due to lack of n-gram overlaps.
- **Case 2**: 0.5 BLEU score due to small degree of overlap.

### BERT Score
- **Case 1**: High semantic similarity with a BERT Precision score of 0.90.
- **Case 2**: High semantic similarity with a BERT Precision score of 0.83.

### WUPS Score
- **Case 1**: 0.76 at 0 threshold, 0.0076 at 0.9 threshold.
- **Case 2**: 0.11 at 0 threshold, 0.011 at 0.9 threshold.

### VQA Accuracy
- **Case 1**: 0.67 due to matching 2 out of 10 available answers.
- **Case 2**: 0.33 due to matching 1 out of 10 available answers.

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
    git clone <repository-url>
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

1. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805, 2018.
2. Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. In International conference on machine learning, pages 12888–12900
