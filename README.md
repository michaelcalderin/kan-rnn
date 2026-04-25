# Project Overview
This project implements a KAN-RNN (Kolmogorov-Arnold Network and RNN hybrid) which takes an RNN and replaces learned weights at the neurons with learned functions (B-splines) at the edges. "Efficient KAN" was used for this implementation. Results on a sentiment analysis task were compared to standard RNNs and LSTMs (Long Short-Term Memory). The project is organized to allow for exploration of KANs and their potential role in producing smaller, more expressive models for sequential tasks.

## Important Overview Reference
Please refer to the **overview-and-results.pptx** file in the main directory. This presentation provides an important overview of the project and summarizes the results. It is highly recommended for all readers to review this file for a comprehensive understanding of the project.

## Project Structure
```
overview-and-results.pptx      # Project overview and results presentation (recommended reading)
Resources/
    Data/
        data.csv                # Main dataset for training/testing
        glove.6B.100d.txt       # Pre-trained GloVe word embeddings
Results/
    # Output directory for experiment results
Scripts/
    kan_rnn.py                 # KAN-RNN model implementation and experiments
    lstm.py                    # LSTM model implementation and experiments
    rnn.py                     # Standard RNN model implementation and experiments
```

## Getting Started

### Prerequisites
- Python 3.7+
- Recommended: Create a virtual environment
- Required Python packages (install with pip):
  - numpy
  - pandas
  - torch
  - git+https://github.com/Blealtan/efficient-kan.git

### Setup
1. Clone this repository or download the project files.
2. Install the required Python packages:
   ```bash
   pip install numpy pandas torch git+https://github.com/Blealtan/efficient-kan.git
   ```
3. Place your dataset (`data.csv`) from `Resources/Data/` and GloVe embeddings (`glove.6B.100d.txt`) from `https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt` (or preferred embeddings) into the same directory as the Scripts.

## Usage

Run experiments by executing the desired script from the `Scripts/` directory. For example:

```bash
python Scripts/rnn.py
python Scripts/lstm.py
python Scripts/kan_rnn.py
```

Each script has its own command-line arguments for configuration. Refer to the script source code for details. Here is an example for training the KAN-RNN:

```bash
python Scripts/kan_rnn.py \
  --data data.csv \
  --glove_path glove.6B.100d.txt \
  --embed_dim 100 \
  --hidden_dim 128 \
  --num_layers 1 \
  --dropout 0.3 \
  --batch_size 64 \
  --epochs 10 \
  --lr 1e-4 \
  --seed 42 \
  --max_len 128 \
  --freeze_embeddings \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --weight_decay 0.0
  ```

## Results
### Summary
- Performance was comparable to LSTM in terms of metrics and reduced overfitting
- Both LSTM and KAN-RNN were outperformed by Standard RNN, but RNN showed greater signs of overfitting
- KAN-RNN was significantly slower to train (~20x slower than the RNN in this project)
- Performance of KAN-RNN is sensitive to configuration and not yet well understood
- Similar to LSTM, KAN-RNN seems to struggle with underrepresented classes (perhaps its advantage is in data regimes with higher volume)
- KANs are a promising but early-stage approach. While they offer increased flexibility and potential generalization benefits, they currently involve trade-offs in speed and reliability.

### Future Work
- Evaluate performance on small or imbalanced datasets
- Test in larger data regimes and different domains
- Integrate KAN components into modern architectures (e.g., attention-based models)
- Improve computational efficiency and training speed

## References
- [KAN Paper (Kolmogorov–Arnold Networks)](https://arxiv.org/abs/2404.19756)
- [EfficientKAN GitHub Repository](https://github.com/Blealtan/efficient-kan)
- [Financial Sentiment Analysis Dataset (Kaggle)](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis)

## Citation
If you use this project or its codebase in your research, please cite appropriately.

## Contact Information
For questions, suggestions, or collaboration inquiries, please contact:

- Name: Michael Calderin
- Email: michaelcalderin17@gmail.com

Feel free to reach out regarding any aspect of this project.