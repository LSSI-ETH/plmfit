## Meltome Dataset Description
The Meltome dataset offers an extensive collection of data related to protein sequences and their stability across various conditions. Emphasis is placed on understanding the thermal stability and structural integrity of proteins under different environmental factors.

### Biological Context
The Meltome dataset compiles data from a wide range of organisms, providing insights into how proteins maintain their functionality under thermal stress. This dataset is instrumental in advancing our knowledge of protein behavior, especially in response to temperature changes.

## Dataset Processing
The dataset is processed to focus on relevant attributes such as sequence length, target scores, and dataset categorization. Sequences with big lengths (over 1000 amino acids) are dropped out of the dataset. The final processed dataset is saved as `meltome_data_full.csv` for comprehensive analysis.

## Visualizations and Interpretations

### Score Distribution
![Score Distribution](./plots/score.png)
*Figure 1: Distribution of target scores in the dataset, reflecting the thermal stability or functional integrity of the proteins (Ranges between 30-90°C).*

### Sequence Length Distribution
![Sequence Length Distribution](./plots/seq_len.png)
*Figure 3: Distribution of sequence lengths. This plot helps in understanding the range of protein sizes in the dataset and filtering out extreme cases.*

## Data Analysis and Insights
The analysis focuses on the relationship between protein sequences and their stability scores. By examining these aspects, we can gain a better understanding of protein resilience and adaptability, which is crucial in fields like bioengineering, pharmaceutical development, and molecular biology.

### Target Score Interpretation
Target scores in this dataset range from low to high, indicating the varying levels of protein stability or functionality under different conditions. These scores are pivotal in identifying proteins that maintain stability or functionality under thermal stress, which is a key consideration in therapeutic and industrial applications.
