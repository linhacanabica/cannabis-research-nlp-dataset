# Cannabis Research NLP Dataset

A comprehensive dataset of scientific studies on cannabis and health, classified using fine-tuned AI models.

🔗 **Access the dataset on Kaggle:** [Cannabis Research NLP Dataset](https://www.kaggle.com/datasets/gustavospalencia/cannabis-research-nlp-dataset)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![AI Powered](https://img.shields.io/badge/AI-LLM%20Powered-orange.svg)](https://ai.google.dev/)
[![Data Source](https://img.shields.io/badge/Source-Leafwell-success.svg)](https://leafwell.com)

## Project Overview

This repository contains a structured dataset of scientific literature on cannabis and health applications, processed through an advanced AI classification pipeline. The data combines web-scraped study links from **Leafwell** with intelligent content analysis and classification using fine-tuned Large Language Models (LLMs).

### Key Features

- **Comprehensive Coverage**: 12,293 scientific studies on cannabis and health
- **AI-Powered Classification**: Fine-tuned LLM models for result categorization
- **Quality Data Sources**: Web-scraped links from Leafwell's curated database
- **Structured Format**: Clean CSV format ready for analysis and ML applications
- **Research-Ready**: Standardized fields for academic and clinical research
- **Comparative Analysis**: Both pre and post fine-tuning classifications included

📝 Release Notes  
**Title: v1.0 - AI-Enhanced Metadata & Dual Classification**

This dataset provides a comprehensive collection of over **12,293** scientific studies on the use of cannabis for medicinal and health purposes, expanded and refined through a specialized NLP pipeline. The core collection was built from links scraped from Leafwell's research database, with significant value added by the **Linha Canabica** team's data engineering and AI fine-tuning.

The key feature of this dataset is its **Dual AI-Powered Classification** for study outcome, combined with **AI-Enhanced Metadata** for context:

- `resultIA_no_fine_tunning`: The initial classification from a base Large Language Model (LLM).  
- `resultIA_fine_tunning`: The enhanced classification from the same model after a proprietary fine-tuning process developed by Linha Canabica with human feedback.

This version is the most complete, including studies from **2025** and a rigorous, granular classification of cannabinoids, conditions, and study types.

## Dataset Structure

The dataset is provided in CSV format (`studies_cannabis.csv`) with the following structure:

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Unique study identifier |
| `study_title` | String | Complete study title |
| `study_link` | URL | Source URL (scraped from Leafwell) |
| `resultIA_no_fine_tunning` | String | AI classification before fine-tuning |
| `resultIA_fine_tunning` | String | AI classification after fine-tuning |
| `study_type` | String | Study methodology type |
| `study_year` | Integer | Publication year |
| `cannabinoids` | String | Cannabinoids investigated |
| `organ_systems` | String | Affected biological systems |
| `study_conditions` | String | Medical conditions studied |

### Classification Categories

Both AI classification columns use the following categories:

- **Positive**: Statistically significant beneficial effects demonstrated
- **Negative**: Significant adverse effects or lack of therapeutic benefit
- **Inconclusive**: Mixed, insufficient, or contradictory results
- **Meta-analysis/Review**: Systematic reviews or meta-analyses of existing studies

## Methodology

### Data Collection Process

1. **Web Scraping**: Study links extracted from Leafwell's research database
2. **Content Retrieval**: Full text and abstracts gathered from source URLs
3. **Text Preprocessing**: Content cleaning and standardization
4. **Initial Classification**: Base LLM processes study content
5. **Fine-Tuning**: Model improvement through iterative human feedback
6. **Final Classification**: Enhanced model applied to complete dataset

### AI Model Training

The classification system uses a two-stage approach:

1. **Pre-Fine-Tuning**: Base model classification (stored in `resultIA_no_fine_tunning`)
2. **Post-Fine-Tuning**: Improved model after human feedback integration (stored in `resultIA_fine_tunning`)

This dual approach allows researchers to compare model performance and understand the impact of fine-tuning on classification accuracy.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://gitlab.com/[username]/cannabis-research-nlp-dataset.git
cd cannabis-research-nlp-dataset

# Install dependencies
pip install pandas matplotlib seaborn numpy
```

### Basic Usage

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('studies_cannabis.csv')

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"Year range: {df['study_year'].min()} - {df['study_year'].max()}")
print(f"Total unique studies: {len(df)}")

# Show column information
print("\nColumn summary:")
print(df.info())

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())
```

### Classification Analysis

```python
# Compare pre and post fine-tuning results
print("Classification Distribution - Before Fine-tuning:")
print(df['resultIA_no_fine_tunning'].value_counts())

print("\nClassification Distribution - After Fine-tuning:")
print(df['resultIA_fine_tunning'].value_counts())

# Calculate fine-tuning impact
agreement = (df['resultIA_no_fine_tunning'] == df['resultIA_fine_tunning']).sum()
total = len(df)
print(f"\nClassification agreement: {agreement}/{total} ({agreement/total*100:.1f}%)")
print(f"Classifications changed by fine-tuning: {total-agreement} ({(total-agreement)/total*100:.1f}%)")
```

### Advanced Analysis Examples

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize classification changes
def plot_classification_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Before fine-tuning
    df['resultIA_no_fine_tunning'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title('Classifications Before Fine-tuning')
    ax1.set_xlabel('Classification')
    ax1.set_ylabel('Count')
    
    # After fine-tuning
    df['resultIA_fine_tunning'].value_counts().plot(kind='bar', ax=ax2)
    ax2.set_title('Classifications After Fine-tuning')
    ax2.set_xlabel('Classification')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

plot_classification_comparison()

# Analyze studies by year
def analyze_temporal_trends():
    yearly_stats = df.groupby('study_year').agg({
        'id': 'count',
        'resultIA_fine_tunning': lambda x: (x == 'Positive').sum()
    }).rename(columns={'id': 'total_studies', 'resultIA_fine_tunning': 'positive_studies'})
    
    yearly_stats['positive_rate'] = yearly_stats['positive_studies'] / yearly_stats['total_studies']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Study count over time
    yearly_stats['total_studies'].plot(ax=ax1, kind='line', marker='o')
    ax1.set_title('Number of Studies by Year')
    ax1.set_ylabel('Study Count')
    
    # Positive result rate over time
    yearly_stats['positive_rate'].plot(ax=ax2, kind='line', marker='o', color='green')
    ax2.set_title('Positive Result Rate by Year')
    ax2.set_ylabel('Positive Rate')
    ax2.set_xlabel('Year')
    
    plt.tight_layout()
    plt.show()
    
    return yearly_stats

temporal_analysis = analyze_temporal_trends()
```

### Medical Condition Analysis

```python
# Analyze most studied conditions
def analyze_conditions():
    # Split conditions and count frequency
    all_conditions = []
    for conditions in df['study_conditions'].dropna():
        all_conditions.extend([c.strip() for c in conditions.split(',')])
    
    condition_counts = pd.Series(all_conditions).value_counts()
    
    print("Top 10 Most Studied Conditions:")
    print(condition_counts.head(10))
    
    # Plot top conditions
    condition_counts.head(15).plot(kind='barh', figsize=(10, 8))
    plt.title('Most Frequently Studied Medical Conditions')
    plt.xlabel('Number of Studies')
    plt.tight_layout()
    plt.show()
    
    return condition_counts

condition_analysis = analyze_conditions()

# Analyze cannabinoid research focus
def analyze_cannabinoids():
    all_cannabinoids = []
    for cannabinoids in df['cannabinoids'].dropna():
        all_cannabinoids.extend([c.strip() for c in cannabinoids.split(',')])
    
    cannabinoid_counts = pd.Series(all_cannabinoids).value_counts()
    
    print("Cannabinoid Research Distribution:")
    print(cannabinoid_counts.head(10))
    
    cannabinoid_counts.head(10).plot(kind='bar', figsize=(10, 6))
    plt.title('Cannabinoid Research Focus')
    plt.xlabel('Cannabinoid')
    plt.ylabel('Number of Studies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return cannabinoid_counts

cannabinoid_analysis = analyze_cannabinoids()
```

## Technical Implementation

### Web Scraping from Leafwell

The study links were collected from Leafwell using automated web scraping.

### AI Classification Pipeline

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CannabisStudyClassifier:
    def __init__(self, model_path="./models/cannabis_classifier"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.categories = ['Positive', 'Negative', 'Inconclusive', 'Meta-analysis/Review']
        
    def classify_study(self, title, abstract=""):
        """Classify a study based on title and optional abstract"""
        text = f"{title}. {abstract}".strip()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            
        predicted_idx = torch.argmax(probabilities, dim=-1).item()
        confidence = torch.max(probabilities).item()
        
        return {
            'classification': self.categories[predicted_idx],
            'confidence': confidence,
            'probabilities': {
                cat: prob.item() 
                for cat, prob in zip(self.categories, probabilities[0])
            }
        }

    def batch_classify(self, studies):
        """Classify multiple studies efficiently"""
        results = []
        for study in studies:
            result = self.classify_study(study['title'], study.get('abstract', ''))
            results.append(result)
        return results

# Example usage
classifier = CannabisStudyClassifier()
result = classifier.classify_study(
    title="CBD reduces seizure frequency in treatment-resistant epilepsy patients"
)
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## Data Quality and Validation

### Quality Metrics

- **Dataset Size**: 7640 unique studies
- **Date Coverage**: 1975-2024 (49 years)
- **Classification Agreement**: Analysis of pre vs post fine-tuning results
- **Source Validation**: All links verified for accessibility
- **Duplicate Detection**: Automated removal based on title similarity

### Fine-Tuning Performance

The model improvement through fine-tuning can be measured by comparing the two classification columns:

```python
# Calculate classification stability and changes
def analyze_fine_tuning_impact(df):
    # Classifications that remained the same
    stable = df[df['resultIA_no_fine_tunning'] == df['resultIA_fine_tunning']]
    
    # Classifications that changed
    changed = df[df['resultIA_no_fine_tunning'] != df['resultIA_fine_tunning']]
    
    print(f"Stable classifications: {len(stable)} ({len(stable)/len(df)*100:.1f}%)")
    print(f"Changed classifications: {len(changed)} ({len(changed)/len(df)*100:.1f}%)")
    
    # Analyze change patterns
    change_patterns = changed.groupby(['resultIA_no_fine_tunning', 'resultIA_fine_tunning']).size()
    print("\nClassification change patterns:")
    print(change_patterns)
    
    return stable, changed

stability_analysis = analyze_fine_tuning_impact(df)
```

## Use Cases

### Academic Research
- Literature reviews and systematic analysis
- Meta-analysis preparation with pre-classified studies
- Research trend analysis over time
- Interdisciplinary cannabis research exploration

### Clinical Applications
- Evidence-based treatment planning
- Clinical trial design and hypothesis generation
- Therapeutic indication research
- Safety and efficacy assessment

### Machine Learning
- Benchmark dataset for medical text classification
- Transfer learning for healthcare NLP
- Model comparison and evaluation
- Fine-tuning methodology research

### Industry and Policy
- Regulatory submission support
- Market research and competitive analysis
- Product development guidance
- Policy development evidence base

## Contributing

We welcome contributions from the research community:

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add improvement'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Merge Request

### Contribution Areas
- **Data Quality**: Report errors or suggest improvements
- **Analysis Tools**: Add new analysis functions or visualizations
- **Documentation**: Improve guides and examples
- **Model Performance**: Suggest classification improvements
- **New Features**: Add functionality or export formats

### Usage Terms
- Free for academic and commercial use
- Attribution required in publications (Leafwell, Linha Canabica, Gustavo Palencia)
- Modifications and redistribution allowed
- No warranty provided

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{cannabis_research_nlp_2024,
  title={Cannabis Research NLP Dataset: AI-Powered Scientific Literature Classification},
  author={[Author Name]},
  year={2024},
  publisher={GitLab},
  url={https://gitlab.com/[username]/cannabis-research-nlp-dataset},
  note={Data scraped from Leafwell, classified using fine-tuned LLMs}
}
```

## Support

### Contact Information
- **Project Lead**: [Gustavo Palencia] - [talkto@gustavospalencia.com]
- **Technical Support**: [talkto@gustavospalencia.com]
- **Issues**: [GitLab Issues Page](https://gitlab.com/[username]/cannabis-research-nlp-dataset/-/issues)

## Donations

If you find this collection useful and want to support the project, you can make a donation in USDT/Bitcoin or PIX Brazil:  

- **USDT Address:** TP9mKAAsGxCaJ7kXAmPMizHPmsCJYDMhGE
- **Network:** TRC20

- **BTC Address:** 1Je9R3jnXPwVsx7cwoLFSiPeJwVhEfGwY3
- **Network:** BTC  

Or PIX Brazil: gustavo.palencia@pague.com.gi

Thank you for your support! 🙏

## Acknowledgments

- **Leafwell** for providing access to curated cannabis research database
- **Linha Canabica** Brazil’s medicinal-cannabis marketplace with 50,000+ users, ~13,000 monthly organic visits
- **Open Source Community** for tools and libraries used in this project
- **Research Contributors** for validation and feedback
- **Cannabis Research Community** for domain expertise

---

**Version**: 3.0  
**Last Updated**: 2025  
**Dataset File**: `studies_cannabis.csv`  
**Total Studies**: 7640
