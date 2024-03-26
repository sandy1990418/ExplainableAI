# Explainable AI
Briefly introduce methods of explainable ai in LLM

## Overview
Explainable AI is crucial for ensuring transparency, trustworthiness, and accountability in AI systems, as it enables users to understand how AI models arrive at their decisions or predictions, fostering collaboration, regulatory compliance, and domain understanding, ultimately leading to the development of AI systems that are transparent, trustworthy, accountable, and aligned with human needs and values.

<br>


## Local Test V.S. Global Test
In explainable AI, "Local test" refers to examining the impact of different examples on the model, while "Global Test" pertains to understanding what the model has learned overall. Below, we will introduce methods focusing on both Local and Global perspectives:

<br>

## Local Test

### NLP Explanation

We use raw text data and manually annotated explanations to train a new language model, helping us understand the logic behind language model-generated text.

- **Pros**:
    1. **Easy to Understand**:
        Using natural language explanations makes the decision-making process of the model understandable even to individuals without specialized knowledge.
    2. **Feedback Integration**:
        Feedback from users or domain experts can be integrated to iteratively improve explanations and the underlying model.

- **Cons**:
    1. **Subjective**:
        Influenced by the subjectivity of training data and manual annotations, potentially leading to biased explanations.
    2. **Human Resource for Annotation**:
        Lack of annotated datasets for this purpose requires significant human effort for annotation.
    3. **Constraint**:
        May not cover all possible explanations, especially when dealing with domain-specific terms.

<br>

### Example-Based Explanation

Example-based explanation methods utilize various cases to observe how changes in data affect the model's behavior, thereby explaining its predictive mechanisms.

#### Method 1 - Counterfactual Explanations

Counterfactual explanations involve minimal changes to features to examine which alterations would lead to desired predictions, offering insights into feature importance. For instance, if a loan application is rejected by the model, identifying which feature changes (e.g., income, credit card count, age) could alter the prediction to approval.

<br>

#### Method 2 - Adversarial Examples

Adversarial examples aim to deceive the model by making slight changes in input data, demonstrating vulnerabilities in the model's decision-making process. For instance, 1-pixel attacks manipulate images in imperceptible ways to provoke incorrect predictions.

![Adversarial Example](https://res.cloudinary.com/practicaldev/image/fetch/s--hDIjj9cc--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://miro.medium.com/max/610/1%2AH9zKuBbxlB6ZsvPPfIKOGw.png)
***source:https://reurl.cc/80ERbg***

- **Pros**
  1. **Human-friendly**: These explanations are easy for humans to understand and interpret.
  1. **Selective**: They focus on a small number of feature changes, which can highlight the most influential factors affecting a prediction.
  1. **Informative**: Counterfactual explanations provide actionable insights by suggesting changes that could lead to a different outcome, aiding users in understanding how to improve their results.

- **Cons**
  1. **Rashomon effect**: Different interpretations of model behavior may seem reasonable when changing feature directions, leading to contradictory explanations.
  1. **Subjective**: Interpretations may be influenced by subjective perspectives or biases.
  1. **Computational complexity**: Computing different combinations of feature changes can be computationally intensive.

<br>

### Feature Attribution

Feature Attribution assess the relevance of each input feature to model predictions.
    

#### Method 1 - Perturbation-Based Explanation
This method investigates output variations under different input perturbations (such as removing, masking, or altering input features), where preserving important input information should yield predictions similar to the original ones.

![Perturbation-Based Explanation](https://miro.medium.com/v2/resize:fit:786/format:webp/1*FU-IFEBAg0ipWx4dX5wveA.png)
***source:http://heatmapping.org/***
- Pros
    1. Captures non-linear feature interactions.

- Cons
    1. Sensitivity to perturbation magnitude.

<br>

#### Method 2 - Gradient-Based Explanation
This approach considers higher gradients or feature values as indicative of higher importance, analyzing partial derivatives of outputs with respect to each input dimension to determine the significance of each input feature.

e.g.: Saliency map, Grad-CAM


![Saliency map](https://media.geeksforgeeks.org/wp-content/cdn-uploads/20210722235025/Saliency-maps-generated-using-Grad-CAM-Grad-CAM-and-Smooth-Grad-CAM-respectively_W640.jpg)
***source:https://www.geeksforgeeks.org/what-is-saliency-map/***


- Pros
    1. Model-agnostic: Applicable to a wide range of machine learning models without architectural modifications.

- Cons
    1. Vulnerable to noisy gradients.
    2. Complexity with high-dimensional data.
    
<br>

#### Method 3 - Surrogate Models
Surrogate models simplify the complex and non-linear relationship between input space and prediction outputs in black-box models, using simpler and more understandable models to explain individual predictions. Common surrogate models include decision trees, linear models, decision rules, and other white-box models.
    
- Pros
    1. Flexible explanation techniques: Offer flexibility in selecting explanation methods according to specific interpretability needs or domain requirements.

- Cons
    1. Loss of fidelity: May not fully capture the complexity of the original black-box model, resulting in oversimplified or inaccurate representations of the true model behavior.
    
<br>

#### Method 4 - Decomposition-Based Methods
These methods analyze the contribution of each input to the prediction outcome through linear combinations.
    
e.g. : LRP, DTD

![LRP](https://raw.githubusercontent.com/prashanth41/Layer-wise_relevance_propagation/master/lrp.png)
***source:https://github.com/prashanth41/Layer-wise_relevance_propagation***

