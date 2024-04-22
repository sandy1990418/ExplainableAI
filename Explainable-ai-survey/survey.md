# Explainable AI Survey
Briefly introduce methods of explainable ai in LLM

## Overview
Explainable AI is crucial for ensuring transparency, trustworthiness, and accountability in AI systems, as it enables users to understand how AI models arrive at their decisions or predictions, fostering collaboration, regulatory compliance, and domain understanding, ultimately leading to the development of AI systems that are transparent, trustworthy, accountable, and aligned with human needs and values.

<br>

## Table of Contents
1. [Local Explanation](#local-explanation)
    - [NLP-Explanation](#nlp-explanation)
    - [Example-Based-Explanation](#example-based-explanation)
    - [Feature-Attribution](#feature-attribution)
    - [Attention Based](#attention-based)

2. [Global Explanation](#global-explanation)
    - [Probing-Based-Explanation](#probing-based-explanation)
    - [Neuron-Activation-Explanation](#neuron-activation-explanation)
    - [Concept-Based-Explanation](#concept-based-explanation)


<br>


## Local explanation V.S. Global explanation
In explainable AI, "Local explanation" refers to examining the impact of different examples on the model, while "Global explanation" pertains to understanding what the model has learned overall. Below, we will introduce methods focusing on both Local and Global perspectives:

<br>

## Local-Explanation

### NLP-Explanation

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

### Example-Based-Explanation

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

### Feature-Attribution

Feature Attribution assess the relevance of each input feature to model predictions.
    

#### Method 1 - Perturbation-Based Explanation
This method investigates output variations under different input perturbations (such as removing, masking, or altering input features), where preserving important input information should yield predictions similar to the original ones.

![Perturbation-Based Explanation](https://miro.medium.com/v2/resize:fit:786/format:webp/1*FU-IFEBAg0ipWx4dX5wveA.png)
[source](http://heatmapping.org/)


- Pros
    1. Captures non-linear feature interactions.

- Cons
    1. Sensitivity to perturbation magnitude.

<br>

#### Method 2 - Gradient-Based Explanation
This approach considers higher gradients or feature values as indicative of higher importance, analyzing partial derivatives of outputs with respect to each input dimension to determine the significance of each input feature.

e.g.: Saliency map, Grad-CAM


![Saliency map](https://media.geeksforgeeks.org/wp-content/cdn-uploads/20210722235025/Saliency-maps-generated-using-Grad-CAM-Grad-CAM-and-Smooth-Grad-CAM-respectively_W640.jpg)
[source](https://www.geeksforgeeks.org/what-is-saliency-map/)


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
[source](https://github.com/prashanth41/Layer-wise_relevance_propagation)

- Pros
    1. Model-agnostic: Applicable to a wide range of machine learning models without architectural modifications.

- Cons
    1. Vulnerable to noisy gradients.
    2. Complexity with high-dimensional data.

<br>

### Attention-Based
Visualizing attention heads using bipartite graphs or heatmaps to understand the focus of the model on individual inputs.

![Attention](https://github.com/jessevig/bertviz/blob/master/images/head-view.gif?raw=true)

[source](https://github.com/jessevig/bertviz/tree/master)


- Pros
    1. Enhances interpretability by providing visual representations of model attention.

- Cons
    1. May not accurately capture the importance of features compared to other explanation methods.
    2. Limited in capturing syntactic structures present in text.

<br>

## Global-Explanation
 Understanding the encoded content of individual components (neurons, hidden layers, and larger modules) and explaining the knowledge/language properties learned by each component.

<br>

### Probing-Based-Explanation

#### Method 1 - Classifier-Based Probing
Training a shallow classifier on top of pre-trained or fine-tuned language models (e.g., BERT, T5) to identify specific linguistic properties or reasoning capabilities acquired by the model.
        
The pre-trained model parameters are frozen, including representations, phrases, and attention weights, which are then fed into a probing classifier tasked with recognizing certain language attributes or inferential abilities.

<br>

#### Method 2 - Parameter-Free Probing
Designing datasets tailored to specific language properties (e.g., syntax) to measure whether a language model's probability of positive instances exceeds that of negative instances.


- Pros
    1. Identification of Linguistic Properties:Interpret the language models' performance across various linguistic tasks and facilitate model improvement.
    2. Flexible Application: Applicable to various pre-trained models, enabling analysis across different architectures.

- Cons
    1. Human Resource for Annotating:Requires significant annotated data to train shallow classifiers.
    2. Dependency on Pre-Trained Models:The accuracy and credibility of explanations rely heavily on the capabilities of pre-trained models, potentially limiting their applicability and reliability in diverse scenarios.
    3. Limitations in Capturing Complex Cross-Linguistic Structures:Due to language complexity, these techniques may struggle to offer comprehensive insights, leading to potential gaps or inaccuracies in interpreting language models' behavior.

<br>

### Neuron-Activation-Explanation
Neuron activation explanation assumes that models with similar learning characteristics often share similar neurons. Based on this assumption, shared neurons can be ranked using different indicators such as correlation measures and learning weights, allowing for a better understanding of their contributions to model behavior and learning characteristics.


Steps for Neuron Activation Explanation:
1. Identify important neurons through unsupervised learning (based on statistical properties or information within the model to identify neurons with significant impact on model performance).
2. Establish the relationship between language features and individual neurons through supervised learning.

<br>

- OpenAI's Application in Explaining GPT Models: <br>
OpenAI employs Neuron Activation Explanation to elucidate the workings of their GPT (Generative Pre-trained Transformer) models. For instance:

    **Step 1 : Explanation Generation**:<br>
    OpenAI's GPT-4 generates explanations based on provided text, focusing on specific content domains like Marvel. It assesses text and activation patterns to determine relevance to movies, characters, and entertainment.
    **Step 2 : Neuron Simulation**:<br>
    Using GPT-4, OpenAI simulates the future actions of GPT-2 neurons based on received text and activation patterns, predicting the next steps in processing information.
    **Step 3 : Comparison and Evaluation**:<br>
    Comparing and evaluating scores of simulated neurons by GPT-4 with real neurons of GPT-2 assesses the accuracy of GPT-4's predictions.

![openai](https://img.36krcdn.com/hsossms/20230511/v2_00d5d8f520064baabb24d0e8e25bc8d6@5888275_oswg539098oswg1080oswg1046_img_000?x-oss-process=image/format,jpg/interlace,1)
[source](https://openai.com/research/language-models-can-explain-neurons-in-language-models),[github](https://github.com/openai/automated-interpretability)


- Microsoft - Summarize and Score (SASC) explanation

    **Step 1: Summarization**
    - Extract unique ngrams from a specified text corpus.
    - Evaluate ngrams using the module f to identify those eliciting the largest positive response.
    - Use a helper language model (LLM) for summarization, selecting a random subset of top ngrams.

    **Step 2: Synthetic Scoring**
    - Generate synthetic data for each candidate explanation using the LLM.
    - Compare module responses to text related to the explanation (Text+) versus unrelated synthetic text (Text−).
    - Compute an explanation score as the difference between responses, reported in standard deviation units (σf).


![openai](https://s4.itho.me/sites/default/files/images/FireShot%20Capture%201770%20-%20Language%20models%20can%20explain%20neurons%20in%20language%20models%20-%20openai_com.png)

[source](https://github.com/microsoft/automated-explanations?tab=readme-ov-file),[github](https://github.com/microsoft/automated-explanations?tab=readme-ov-file)

<br>

- Pros
    1. Faster and more efficient compared to methods producing token-level activations.
    2. Simplifies explanations by describing functions rather than producing token-level activations.

- Cons
    1. Less effective at finding patterns dependent on sequences or ordering.
    2. May not capture intricate details present in token-level activations.

<br>

### Concept-Based-Explanation
Measures the relevance of concepts understandable by humans in model outputs. 

[TCAV](https://beenkim.github.io/slides/TCAV_ICML_pdf.pdf), developed by Google, is a renowned Concept-Based Explanation method. The steps of TCAV are as follows:

**Step 1: Define** <br>
Define the concept of interest.

**Step 2: Measure**  <br>
Generate TCAV scores to measure the sensitivity of predictions to specific concepts.
   - Linear classifier: Train a linear classifier to separate activations.
   - CAV is the vector orthogonal to the decision boundary.

**Step 3: Evaluate**  <br>
Evaluate the global relevance of learned CAVs.


![tcva](https://miro.medium.com/v2/resize:fit:828/format:webp/1*bRaiBkyf1165WT9xiYu9vA.png)
[source](https://beenkim.github.io/slides/TCAV_ICML_pdf.pdf)


- Pros
  1. No Machine Learning Expertise Required<br>
     Users can utilize TCAV without extensive ML knowledge, making it accessible for domain experts to evaluate complex neural network models.
  2. Customizability<br>
     TCAV offers flexibility beyond feature attribution, allowing users to investigate any concept defined by their dataset. Users can adjust the complexity of explanations based on their understanding of the problem.

- Cons
  1. Performance on Shallow Neural Networks<br>
     TCAV may not perform well on shallow networks as concepts in deeper layers are typically more separable.
  2. Difficulty with Abstract Concepts<br>
     TCAV is challenging to apply to abstract or general concepts due to the need for extensive training data, limiting its effectiveness for concepts like "happiness.”
  3. Annotation Cost<br>
     TCAV requires additional annotations for concept datasets, making it expensive for tasks without readily available labeled data.
  4. Limited Application in Text and Tabular Data<br>
     While popular for image data, TCAV's applicability to text and tabular data is relatively restricted.


<br>

## References
[1] Zhao, H., Chen, H., Yang, F., Liu, N., Deng, H., Cai, H., Wang, S., Yin, D., & Du, M. (2024). Explainability for large language models: A survey. ACM Transactions on Intelligent Systems and Technology, 15(2), 1–38. https://doi.org/10.1145/3639372

[2] [可解釋 AI (XAI) 系列 — 01 基於遮擋的方法 (Perturbation-Based): Occlusion Sensitivity, Meaningful Perturbation](https://medium.com/ai-academy-taiwan/%E5%8F%AF%E8%A7%A3%E9%87%8B-ai-xai-%E7%B3%BB%E5%88%97-01-%E5%9F%BA%E6%96%BC%E9%81%AE%E6%93%8B%E7%9A%84%E6%96%B9%E6%B3%95-perturbation-based-40899ba7e903)