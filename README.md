# HistoAlign: A Dual-Encoder Many-to-Many Alignment Framework for Zero-Shot Histopathology Classification


## Project Metadata
### Authors
- **Team:** QURATULAN ARSHAD
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM 

## Introduction
Histopathology image analysis is a cornerstone of cancer diagnosis, prognosis, and treatment
planning. Pathologists carefully examine stained tissue slides to identify morphological and
cellular features that indicate disease. In recent years, deep learning particularly convolutional
neural networks (CNNs) and vision transformers (ViTs) has achieved remarkable success in
tasks such as cancer subtype classification and tumor segmentation [1, 2]. Despite these ad-
vances, supervised models require large quantities of annotated data, which are costly and
time-consuming to obtain since expert pathologists must provide high-quality labels.
Beyond the annotation burden, histopathology introduces domain-specific challenges. Vari-
ability in staining protocols, scanning devices, and tissue preparation, coupled with inter- and
intra-observer differences, often limits the generalizability of trained models. Consequently,
models optimized for one institution may underperform when applied to others. Additionally,
rare or newly identified cancer subtypes frequently lack sufficient labeled samples, hindering
the development of robust classifiers.


Recent advances in vision-language models (VLMs), such as CLIP [3], offer a promising
alternative. VLMs learn joint embeddings of images and text, enabling zero-shot classifica-
tion—assigning labels to classes unseen during training using only textual prompts. Medical
adaptations of these models, such as BioCLIP and MedCLIP, further enhance alignment be-
tween medical images and domain-specific textual knowledge, improving performance under
domain shift [4, 5].


Building upon this direction, the CPLIP framework (Comprehensive Pathology Language-
Image Pretraining) introduces a many-to-many alignment strategy between “bags” of tex-
tual prompts and visual patches [6]. Unlike one-to-one alignment approaches, CPLIP cap-
tures richer relationships across modalities by constructing pathology-specific prompt dictio-
naries with multiple descriptive variants (e.g., symptom, cause, morphology). This comprehen-
sive alignment, combined with contrastive learning, enables stronger generalization to unseen
classes and datasets. Empirical evaluations demonstrate that CPLIP significantly outperforms
earlier VLM-based methods such as PLIP, MI-Zero, and BiomedCLIP in both zero-shot classi-
fication and segmentation tasks.


References:

[1] Geert Litjens et al. A survey on deep learning in medical image analysis. Medical image
analysis, 42:60–88, 2017.

[2] Daisuke Komura and Shumpei Ishikawa. Machine learning methods for histopathological
image analysis. Computational and structural biotechnology journal, 16:34–42, 2018.

[3] Alec Radford et al. Learning transferable visual models from natural language supervision.
In International conference on machine learning, pages 8748–8763. PMLR, 2021.

[4] Y Zhang et al. A survey on the use of CLIP in medical imaging. arXiv preprint
arXiv:2303.08983, 2023.

[5] Xin Gu et al. Medicalclip: Contrastive learning of medical visual and language represen-
tations. arXiv preprint arXiv:2210.11372, 2022.

[6] Sajid Javed, Arif Mahmood, Iyyakutti Iyappan Ganapathi, Fayaz Ali Dharejo, Naoufel
Werghi, and Mohammed Bennamoun. Cplip: Zero-shot learning for histopathology with
comprehensive vision-language alignment. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 11450–11459, 2024.
 
## Problem Statement
Despite the promise of vision-language models for zero-shot histopathology, two key chal-
lenges remain:

• **Limited modality alignment.** Most existing approaches align each image or patch with
a single prompt (one-to-one), which fails to capture the complex and multi-faceted nature
of pathology images. Rich contextual cues such as morphology, etiology, and descriptive
variations are often underutilized. CPLIP addresses this by introducing many-to-many
alignments between prompt bags and image bags, achieving more comprehensive modal-
ity integration [6].

• **Dataset heterogeneity and domain shift.** Histopathology datasets differ across staining
protocols, scanners, resolutions, and institutions. Models trained on one dataset often
generalize poorly to others. While CPLIP improves robustness under these conditions,
further strategies are needed to handle unseen classes, rare cancer subtypes, and cross-
dataset variability.


we propose to extend and adapt vision-language zero-shot learning to histopathology image classification,
with a focus on cancer subtypes such as renal cell carcinoma, non-small cell lung carcinoma, and metastatic
breast cancer. Our goal is to design a robust and morphologically aware framework that can handle dataset 
heterogeneity, unseen classes, and limited annotations, while remaining interpretable and clinically relevant.


## Application Areas and Project Domain

The proposed framework can be applied to:

• Assist pathologists by providing rapid preliminary classification of histopathology slides.
• Improve diagnostic support in resource-limited settings with few pathologists.
• Enable transfer learning across institutions and datasets without requiring costly annotations.


## What is the paper trying to do, and what are you planning to do?
We propose to investigate the application of zero-shot learning with Vision-Language Models
(VLMs) for histopathology image classification. The method will involve the following steps:

1. **Feature Extraction:** Use pre-trained VLMs (e.g., CLIP, BioCLIP, or MedCLIP) to ex-
tract joint image-text embeddings.

3. **Zero-Shot Classification:** Formulate class-specific prompts (e.g., “clear cell renal car-
cinoma” or “lung adenocarcinoma”) and map them into the text embedding space.

4. **Alignment:** Compute similarity between histopathology image embeddings and text em-
beddings to perform classification without explicit task-specific training.

5. **Evaluation:** Compare zero-shot performance against baseline supervised methods.



### Project Documents
- **Presentation:** [Project Presentation]
- **Report:** [Project Report]

### Reference Paper
- [CPLIP: Zero-Shot Learning for Histopathology with Comprehensive
Vision-Language Alignment] https://openaccess.thecvf.com/content/CVPR2024/papers/Javed_CPLIP_Zero-Shot_Learning_for_Histopathology_with_Comprehensive_Vision-Language_Alignment_CVPR_2024_paper.pdf 

### Reference Dataset
- [BreakHis dataset]
 https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images?resource=download 

### Terminologies
- **Pathology:** Branch of medical science that is focused on the study and diagnosis of disease.
- **Histopathology:** Histopathology is the diagnosis and study of diseases of the tissues, and involves examining tissues and/or cells under a microscope.
- **Computational Pathology:** A brand-new discipline that aims to enhance patient care by utilizing advances in artificial intelligence and data
                               generated from anatomic and clinical pathology.
- **Vision Language Model:** Fusion of vision and natural language models. It ingests images and their respective textual descriptions as inputs and learns to associate the knowledge from the two modalities.
- **Weakly Supervised Learning:** Model learns from labels that are cheap, incomplete, or noisy, not perfect ground truth. There is some supervision, but it is weak.
- **Self-Supervised Learning** Model learns from unlabeled data only by creating a fake task from the data itself. No human given labels and the model makes supervision from the input.
- **Vison-Language Supervised Learning:** Model learns using paired images and text as supervision. Each training sample is (image, caption) or (image, label sentence). The model is trained so that correct text matches the image better than wrong text.
- **Zero Shot Laearning:** A Zero-shot learning is a machine learning problem in which an AI model is trained to recognize and categorize objects or concepts that it has never seen. In zero shot, we do not train the model on labeled WSI examples for each diagnostic category. Instead, the model uses text prompts(e.g., “clear cell renal carcinoma”, “benign stroma”, “invasive ductal carcinoma”) and matches them with the image features learned during pretraining.
- **Whole Slide Images:** Also called “virtual” microscopy, involves digitally scanning a tissue slide containing thin sections of tissue specimens for microscopic examination and storing it as digital images. This process allows for remote collaboration.
- **Tile-level zero-shot classification:** The ability of a machine learning model to classify individual tiles or patches of a whole slide image (WSI) into their correct categories without having been explicitly trained on those specific tiles or annotations.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.

### Problem Statements
- **Problem 1:** Most existing approaches align each image or patch with a single prompt (one-to-one), which fails to capture the complex and multi-faceted nature
                 of pathology images.
- **Problem 2:** Histopathology datasets differ across staining protocols, scanners, resolutions, and institutions. Models trained on one dataset often
                 generalize poorly to others.

### Loopholes or Research Areas
- **Weak and noisy labels instead of perfect annotations:**
- Pixel level masks for nuclei, glands, tumor regions are rare and expensive.
- Huge scope for better weakly supervised and point supervised methods that get close to fully supervised performance using cheap labels.
  
  -**Research Ideas:**

Develop a model that uses a few pixel level masks plus many slide level labels, and compare it directly with a fully supervised U-Net baseline.

Design a MIL based method that uses only slide level tumor labels but still produces pixel level heatmaps that pathologists rate as clinically meaningful.

Propose a label noise modeling method that treats each pathologist as a noisy annotator and learns a consensus label from conflicting masks.
  
- **Poor generalization across hospitals and scanners:**
- Models often drop in performance when moved from one center or stain protocol to another.
- Strong need for domain generalization and test time adaptation so one model works reliably across labs
  
  -**Research Ideas:**

Train a stain and scanner invariant representation using self supervised pretraining on multi center WSIs, then evaluate zero shot transfer to a new hospital.

Implement a test time adaptation module that adjusts feature statistics on unlabeled slides from a new center and measure gain over no adaptation.

Build a benchmark where one trains on center A and tests on centers B and C, then compare different domain generalization strategies on the same setup
  
- **Vision language and foundation models for pathology:**

- CLIP style models trained on real pathology images and pathology text are still very early.
- Great opportunity to build and evaluate pathology specific vision language models for zero shot classification and segmentation, and to make their outputs more interpretable for pathologists
  
-**Research Ideas:**

Create a small pathology specific CLIP like model using WSIs plus text from pathology reports, and test zero shot tumor subtype classification.

Explore prompt engineering for histology, for example compare simple labels (tumor, normal) versus richer prompts that include tissue type and grade.

Design a vision language system that highlights image regions while generating a short textual explanation and ask pathologists to rate usefulness and correctness.
  
### Problem vs. Ideation: Proposed Idea to Solve the Problems

**HistoAlign:** Binary Zero-Shot Histopathology Classification 

HistoAlign is a dual-encoder zero-shot learning framework for binary classification of breast histopathology images using the PLIP and CLIP models.  

The framework performs \*benign vs. malignant\* classification in a pure zero-shot setting and runs entirely on \*\*CPU\*\*, making it accessible and reproducible for researchers without GPU resources.

**Overview**

Histopathological image analysis traditionally relies on large, annotated datasets and model fine-tuning.  
\*\*HistoAlign\*\* addresses this limitation by using a dual-encoder system that aligns visual and textual features through a many-to-many cross-modal fusion strategy.  
It leverages pathology-specific and general vision encoders to capture both domain-aware and generalizable representations for robust zero-shot classification.

 **Key Features**

**Dual-Encoder Design:** Integrates PLIP (pathology-aware) and CLIP (general-purpose) encoders.  

**Many-to-Many Alignment:** Fuses all visual and textual embedding pairs for enhanced semantic alignment.  

**Clinical Prompt Engineering:** Utilizes text templates inspired by diagnostic language used in pathology reports.  

**CPU Compatibility:** Fully optimized for CPU-only systems.  

**Comprehensive Evaluation:** Automatically computes metrics, ROC-AUC, and visual outputs (confusion matrix, ROC curve).


## Project Technicalities
**System Requirements**

| Component | Specification |

|------------|---------------|

| \*\*Operating System\*\* | Windows 10/11, Ubuntu 20.04+, macOS |
| \*\*Python Version\*\* | ≥ 3.8 |
| \*\*Processor\*\* | Intel i5 / AMD Ryzen 5 or higher |
| \*\*RAM\*\* | 16 GB recommended |
| \*\*GPU\*\* | Not required |
| \*\*Storage\*\* | 10 GB free space |

**Required Libraries**

Install all dependencies before running:

```bash
pip install torch torchvision torchaudio
pip install transformers==4.46.3
pip install pandas numpy scikit-learn matplotlib tqdm pillow

**Model Workflow**

1. **Input Stage:**

Breast histopathology images (BreakHis 10-folder subset).
Textual prompts describing benign and malignant morphology.

2. **Feature Encoding:**

PLIP extracts pathology-specific visual features.
CLIP provides general visual–semantic representations.
Text prompts are encoded into class-level textual prototypes.

3. **Cross-Modal Alignment:**

Computes similarities across all encoder pairs {(V₁,T₁), (V₁,T₂), (V₂,T₁), (V₂,T₂)}.
Averages similarities to produce the final class logits.

4. **Prediction \& Output:**

Softmax applied over fused logits to compute class probabilities.
Automatically generates confusion matrix, ROC curve, and a detailed metrics report.

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.

