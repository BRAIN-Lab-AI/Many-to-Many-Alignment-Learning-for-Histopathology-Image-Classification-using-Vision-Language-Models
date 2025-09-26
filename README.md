# Many-to-Many-Alignment-Learning-for-Histopathology-Image-Classification-using-Vision-Language-Models

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


## Application Area and Project Domain

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

# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

### Reference Dataset
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)


## Project Technicalities

### Terminologies
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **Latent Space:** A compressed, abstract representation of data where complex features are captured.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.
- **Tokenization:** The process of breaking down text into smaller units (tokens) for processing.
- **Noise Vector:** A randomly generated vector used to initialize the diffusion process in generative models.
- **Decoder:** A network component that transforms latent representations back into image space.
- **Iterative Refinement:** The process of gradually improving the quality of generated data through multiple steps.
- **Conditional Generation:** The process where outputs are generated based on auxiliary inputs, such as textual descriptions.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
