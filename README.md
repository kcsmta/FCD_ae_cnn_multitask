<h2 align="center">
Faulty Code Detection 
</h2>

The official implementation of the paper [Semi-supervised Multi-task Learning Using Convolutional AutoEncoder for Faulty Code Detection with Limited Data]().

<!-- Table of content-->

## Table of contents
* [Introduction](#introduction)
* [Installation](#installation)
* [Usage](#usage)
* [Contact](#contact)


## Introduction
We proposes a semi-supervised multitask learning framework to overcome the limited data problem for **F**aulty **C**ode **D**etection. 
The framework has two parts including 1) a convolutional autoencoder performing the auxiliary task that is learning the latent representations of programs 
and 2) a sequence-based convolution for the main task of faulty code prediction. Our proposed model is shown below:

<p align="center">
<img src="resources/model.png" width="800" height="300" title="Multi-task learning model">
</p>

You can follow this [paper]() to see the experimental results and discussion.

## Installation

* Python 3.5+ is required.

**1. Clone project**

```sh
git clone https://github.com/kcsmta/FCD_ae_cnn_multitask.git
```

**2. Run following command:**

```sh
# Create virtual enviroment
python venv ae-cnn-multitask-env
# Active virtual enviroment
source ae-cnn-multitask-env/bin/activate
# Install requirement packages
(venv) pip install -r requirements.txt
```

## Usage

**1. Baseline model**

**2. CNN transfer learning**

**3. CNN with AE init**

**4. Multi-task model**

**5. Multi-task model with AE init**

To visualize embedding vector:

To plot ROC curve:


## Contact
Authors: \
[Anh Phan Viet](): anhpv@lqdtu.edu.vn \
[Khanh Duy Tung Nguyen]() : tungkhanh@lqdtu.edu.vn