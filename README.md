# Predicting Type 2 Diabetes Mellitus with non-invasive measurements

Authors : AMANI Alexandra, BISCARRAT Lilian, CHAUSSARD Alexandre, CLERGUE Eva, NORMAND Sophie, SALEMBIEN Tom

This challenge was done as a project for the Master 2 Data Science (2021/2022), DATACAMP course

## Introduction

Diabetes is a growing disease, affecting over 10% of the worldwide population. The vast majority of diabetics are type 2 diabetics (96%), that is generally characterized by a resistance to insulin, which is a molecule that enables glucose to enter the cells and provide energy. Not being able to properly consume sugar results in several complications that are all tied up to the high sugar level concentration in the blood due to not being able to consume it, resulting in possible blindness, hearth disease, infections, and so on.

Therefore, it is one today's most important challenge to properly diagnose diabetics, as short as possible to prevent complications to settle. At the moment, diabetes is diagnosed in multiple ways, one of the most secure and proper one being HbA1c measurement from blood samples. The measurement of HbA1c for a sain individual is around 5.7%, while it is said to be diabetic-like when it reaches 6.5% and more.

While this technique has proved to be really efficient in the diagnosis, it has two main downsides that we would like to tackle in this study:

- First, HbA1c is a post-disease settlement indicator, as it requires the patient to have abnormally high sugar levels over 3 months to be indicating the disease. One would rather like to be able to diagnose diabetes before it reaches that critical point.
- Second, HbA1c being a blood measurement, which is naturally invasive for the patient and some may even be really reluctant and refuse to go for a diagnosis. Also, it requires careful analysis in a medical lab, which obviously costs money.

As a result, the goal of our study is to provide a machine learning algorithm that is able to predict whether or not somebody may become diabetic in the coming time, based on blood glucose measurements from a non-invasive Continuous Glucose Monitor (CGM), as well as gathered clinical data for the patient over the duration of the [reference study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0225817#sec018).

## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

Then you will need to download and preprocess the dataset by running the following command:
```
python download_data.py
```

In addition to the training/testing sets, this command will generate an `external_data.csv` file containing continuous glucose measurements over 48h for each individual. You can use that file in your workflow to build relevant features.

### Challenge description

Get started with the [starting kit notebook](DT2_starting_kit.ipynb)


### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further
