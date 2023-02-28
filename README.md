# Early detection of Type 2 Diabetes Mellitus

Authors : AMANI Alexandra, BISCARRAT Lilian, CHAUSSARD Alexandre, CLERGUE Eva, NORMAND Sophie, SALEMBIEN Tom

This challenge was done as a project for the Master 2 Data Science (2021/2022), DATACAMP course

## Introduction

Diabetes is a growing disease, affecting over 10% of the worldwide population. The vast majority of diabetics are type 2 diabetics (96%), that is generally characterized by a resistance to insulin, or limited production of insulin by the pancreatic cells, which is a molecule that enables glucose to enter the body cells and provide energy. Not being able to properly consume sugar results in several complications that are all tied up to the high sugar level concentration in the blood due to not being able to consume it, resulting in possible blindness, hearth disease, infections, and so on.

Therefore, it is one today's most important challenge to properly diagnose diabetics, as early as possible to prevent complications from settling. At the moment, diabetes is diagnosed in multiple ways, one of the most secure and proper one being HbA1c measurement from blood samples (glycohemoglobin). The measurement of HbA1c for a sain individual is around 5.7%, while it is said to be diabetic-like when it reaches 6.5% and more.

While this technique has proved to be really efficient in the diagnosis, it has one major downside that we would like to tackle in this study. Indeed, HbA1c reflects the last 3 months of blood sugar average, therefore, in order to see that diabetes has settled, the individual has to have been severly diabetic for 3 months at least. Moreover, there exists different severities of type 2 diabetes, resulting in sometimes having to be diabetic (or pre-diabetic) for an even longer period before HbA1c becomes significant. This delay in the diagnosis could severly impact the individual, with possible irrevertible side effects.

As a result, the goal of our study is to provide a machine learning algorithm that is able to predict whether or not somebody may develop a type 2 diabetes in the coming months or years, based on blood glucose measurements from a non-invasive Continuous Glucose Monitor (CGM), as well as gathered clinical data from the patient over 3 years. The goal of the algorithm is not to provide a diagnosis of the disease, but rather to unveil risky situations so that possibly future diabetics can be taken in charge before the disease settles and causes damages. Basically, it aims at detecting early stage diabetes so more tests can be ran by doctors and possibly give early treatements.
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
