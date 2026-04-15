Horse Health Prediction
The goal of this competition is to predict the health outcome of each horse case in the test set.


Horse Health Prediction

Submit Prediction
Overview
The goal of this competition is to predict the health outcome of each horse case in the test set.

This competition is designed in the style of a Kaggle Playground tabular classification challenge. Participants are given a training set with labels and a test set without labels. For each row in the test set, they must predict the target column outcome.

Start

3 hours ago
Close

16 hours to go
Description
Each row corresponds to a horse case with tabular features, including categorical and numerical medical variables. Your task is to build a machine learning pipeline that predicts the final outcome of the case.

The target variable is:

outcome
Possible classes are:

lived
died
euthanized
This competition is suitable for:

baseline tabular ML pipelines
missing-value handling
categorical encoding
feature engineering
model ensembling
leaderboard strategy and validation design
Evaluation
Submissions are evaluated using the micro-averaged F1 score between the predicted and actual values.

For multi-class classification, the micro-averaged F1 score aggregates true positives, false positives, and false negatives across all classes before computing the final F1 value.

Higher is better.

Dataset Description
Data Description
Target
The target column is:

outcome
The prediction classes are:

lived
died
euthanized
File Overview
train.csv
Contains the full training data, including:

id
Data Description
The dataset contains clinical and subjective medical measurements collected from horses. The goal is to predict the final case outcome.

Columns
Surgery: Whether the horse had surgery (1 = Yes, 2 = No)
Age: Horse age category (1 = Adult, 2 = Young)
Hospital Number: Unique case identifier
Rectal Temperature: Rectal temperature in Celsius
Pulse: Heart rate in beats per minute
Respiratory Rate: Respiratory rate
Temperature of Extremities: Indicator of peripheral circulation
Peripheral Pulse: Subjective pulse assessment
Mucous Membranes: Mucous membrane color assessment
Capillary Refill Time: Clinical refill-time judgment
Pain: Subjective pain assessment
Peristalsis: Gut activity indicator
Abdominal Distension: Severity of abdominal distension
Nasogastric Tube: Presence of gas in the tube
Nasogastric Reflux: Amount of reflux
Nasogastric Reflux pH: Reflux pH level
Rectal Examination - Feces: Feces assessment during rectal exam
Abdomen: Abdomen assessment
Packed Cell Volume: Red cell volume percentage
Total Protein: Blood protein level
Abdominocentesis Appearance: Appearance of abdominal fluid
Abdominocentesis Total Protein: Protein level in abdominal fluid
Surgical Lesion: Whether the lesion was surgical (1 = Yes, 2 = No)
Type of Lesion 1-3: Site/type/subtype/specific lesion code
CP Data: Whether pathology data is present (1 = Yes, 2 = No)
Target
Outcome: Final horse outcome (1 = Lived, 2 = Died, 3 = Euthanized)- target column outcome
test.csv
Contains the test data, including:

id
input features
This file does not include the target column.

sample_submission.csv
Shows the expected submission format.

Column Notes
id: unique row identifier
other columns: feature columns used to predict the horse outcome
outcome: target label
Typical Feature Types
Depending on the exact version of the dataset, features may include:

categorical variables
numerical laboratory values
missing values
ordinal-style medical measurements
Participants are encouraged to inspect feature types carefully and design preprocessing pipelines accordingly.

Suggested Baseline Workflow
Load the training and test data
Separate categorical and numerical columns
Impute missing values
Encode categorical columns
Train a classifier such as XGBoost or Logistic Regression
Validate using stratified cross-validation
Generate predictions for the test set
Files
3 files

Size
232.45 kB

Type
csv

License
Subject to Competition Rules

