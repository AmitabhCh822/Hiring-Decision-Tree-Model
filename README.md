# Candidate Interview Outcome Predictor Using Decision Trees

## Introduction
This project utilizes a decision tree model to predict the outcomes of candidate interviews. It is implemented in Python, leveraging libraries like pandas and sklearn. The model is based on entropy-based feature selection and evaluates various candidate attributes to predict interview success.

## Project Description
The Candidate Interview Outcome Predictor aims to analyze interview data and predict whether a candidate will do good in an interview. This is achieved through a decision tree algorithm, which considers various attributes like candidate experience level, language proficiency, social media presence, and educational background.

## Installation and Setup
To set up this project on your local machine:

1. Clone the repository:

git clone [repository URL]


2. Navigate to the project directory:

cd [local repository]


3. (Optional) If you have a specific environment or dependency setup, include those instructions here.

## Usage
To run the project:

1. Ensure you have Python installed on your machine.
2. Execute the script with:

python decision_tree_model.py


Replace `decision_tree_model.py` with the actual script name.

## Data
This project uses candidate interview data, including attributes like `level`, `lang`, `tweets`, and `grad_degree`. While the actual dataset used is confidential, the model can be adapted for similar datasets in interview outcome prediction.

## Model Details
The model employs a decision tree classifier from sklearn. It begins by calculating the entropy for different attributes and uses this information to build the tree, focusing on the attributes that offer the highest information gain.

## Evaluation and Results
The decision tree model's performance is evaluated using standard metrics such as accuracy, precision, and recall. The model achieves an accuracy of XX%, indicating its effectiveness in predicting interview outcomes.

## Contributing
Contributions to this project are welcome. Please adhere to this project's code of conduct while contributing.

## Contributor
Amitabh Chakravorty

## Contact
For queries or collaboration, feel free to reach out to me at amitabh822@gmail.com
