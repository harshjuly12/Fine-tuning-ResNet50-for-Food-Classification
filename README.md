<table>
  <tr>
    <td><img src="https://github.com/harshjuly12/Fine-tuning-ResNet50-for-Food-Classification/assets/112745312/2013d017-4591-4860-9b9a-0536204d1169" width="120" style="margin-right: 10;"></td>
    <td><h1 style="margin: 0;">Fine Tuning ResNet50 For Food Classification & Calories Estimation</h1></td>
  </tr>
</table>

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Analysis and Results](#analysis-and-results)
8. [Contributing](#contributing)
9. [License](#license)
10. [Author](#author)
    
## Project Overview
Develop a model that can accurately recognize food items from images and estimate their calorie content, enabling users to track their dietary intake and make informed food choices.

## Dataset
The dataset used for this project can be found on Kaggle: [Food-101](https://www.kaggle.com/dansbecker/food-101)

## Project Structure
The project is structured as follows:
- **Understand dataset structure and files**: Analyze the dataset to understand its organization and contents.
- **Visualize random image from each of the 101 classes**: Display random images to get a sense of the data distribution.
- **Split the image data into train and test using train.txt and test.txt**: Prepare the data for training and testing the model.
- **Create a subset of data with few classes (3) - train_mini and test_mini for experimenting**: Use a smaller subset of the data for initial experiments.
- **Fine-tune ResNet50 Pretrained model using Food 101 dataset**: Apply transfer learning to adapt a pre-trained model to the specific task.
- **Visualize the accuracy and loss plots**: Plot the training history to evaluate model performance.
- **Predicting classes for new images from the internet using the best trained model**: Test the model on new, unseen images.
- **Fine-tune ResNet50 model with 11 classes of data**: Further refine the model using a larger subset of the data.

## Requirements
To run the project, you need the following libraries:
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- TensorFlow or PyTorch (depending on the chosen framework)
- OpenCV

## Installation
1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd FoodCaloriesEstimation
   ```
2. **Install the required Python libraries:**
   ```sh
   pip install -r requirements.txt
   ```
## Usage
1. **Navigate to the project directory:**
   ```sh
   cd FoodCaloriesEstimation
   ```
2. **Run the Jupyter notebook for detailed steps and execution:**
   ```sh
   jupyter notebook FoodCaloriesEstimation.ipynb
   ```
**Technical Method
This project employs the following technical method:
Transfer Learning with ResNet50: A pre-trained ResNet50 model is fine-tuned on the Food-101 dataset to leverage its existing feature extraction capabilities for the specific task of food classification and calorie estimation.**

## Analysis and Results
The notebook contains the following steps:
1. Importing Libraries: Importing necessary libraries for analysis and visualization.
2. Data Exploration: Exploring the dataset to understand the distribution and relationships between different variables.
3. Data Preprocessing: Preparing the data for training by resizing images and normalizing pixel values.
4. Model Training: Training the ResNet50 model on the dataset.
5. Evaluation: Evaluating model performance using accuracy and loss metrics.
6. Prediction: Predicting food classes for new images and estimating their calorie content.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
For any questions or suggestions, please contact:
- Harsh Singh: [harshjuly12@gmail.com](harshjuly12@gmail.com)
- GitHub: [harshjuly12](https://github.com/harshjuly12)
