# Electric Motor Temperature Prediction

## Overview

This project focuses on predicting the permanent magnet (PM) surface temperature in electric motors using machine learning models. The system analyzes sensor data from electric motors to forecast temperature, which is crucial for preventing overheating and ensuring operational safety.

The project includes:
- Data analysis and model training using Jupyter Notebook
- A web application for manual temperature predictions
- Pre-trained machine learning models

## Screenshots: 
Landing Page:
![Screenshot_25-12-2025_201240_192 168 0 100](https://github.com/user-attachments/assets/a9522c17-e2fc-4f3b-b41c-1c5739f12679)

Prediction Page:
![Screenshot_25-12-2025_201955_192 168 0 100](https://github.com/user-attachments/assets/e048b0c0-feba-45ab-a1a8-532c7cdea6e1)

Analysis:
<img width="1255" height="529" alt="image" src="https://github.com/user-attachments/assets/4c01d23c-ebf3-4ff8-b86f-fcdfa24c281e" />
<img width="1636" height="833" alt="image15" src="https://github.com/user-attachments/assets/988e39e1-48b7-40e0-8e07-725955941f41" />
<img width="1202" height="907" alt="image16" src="https://github.com/user-attachments/assets/9ed345e7-5b8f-447c-a8c3-10fa5aa712ea" />

## Features

- **Data Analysis**: Comprehensive exploratory data analysis (EDA) including univariate and multivariate analysis
- **Machine Learning Models**: Implementation of multiple regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Support Vector Regressor
- **Model Evaluation**: Performance comparison using RMSE and R-squared metrics
- **Web Interface**: Flask-based web application for real-time predictions
- **Data Preprocessing**: Feature scaling and normalization using MinMaxScaler

## Project Structure

```
Electric_Motor_Temperature_Prediction/
├── app.py                          # Flask web application
├── commit-agent.py                 # Commit message generation utility
├── measures_v2.csv                 # Dataset containing motor sensor measurements
├── Rotor Temperature Detection.ipynb  # Jupyter notebook for data analysis and model training
├── model_dr.save                   # Saved Decision Tree model
├── model_lr.save                   # Saved Linear Regression model
├── model_rf.save                   # Saved Random Forest model
├── model_svm.save                  # Saved SVM model
├── transform.save                  # Saved MinMaxScaler transformer
└── templates/
    └── Manual_predict.html         # HTML template for prediction interface
```

## Dataset

The dataset (`measures_v2.csv`) contains sensor measurements from electric motors with the following features:
- `ambient`: Ambient temperature
- `coolant`: Coolant temperature
- `u_d`, `u_q`: Voltage components (d and q axes)
- `motor_speed`: Motor speed
- `i_d`, `i_q`: Current components (d and q axes)
- `pm`: Permanent magnet surface temperature (target variable)
- `stator_yoke`, `stator_tooth`, `stator_winding`: Stator temperatures
- `torque`: Motor torque
- `profile_id`: Measurement profile identifier

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Electric_Motor_Temperature_Prediction
   ```

2. Install required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn joblib flask
   ```

3. For the commit agent utility (optional):
   ```bash
   pip install python-dotenv langchain-google-genai langchain-core langgraph
   ```
   Set up your Google API key in a `.env` file:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

### Running the Web Application

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:5000`

3. Enter the sensor values in the form and click "Predict" to get the temperature prediction.

### Data Analysis and Model Training

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook "Rotor Temperature Detection.ipynb"
   ```

2. Run the cells sequentially to:
   - Load and explore the dataset
   - Perform data preprocessing
   - Train machine learning models
   - Evaluate model performance
   - Save trained models

### Using the Commit Agent

The `commit-agent.py` script generates conventional commit messages based on staged Git changes:

```bash
python commit-agent.py
```

## Model Performance

Based on the evaluation in the notebook, the models achieved the following performance metrics:

- **Linear Regression**: RMSE and R² scores
- **Decision Tree**: RMSE and R² scores  
- **Random Forest**: RMSE and R² scores
- **SVM**: RMSE and R² scores

(The web application uses the Decision Tree model for predictions)

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib
- flask
- (Optional) python-dotenv, langchain-google-genai, langchain-core, langgraph

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open-source.

## Contact

For questions or suggestions, please open an issue in the repository.
