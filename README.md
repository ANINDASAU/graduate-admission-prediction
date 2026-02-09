# 🎓 Graduate Admission Prediction

A machine learning application that predicts the likelihood of graduate school admission using an Artificial Neural Network (ANN). Built with **Streamlit** for an interactive and user-friendly interface.

## ✨ Features

- **📊 Interactive Dashboard**: User-friendly Streamlit interface for real-time predictions
- **🧠 Advanced ML Model**: Trained Artificial Neural Network with optimized architecture
- **🎯 Comprehensive Input Parameters**: Considers 7 key factors for prediction:
  - GRE Score (260-340)
  - TOEFL Score (80-120)
  - University Ranking (1-5)
  - Statement of Purpose (SOP) Strength (1-5)
  - Letter of Recommendation (LOR) Strength (1-5)
  - Undergraduate GPA (CGPA) (6-10)
  - Research Experience (Yes/No)
- **💡 Intelligent Feedback**: Provides interpretation of prediction results
- **⚡ Fast & Accurate**: Real-time predictions with normalized data preprocessing

## 🛠️ Technologies Used

- **Python 3.x** - Core programming language
- **TensorFlow/Keras** - Deep learning framework
- **Streamlit** - Web application framework
- **NumPy** - Numerical computing
- **Scikit-learn** - Data preprocessing and scaling
- **Pandas** - Data manipulation

## 📋 Requirements

```
tensorflow
numpy
pandas
scikit-learn
streamlit
```

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd graduate-admission-prediction
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv myvenv
   myvenv\Scripts\activate  # On Windows
   source myvenv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📱 Usage

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

### How to Use:
1. Adjust the sliders for your academic and personal details
2. Select your research experience status
3. Click the **"🔮 Predict Chance of Admission"** button
4. View your predicted admission probability and personalized interpretation

## 📁 Project Structure

```
graduate-admission-prediction/
├── streamlit_app.py          # Main Streamlit application
├── model.h5                  # Trained ANN model
├── scaler.pkl                # Data scaler for preprocessing
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🧠 Model Details

- **Architecture**: Artificial Neural Network (ANN)
- **Input Features**: 7 normalized parameters
- **Output**: Admission probability (0-1)
- **Training Data**: Graduate admission dataset with historical applicant records
- **Performance**: Trained to achieve high accuracy on test data

### Data Preprocessing:
- StandardScaler used for feature normalization
- Input features scaled to 0-1 range for optimal model performance

## 📊 Prediction Interpretation

- **80%+ probability**: High chance of admission ✅
- **60-80% probability**: Good chance - Keep improving 📈
- **40-60% probability**: Moderate chance - Focus on weaker areas ⚠️
- **Below 40%**: Low chance - Significant improvement needed ❌

## 💾 Model Files

- **`model.h5`**: Serialized Keras model containing:
  - Network architecture
  - Trained weights
  - Optimization parameters
  - Activation functions

- **`scaler.pkl`**: StandardScaler object for consistent data preprocessing

## 🔧 How It Works

1. **User Input**: Collects 7 parameters via interactive sliders
2. **Preprocessing**: Scales input data using the saved scaler
3. **Prediction**: Passes normalized data through the trained ANN
4. **Output**: Returns probability (0-1) and interpretation
5. **Feedback**: Provides personalized recommendations

## 🎯 Use Cases

- **Graduate Applicants**: Assess admission chances before applying
- **University Counselors**: Guide students on profile improvements
- **Research**: Analyze factors influencing graduate admissions
- **Educational Planning**: Identify areas for improvement

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 📧 Support

For questions or issues, please open an issue on GitHub or contact the repository maintainer.

## 🎉 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) - The fastest way to build data apps
- Powered by [TensorFlow](https://www.tensorflow.org/) - End-to-end ML platform
- Data preprocessing with [Scikit-learn](https://scikit-learn.org/)

---

**Made with ❤️ for graduate applicants everywhere** 🎓

