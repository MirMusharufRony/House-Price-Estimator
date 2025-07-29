
# King County House Price Prediction & Geographical Visualization

This project explores **house sales data for King County (Seattle area)**, performing **exploratory data analysis (EDA)**, **geographical visualization**, and building a **neural network model** to predict house prices.  

The dataset contains homes sold between May 2014 and May 2015, including features like price, square footage, number of bedrooms, location (latitude/longitude), and more.

---

## Project Workflow
### 1. Data Loading & Preprocessing
- Loaded `kc_house_data.csv`  
- Converted `date` column to `datetime`  
- Extracted `year` and `month` from sale dates  
- Dropped unnecessary columns (`id`, `zipcode`, `date`)  
- Checked for missing values and outliers  

### 2. Exploratory Data Analysis (EDA)
Used **Matplotlib** and **Seaborn** to explore:
- **Price distribution** (skewed towards lower values)  
- **Bedroom count distribution**  
- **Correlation of price with features** (`sqft_living`, `grade`, etc.)  
- **Boxplots** for price vs. bedrooms, waterfront view, and sale month  
- **Scatterplots** for price vs. living area, price vs. grade  

### 3. Geographical Visualization
- **Plotly Mapbox**: Interactive map showing house locations.  
  - Color-coded by **price**.  
  - Point size proportional to **sqft_living**.  
  - Hover tooltips for **price, sqft, bedrooms**.  
- **Seaborn scatterplots**: Static geographical distribution plots (latitude vs. longitude with price coloring).  

### 4. Feature Scaling
- Used **MinMaxScaler** to normalize features for the neural network.

### 5. Neural Network Model
Built a **deep learning regression model** using **TensorFlow/Keras**:
- **4 hidden layers** with `relu` activation  
- **Output layer**: Single neuron (predicting price)  
- **Optimizer**: Adam  
- **Loss function**: Mean Squared Error (MSE)  

### 6. Training & Evaluation
- Split data: **70% training / 30% testing**  
- Trained over **400 epochs** with a batch size of 128.  
- Evaluated using:
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**
  - **Explained Variance Score**
- Plotted **training & validation loss curves**.

---

## Technologies Used
- **Python**  
- **Pandas, NumPy** – Data processing  
- **Matplotlib, Seaborn** – Visualizations  
- **Plotly (Mapbox)** – Interactive geographical mapping  
- **Scikit-learn** – Feature scaling & metrics  
- **TensorFlow/Keras** – Neural network model  

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/king-county-house-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python main.py
   ```
4. For the interactive map:
   - The Plotly map will open in your **default browser**.
   - Or view `house_map.html` (auto-generated) in any browser.

---

## Results
- Built a regression model to predict house prices with good accuracy.  
- Interactive map provides clear insights into **location-based pricing trends**.  
- Price strongly correlates with **sqft_living**, **grade**, and **waterfront presence**.  

---

## Dataset
The dataset (`kc_house_data.csv`) includes:
- **Price** (target)  
- **Bedrooms, bathrooms, sqft_living, sqft_lot**  
- **Floors, condition, grade**  
- **Waterfront, view**  
- **Latitude & longitude** (for mapping)  
- **Year built & renovated**  
- **Sale date**  

---

## Future Improvements
- Add **hyperparameter tuning** for the neural network.  
- Experiment with **ensemble models** (Random Forest, XGBoost).  
- Create a **web dashboard** using Streamlit for real-time predictions.  
