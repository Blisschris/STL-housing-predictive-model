# 🏠 Saint Louis House Price Predictor

This project trains a linear regression model to predict housing prices in various neighborhoods of Saint Louis, MO using synthetic data.

## 📄 Files

- `saint_louis_house_data.csv` –  housing data
- `stl_house_price_predictor.py` – Model training and evaluation script
- `model_stl.pkl` – Trained model saved via pickle
- `price_plot_stl.png` – Visual of predicted vs actual prices

## 📈 Example Output

![Plot](price_plot_stl.png)

## ▶️ How to Run

```bash
pip install -r requirements.txt
python stl_house_price_predictor.py
```

## 🔧 Features Used
- Square footage
- Bedrooms
- Bathrooms
- Age
- Neighborhood (encoded)

## 📝 License
MIT – For educational/demo use.
