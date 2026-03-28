# HDB Resale Flat Pricing — Machine Learning Pipeline

## Overview
This project explores the key factors influencing HDB resale flat prices in Singapore 
and develops a machine learning model to predict resale value based on property 
attributes and geospatial features.

The analysis follows a full end-to-end data science pipeline: data cleaning, 
exploratory data analysis, feature engineering, machine learning modelling, 
and post-model analysis.

## Dataset
Source: [data.gov.sg — HDB Resale Flat Prices](https://data.gov.sg)  
193,876 HDB resale transactions across Singapore.

## Team Roles
| Member | Role |
|--------|------|
| Chew & Isabel | Data Cleaning & Engineering |
| Mabel | Exploratory Data Analysis & Post-ML EDA |
| Darryl | Feature Engineering |
| Moe | Machine Learning Modelling |

## Pipeline

### 1. Data Cleaning
- Removed duplicates and standardised data types
- Parsed storey range into numeric average floor values
- Converted remaining lease strings to decimal years
- Removed outliers in floor area (>350 sqm)

### 2. Exploratory Data Analysis
- Distribution analysis of resale price, floor area, and remaining lease
- Relationship analysis between price and structural/locational attributes
- Temporal trend analysis by quarter and year
- Correlation heatmap of key numeric variables

### 3. Feature Engineering (Darryl)
Geospatial features derived via the **OneMap API**:
- Postal codes and coordinates (latitude/longitude) for each HDB block
- **MRT proximity**: distance to nearest station, within-500m and within-1km flags
- **CBD distance**: straight-line and estimated public transport/driving time
- **School proximity**: nearest top primary, secondary, JC, and polytechnic
- **Amenity cluster scores**: malls, hawker centres, hospitals, bus interchanges,
  supermarkets (Cold Storage, FairPrice, Giant, Sheng Siong)
- **Noise scores**: based on MRT proximity and floor level
- **Floor features**: floor tier, floor category, floor premium factor
- **Lease features**: estate age buckets, mature estate and very old estate flags
- **Transaction history**: block-level and town-level transaction counts
  and cumulative counts

### 4. Machine Learning Modelling
Models trained using scikit-learn pipelines with StandardScaler and OneHotEncoder
on a dataset of 193,876 HDB resale transactions:

| Model | R² | RMSE (SGD) |
|-------|----|------------|
| Linear Regression | 0.90 | $55,930 |
| Random Forest (400 estimators) | 0.97 | $28,033 |

The Random Forest model significantly outperformed Linear Regression, achieving
an R² of 0.97 and reducing prediction error by ~50% (RMSE of $28,033 vs $55,930).

Feature importance analysis identified **accessibility** (MRT proximity, CBD distance)
and **flat characteristics** (floor area, storey level) as the strongest predictors
of resale price.

### 5. Post-ML EDA
- Top 10 feature importance visualisation
- Feature importance by category (Accessibility, Amenities, Flat Characteristics,
  Lease Attributes, Temporal Factors)
- Cumulative importance curve
- Correlation heatmap between top features and resale price

## Tech Stack
- Python (pandas, NumPy, scikit-learn, seaborn, matplotlib)
- OneMap API (geospatial geocoding)
- Git (version control)

## Repository Structure
```
├── DAA_Group_2_Internal_Project.ipynb   # Full pipeline notebook
├── HDB_Resale_Prices.xlsx               # Raw dataset
├── HDB_Resale_Prices_Data_Engineered.csv
├── HDB_Resale_Prices_Features_Engineered.csv
├── HDB_Resale_Prices_Features_Importances.csv
└── README.md
```

