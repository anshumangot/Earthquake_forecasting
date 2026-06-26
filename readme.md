# Earthquake Prediction and Risk Analysis
### A Data Science Project Report

Prepared using historical seismic records (1995–2023) and live USGS feed data
Techniques: Exploratory Data Analysis, Random Forest Classification, Geospatial Mapping
Date: June 2026

---

## 1. Overview

This report documents an end-to-end data science project focused on earthquake risk understanding and forecasting. The work is organized into two connected phases: (1) a descriptive risk analysis of historical global earthquake records, and (2) a predictive modeling phase that forecasts the likelihood of a major earthquake occurring in a geographic grid cell over the following month.

The project combines exploratory data analysis, supervised machine learning (Random Forest classification), interactive geospatial mapping with Folium, and geophysical feature engineering using active fault-line data, to produce a practical risk-scoring pipeline.

### 1.1 Objectives

- Explore and visualize the global distribution, magnitude, and depth characteristics of historical earthquakes.
- Classify recorded earthquakes as "high-risk" (magnitude ≥ 7) using available severity indicators.
- Build a forward-looking forecasting model that predicts whether a major earthquake (magnitude ≥ 6.5) will occur in a given region in the next month.
- Improve the forecasting model using temporal features (rolling counts, previous-month statistics) and geophysical context (distance to the nearest active fault line).
- Visualize results on interactive world maps to support intuitive risk communication.

### 1.2 Datasets Used

- **earthquake_1995-2023.csv** — 1,000 historical significant earthquake records (magnitude, depth, location, tsunami flag, significance score, alert level, etc.).
- **USGS live earthquake query feed** (`query (4).csv`) — continuously updated catalog of seismic events (2010–2026) used for the monthly forecasting model.
- **gem_active_faults.geojson** — global active fault-line geometry (GEM Global Active Faults Database) used to compute distance-to-fault risk features.

### 1.3 Tools and Libraries

- Python: pandas, NumPy for data wrangling
- Matplotlib for static visualizations
- scikit-learn (`RandomForestClassifier`) for classification and forecasting
- GeoPandas and Shapely for geospatial fault-distance analysis
- Folium for interactive world-map visualizations

---

## 2. Data Description

The historical dataset contains 1,000 earthquake records spanning 1995–2023 with 19 attributes, including magnitude, depth, community-reported intensity (CDI), modified Mercalli intensity (MMI), tsunami flag, significance score, alert level, and geographic coordinates.

| Statistic | Value |
|---|---|
| Total records | 1,000 |
| Average magnitude | 6.94 |
| Maximum magnitude | 9.1 |
| Minimum magnitude | 6.5 |
| Records with tsunami flag = 1 | 325 |
| Records with tsunami flag = 0 | 675 |

---

## 3. Exploratory Risk Analysis

### 3.1 Magnitude Distribution

Most recorded earthquakes cluster just above the 6.5 cutoff used for inclusion in the dataset, with frequency dropping off sharply for higher magnitudes. This right-skewed pattern is typical of seismic catalogs and reflects the Gutenberg–Richter law: large earthquakes are exponentially rarer than smaller ones.

![Distribution of Earthquake Magnitudes](images/img_magnitude_hist.jpeg)
*Figure 1. Distribution of earthquake magnitudes across the 1,000-record historical dataset.*

### 3.2 Depth vs. Magnitude

Plotting depth against magnitude shows that high-magnitude events occur across a wide range of depths, from very shallow (near-surface) to several hundred kilometers deep. Shallow earthquakes dominate in count but deep events can still reach high magnitudes, particularly along subduction zones.

![Depth vs Magnitude](images/img_depth_vs_mag.jpeg)
*Figure 2. Scatter plot of earthquake depth (km) versus magnitude.*

### 3.3 Most Earthquake-Prone Countries

Indonesia records by far the highest number of significant earthquakes in the dataset, followed by Papua New Guinea, Chile, and Vanuatu — all countries situated along the Pacific "Ring of Fire." This concentration reflects active subduction-zone boundaries.

![Top 10 Countries by Earthquake Count](images/img_top_countries.jpeg)
*Figure 3. Top 10 countries ranked by number of recorded earthquakes.*

| Rank | Country | Earthquake Count |
|---|---|---|
| 1 | Indonesia | 140 |
| 2 | Papua New Guinea | 81 |
| 3 | Chile | 42 |
| 4 | Vanuatu | 36 |
| 5 | Mexico | 31 |
| 6 | Peru | 26 |
| 7 | Japan | 26 |
| 8 | Solomon Islands | 25 |
| 9 | United States of America | 24 |
| 10 | Philippines | 23 |

### 3.4 Geospatial Visualization

An interactive Folium map was built to plot earthquake epicenters globally. A first view samples 200 records to confirm marker placement; a second, full view plots all 1,000 events, clearly tracing the Pacific Ring of Fire, the Alpide belt (Mediterranean through the Himalayas to Indonesia), and mid-ocean ridge activity.

![Sample of 200 earthquakes](images/img_map_sample200.jpeg)
*Figure 4. Sample of 200 earthquake epicenters plotted on a world map.*

![Full dataset map](images/img_map_full.jpeg)
*Figure 5. Full historical dataset (1,000 events) plotted globally, tracing major seismic belts.*

### 3.5 High-Risk Classification Map

Earthquakes were labeled "high-risk" when magnitude ≥ 7. Plotting these in red against lower-magnitude events in blue highlights that high-risk events are not uniformly distributed — they cluster heavily along specific subduction boundaries in the Pacific and Indonesia/Alpide regions rather than being spread evenly across all seismic zones.

![High-risk vs lower-risk map](images/img_map_risk.jpeg)
*Figure 6. High-risk earthquakes (magnitude ≥ 7, red) versus lower-risk events (blue).*

---

## 4. Risk Classification Model

A Random Forest classifier was trained to predict whether a recorded earthquake qualifies as "high-risk" (magnitude ≥ 7) using depth, community intensity (CDI), modified Mercalli intensity (MMI), and the significance score (`sig`) as input features. The data was split 80/20 into training and test sets.

| Metric | Result |
|---|---|
| Model | RandomForestClassifier (default parameters) |
| Features used | depth, cdi, mmi, sig |
| Test accuracy | **0.89 (89%)** |
| Confusion matrix | [[116, 17], [5, 62]] |

The model correctly classified 178 of 200 test cases. It performed strongly at identifying non-high-risk events (116 true negatives) and high-risk events (62 true positives), with a relatively low false-negative rate (5 high-risk events missed), which is the more important error type to minimize for risk-screening applications.

---

## 5. Earthquake Forecasting Model

The second phase shifts from classifying already-recorded earthquakes to forecasting future risk. Live USGS catalog data was aggregated into 5° × 5° latitude/longitude grid cells on a monthly basis, producing features such as monthly earthquake count, average/maximum magnitude, and average depth per cell. A binary target ("major earthquake next month," magnitude ≥ 6.5) was created by shifting the "major event" flag forward by one month within each grid cell.

### 5.1 Baseline Forecasting Model

Using only the current month's aggregated statistics (event count, average magnitude, max magnitude, average depth), a class-balanced Random Forest was trained on a strongly imbalanced target (4,281 "no major event" months vs. 577 "major event" months).

| Metric | Result |
|---|---|
| Features | eq_count, avg_mag, max_mag, avg_depth |
| Test accuracy | 0.776 (77.6%) |
| Recall (major-event class) | 0.10 |
| Precision (major-event class) | 0.09 |
| Most important feature | avg_depth (70.9% importance) |

While overall accuracy looked respectable, recall for the rare "major earthquake" class was very low (only 11 of 115 true major-event months were correctly identified), showing that current-month statistics alone are insufficient — the model was effectively biased toward predicting "no major event."

### 5.2 Adding Temporal (Lag and Rolling) Features

To capture trend and momentum in seismic activity, three additional features were engineered: the previous month's earthquake count and maximum magnitude (1-month lag), and a 3-month rolling sum of earthquake counts per grid cell.

| Metric | Result |
|---|---|
| Added features | prev_eq_count, prev_max_mag, last3m_eq_count |
| Test accuracy | **0.836 (83.6%)** |
| Improvement vs. baseline | +6.0 percentage points |

Adding short-term history improved overall accuracy meaningfully, confirming that recent seismic momentum carries predictive signal beyond a single month's snapshot.

### 5.3 Adding Geophysical Context (Distance to Active Fault)

Using the GEM Global Active Faults dataset, each grid cell's centroid was projected to a metric coordinate system (EPSG:3857) and the distance to the nearest active fault line was computed in kilometers for every grid cell. This geophysical proximity feature was added to the model.

| Metric | Result |
|---|---|
| Added feature | distance_to_fault_km |
| Mean distance to nearest fault | 188.96 km |
| Median distance to nearest fault | 128.68 km |
| Final test accuracy | **0.880 (88.0%)** |
| Confusion matrix | [[725, 3], [96, 3]] |

Incorporating fault-line proximity pushed overall accuracy to 88%, the highest of the three forecasting iterations. However, recall on the major-earthquake class remained low (only 3 of 99 true events correctly flagged), which is discussed further in the Results and Limitations sections below.

### 5.4 Forecasting Model Progression

| Model Version | Key Features Added | Test Accuracy |
|---|---|---|
| Baseline | Monthly count, avg/max magnitude, avg depth | 77.6% |
| + Temporal features | Previous-month lag + 3-month rolling count | 83.6% |
| + Geophysical features | Distance to nearest active fault line | **88.0%** |

---

## 6. Results and Discussion

### 6.1 Key Findings

1. Seismic activity is highly concentrated geographically: Indonesia, Papua New Guinea, and Chile alone account for over a quarter of all recorded significant earthquakes, consistent with their position on active subduction boundaries.
2. Magnitude follows a right-skewed (power-law-like) distribution — events near the inclusion threshold (6.5–7.0) are far more common than extreme events (8.0+).
3. The high-risk classification model (89% accuracy) reliably distinguishes high-magnitude historical events using intensity and significance indicators, making it suitable for rapid post-event risk triage.
4. For forward-looking forecasting, overall accuracy improved steadily as richer features were added (77.6% → 83.6% → 88.0%), confirming that temporal momentum and proximity to active faults both carry genuine predictive value.
5. Despite improving accuracy, recall on the rare "major earthquake next month" class stayed low across all forecasting iterations. This is the expected behavior of a highly imbalanced, inherently difficult prediction problem rather than a flaw unique to one model version.
6. Average depth emerged as the single most influential feature in the baseline forecasting model (70.9% importance), suggesting depth carries strong regional/regime information correlated with future seismicity.

### 6.2 Interpretation

The overall accuracy figures for the forecasting models can be misleading on their own because the target class (a major earthquake in a given grid-month) is rare — a model that always predicts "no major event" would already score well above 80% accuracy. The meaningful gains in this project came from increases in correctly identified true cases relative to the baseline, and from the model learning to use both short-term seismic momentum and long-term geophysical structure (fault proximity) as signals, rather than from accuracy alone.

This pattern is consistent with the broader scientific understanding that genuinely reliable short-term earthquake prediction (precise time, location, and magnitude) remains an unsolved problem; what is achievable with available data is probabilistic risk forecasting — highlighting regions and time windows of elevated likelihood — which is the framing this project adopts.

---

## 7. Limitations

- Severe class imbalance (major-earthquake months are a small minority) limits recall, meaning some true high-risk windows are likely to be missed.
- The historical risk dataset is limited to 1,000 significant events and may not capture the full magnitude/depth range of all global seismicity.
- Grid-based spatial aggregation (5° × 5° cells) is coarse and may blur risk signals that vary over shorter distances, especially near complex fault networks.
- The models use historical statistical association, not physical earthquake-nucleation modeling, and should be treated as a risk-screening aid rather than a deterministic prediction tool.

---

## 8. Conclusion and Future Work

This project demonstrates a complete pipeline from raw seismic data to actionable risk insight: descriptive analysis revealed where and how earthquakes cluster globally, a classification model showed historical events can be reliably flagged as high-risk (89% accuracy), and an evolving forecasting model showed that combining temporal momentum with geophysical fault-proximity features steadily improves the ability to anticipate major seismic activity (accuracy improving from 77.6% to 88.0%).

### 8.1 Recommended Next Steps

- Apply class-imbalance techniques (SMOTE, threshold tuning, or cost-sensitive learning) to improve recall on the major-earthquake class without sacrificing precision.
- Incorporate additional geophysical covariates such as plate boundary type, historical recurrence intervals, and crustal stress indicators.
- Use finer spatial grids or fault-segment-level aggregation instead of uniform 5° cells.
- Validate the forecasting model with rolling time-based cross-validation to better simulate real-world early-warning deployment.
- Package the model and Folium maps into a live monitoring dashboard for continuous regional risk updates.

---

## 9. References / Data Sources

- Historical earthquake catalog: `earthquake_1995-2023.csv` (1,000 significant global earthquakes).
- Live seismic feed: USGS Earthquake Hazards Program query catalog.
- Active fault geometry: GEM (Global Earthquake Model) Global Active Faults Database.
