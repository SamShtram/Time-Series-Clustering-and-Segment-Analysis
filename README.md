# Time-Series-Clustering-and-Segment-Analysis

This project performs unsupervised clustering of physiological time-series signals (ABP, ECG, PPG) from PulseDB / VitalDB using divide-and-conquer algorithms and classical algorithmic reasoning.
It identifies clusters of similar signal segments, finds the most cohesive (closest) pairs within each cluster, and uses Kadane’s algorithm to highlight the most active or anomalous time intervals within each signal.

This approach demonstrates that purely algorithmic techniques — without machine learning — can still extract meaningful physiological insights from biomedical time-series data.

#IN ORDER TO RUN THIS PROJECT

MAKE SURE TO HAVE A RUNNING VERSION OF PYTHON AND GIT ON YOUR LOCAL MACHINE

git clone https://github.com/YourUsername/Time-Series-Clustering-and-Segment-Analysis.git

cd Time-Series-Clustering-and-Segment-Analysis

python -m venv venv

source venv/bin/activate        # Mac/Linux

venv\Scripts\activate           # Windows

pip install -r requirements.txt

python src/main.py

this will run the entire code, it will take upwards of 5 minutes to finsih executing, all results will output into the results folder

