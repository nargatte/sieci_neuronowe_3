# H1 - addition of 3 closest cities will increase accuracy with similar convergence, but with longer compute time

Regression without neighbors:
1. 4 epochs, 33.42% / 8.11%, 8.97 s
2. 41 epochs, 45.92% / 8.54%, 149.94 s

Regression with neighbors:
1. 5 epochs, 14.95% / 10.48%, 25.59 s
2. 4 epochs, 48.76% / 6.02%, 64.36 s

Classification without neighbors:
1. 10 epochs, 75.52% / 60.77 %, 0.5 / 0.5, 46.84 s
2. 11 epochs, 75.52% / 60.77 %, 0.5 / 0.5, 54.63 s

Classification with neighbors:
1. 12 epochs, 76.10% / 60.82 %, 0.5 / 0.5, 74.83 s (not better, just less cases after nan filtering)
2. 15 epochs, 76.10% / 60.82 %, 0.5 / 0.5, 238.28 s (not better, just less cases after nan filtering)


# H2 - mean from 24 predictions will be better for architecutres with little number of weights, whereas 1 prediction will be better for many weights

1 prediction:
1. 4 epochs, 40.63% / 8.11 %, 9.47 s
2. 41 eppchs, 45.92% / 8.54%, 114.28 s

24 predictions - crash (requires 31 GiB for batching)


# H3 - L2 grants faster convergence than L1

L1:
1. 26 epochs, 55.16% / 11.34%, 142.34 s
2. 6 epochs, 15.73 % / 15.23%, 55.99 s
3. 19 epochs, 44.79% / 7.32%, 47.08 s
4. 6 epochs, 42.57% / 6.37%, 31.44 s

L2:
1. 6 epochs, 52.48% / 7.49%, 31.62s
2. 5 epochs, 16.26% / 15.98%, 51.73 s
3. 4 epochs, 28.33% / 7.64%, 15.74 s
4. 8 epochs, 38.05% / 8.22%, 35.63 s

# H4 - cross-entropy and hinge will not differ in terms of convergence

Cross-entropy:
1. 14 epochs, 75.52% / 60.77%, 0.5 / 0.5, 93.73 s
2. 12 epochs, 75.52% / 60.77%, 0.5 / 0.5, 135.40 s
3. 13 epochs, 75.52% / 60.77%, 0.5 / 0.5, 55.08 s
4. 18 epochs, 75.52% / 60.77%, 0.7008 / 0.6983, 111.87 s

Hinge:
1. 5 epochs, 75.52% / 60.77%, 0.5 / 0.5, 35.18 s
2. 5 epochs, 75.52% / 60.77%, 0.5 / 0.5, 55.07 s
3. 6 epochs, 75.52% / 60.77%, 0.5 / 0.5, 28.19 s
4. 6 epochs, 75.52% / 60.77%, 0.5 / 0.5, 39.44 s


# H5 - the more shared weights in network, the smaller difference accuracy on train and test sets

Regression:
1. 27 epochs, 46.57% / 44.87%, 146.79 s (no sharing)
2. 12 epochs, 45.38% / 39.98%, 71.79 s (days sharing)
3. 69 epochs, 41.64% / 39.86%, 1389.94 s (days and cities sharing)

Classification:
1. 5 epochs, 75.52% / 60.77%, 0.5 / 0.5, 39.89 s (no sharing)
2. 10 epochs, 75.52% / 60.77%, 0.5 / 0.5, 79.93 s (days sharing)
3. 11 epochs, 75.52% / 60.77%, 0.5 / 0.5, 258.30 s (days and cities sharing)


# H6 - classification will have better accuracy using ReLUs, while regression will have better accuracy using sigmoids

Regression with ReLU: 6 epochs, 52.48% / 7.49%, 27.76 s
Regression with sigmoid: 5 epochs, 16.26% / 15.98%, 48.25 s
Classification with ReLU: 5 epochs, 75.52% / 60.77%, 0.5 / 0/5, 34.90 s
Classification with sigmoid: 5 epochs, 75.52% / 60.77%, 0.5 / 0.5, 55.95 s


# H7 - aggregation will grant similar results with less computation needed

Regression:
1. 6 epochs, 52.48% / 7.49%, 33.68 s (without aggregation)
2. 11 epochs, 50.98% / 7.49%, 26.54 s(with aggregation)
3. 4 epochs, 45.79% / 7.98%, 6.55 s (with aggregation)

Classification:
1. 5 epochs, 75.52%, 60.77%, 0.5 / 0.5, 32.09 s (without aggregation)
2. 16 epochs, 81.93% / 74.40%, 0.5 / 0.5, 63.68 s (with aggregation) (probably not better, just more non-windy days when aggregated)
3. 16 epochs, 81.93 / 74.40%, 0.5 / 0.5, 43.85 s (with aggregation)


# H8 - predicting middle day will grant better accuracy, but worse convergence

Regression without middle day prediction: 4 epochs, 33.42% / 8.11%, 8.22 s
Regression with middle day prediction: 4 epochs 48.05% / 6.82%, 19.06 s
Classification without middle day prediction: 10 epochs, 75.52% / 60.77%, 0.5 / 0.5, 37.64 s
Classification with middle day prediction: 16 epochs, 75.52% / 60.77%, 0.5 / 0.5, 116.20 s

