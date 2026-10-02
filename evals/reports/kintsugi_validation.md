# Kintsugi DAM 3.1 on its own released validation and test scores

Validation n = 8113, test n = 7034. Label: PHQ-9 >= 10. Source: KintsugiHealth/dam-dataset. Scores only, no audio.

## 1. Their threshold on test

AUC 0.760; at -0.6699: sensitivity 0.581, specificity 0.778, prevalence 0.39.

## 2. Allowing "can't tell" (band chosen on validation, 20% budget)

Band -1.033 to -0.654. On test: sensitivity 0.714, specificity 0.738, undecided 19%.

## 3. By group (their threshold, test set)

| group | value | n | PHQ>=10 | sensitivity | specificity | AUC |
|---|---|---|---|---|---|---|
| gender | Female | 4328 | 0.43 | 0.624 | 0.719 | 0.745 |
| gender | Male | 2596 | 0.31 | 0.451 | 0.865 | 0.761 |
| ethnicity | Asian / Pacific Islander | 400 | 0.37 | 0.483 | 0.802 | 0.744 |
| ethnicity | Black or African American | 275 | 0.43 | 0.647 | 0.737 | 0.759 |
| ethnicity | Black or African American only | 594 | 0.35 | 0.406 | 0.783 | 0.688 |
| ethnicity | Hispanic or Latino | 191 | 0.55 | 0.657 | 0.686 | 0.752 |
| ethnicity | Multi-racial | 400 | 0.41 | 0.663 | 0.667 | 0.726 |
| ethnicity | Other or mixed race | 161 | 0.50 | 0.654 | 0.738 | 0.767 |
| ethnicity | Some other race only | 228 | 0.50 | 0.617 | 0.673 | 0.700 |
| ethnicity | White | 1638 | 0.46 | 0.612 | 0.828 | 0.790 |
| ethnicity | White or Caucasian only | 2918 | 0.31 | 0.568 | 0.782 | 0.755 |
| age_band | 18-29 | 1451 | 0.51 | 0.722 | 0.58 | 0.715 |
| age_band | 30-44 | 2242 | 0.47 | 0.642 | 0.628 | 0.691 |
| age_band | 45-59 | 1502 | 0.41 | 0.521 | 0.808 | 0.740 |
| age_band | 60+ | 1839 | 0.17 | 0.155 | 0.968 | 0.719 |
| income | $100,000-$150,000 | 375 | 0.23 | 0.341 | 0.845 | 0.739 |
| income | $40,000-$79,000 | 1453 | 0.36 | 0.57 | 0.777 | 0.745 |
| income | $80,000-$99,000 | 402 | 0.21 | 0.547 | 0.778 | 0.748 |
| income | Less than $40,000 | 1861 | 0.39 | 0.574 | 0.728 | 0.728 |
| income | More than $150,000 | 103 | 0.23 | 0.417 | 0.899 | 0.818 |
| english_preferred | Yes | 4209 | 0.34 | 0.556 | 0.768 | 0.744 |

Groups at least 10 points from the overall sensitivity or specificity:

- gender = Male: sensitivity 0.451 vs 0.581 overall
- ethnicity = Black or African American only: sensitivity 0.406 vs 0.581 overall
- ethnicity = Multi-racial: specificity 0.667 vs 0.778 overall
- ethnicity = Some other race only: specificity 0.673 vs 0.778 overall
- age_band = 18-29: sensitivity 0.722 vs 0.581 overall
- age_band = 18-29: specificity 0.58 vs 0.778 overall
- age_band = 30-44: specificity 0.628 vs 0.778 overall
- age_band = 60+: sensitivity 0.155 vs 0.581 overall
- age_band = 60+: specificity 0.968 vs 0.778 overall
- income = $100,000-$150,000: sensitivity 0.341 vs 0.581 overall
- income = More than $150,000: sensitivity 0.417 vs 0.581 overall
- income = More than $150,000: specificity 0.899 vs 0.778 overall

## 4. Anhedonia or low mood? (exploratory)

Spearman with PHQ item 1 (interest/pleasure) 0.405; with item 2 (feeling down) 0.456.
Partial, controlling the other item: item 1 0.125; item 2 0.258. n = 7034.

Exploratory, not pre-registered; one model on its developer's data.

## 5. Recording quality (their noise estimate, tertiles)

- noisiest third: AUC 0.728 (n 2345)
- middle: AUC 0.763 (n 2344)
- cleanest third: AUC 0.785 (n 2345)
