"""
Part of Phase 7: train/validation/test split
------------------------------------------------
WHY A STRATIFIED CUSTOMER-LEVEL SPLIT (not a further time-based split,
unlike Day 2 or Day 11): the time dimension in this project was already
fully spent constructing the observation/prediction window split itself
-- there's no THIRD time period left to hold out chronologically without
either shrinking the already-modest observation window or losing the
ability to construct a target at all. A stratified split on `will_return`
(to preserve the 70.9%/29.1% return/churn balance across splits) is the
correct choice given that constraint, the same category of reasoning
Day 7 and Day 9 used when no further time axis was available to them.
"""

from sklearn.model_selection import train_test_split


def split_data(df, feature_cols, test_size=0.15, val_size=0.15, random_state=42):
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["will_return"], random_state=random_state)
    val_relative = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_relative, stratify=train_val["will_return"], random_state=random_state)

    return (
        train[feature_cols], val[feature_cols], test[feature_cols],
        train[["will_return", "future_spend"]], val[["will_return", "future_spend"]], test[["will_return", "future_spend"]],
    )
