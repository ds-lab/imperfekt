class VITALS:
    PARAMS = ["heartrate", "resprate", "o2sat", "sbp"]

    # Based on National Early Warning Score (Score = 0)
    NORMAL_RANGES_MAR_MNAR_TEST = {
        "heartrate": (51, 90),
        "resprate": (12, 20),
        "sbp": (111, 219),
        "o2sat": (96, 100),
    }

    IMPUTATION_VALUE_MAR_MNAR_TEST = "mean"  # options: zero|mean|median|ffill_within_id
    STANDARDIZE_MAR_MNAR_TEST = True
