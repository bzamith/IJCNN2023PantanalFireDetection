"""Module which contains the ClassificationAlgorithmEnum enum class"""

from src.enum.enum_class import EnumClass


class ClassificationAlgorithmEnum(EnumClass):
    """Enum for different classification algorithms"""

    MLP = "Multi Layer Perceptron"
    SVM = "Support Vector Machine"
    KNN = "K-Nearest Neighbours"
    DTREE = "Decision Tree"
    RFOREST = "Random Forest"
    NB = "Naive Bayes"
    LR = "Logistic Regression"
    XGBOOST = "XGBoost"
    LIGHTGBM = "LightGBM"
    CATBOOST = "CatBoost"
