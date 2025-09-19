# modules/ml_pipeline.py

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

def search_best_model(X_train, y_train, model, param_grid=None, scoring_metric="recall", n_jobs=-1, use_pca=True):
    """
    GridSearchCV kết hợp PCA (tùy chọn) và model bất kỳ.
    
    Args:
        X_train: DataFrame hoặc array, dữ liệu train
        y_train: Series hoặc array, nhãn train
        model: sklearn estimator, ví dụ RandomForestClassifier()
        param_grid: dict, tham số cho GridSearchCV
        scoring_metric: str, metric để tối ưu
        n_jobs: int, số CPU sử dụng
        use_pca: bool, có thêm PCA vào pipeline hay không
        
    Returns:
        dict: best_params và best_score
    """
    
    steps = []
    if use_pca:
        steps.append(('pca', PCA()))
    steps.append(('clf', model))
    
    pipeline = Pipeline(steps)
    
    if param_grid is None:
        param_grid = {}  # Trường hợp không muốn search tham số
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring=scoring_metric,
        n_jobs=n_jobs
    )
    
    grid_search.fit(X_train, y_train)
    
    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_
    }
