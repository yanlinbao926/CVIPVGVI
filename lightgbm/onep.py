import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings


warnings.filterwarnings("ignore")

class LightGBMGVITrainer:
    def __init__(self, output_dir="./output"):
        """
        初始化绿视率(GVI) LightGBM 训练器 (极速静默版)
        :param output_dir: 模型和评估结果的保存目录
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        

        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)
        
 
        self.model = lgb.LGBMRegressor(
            objective="regression",
            num_leaves=31,  
            learning_rate=0.1,  
            n_estimators=500,  
            max_depth=-1,  
            reg_alpha=0.0,  
            reg_lambda=0.0,  
            min_child_samples=10,  
            subsample=0.8,  
            colsample_bytree=0.8,  
            random_state=42,
            n_jobs=4,       
            verbose=-1       
        )


        self.param_grid = {
            "num_leaves": [31, 50, 100], 
            "max_depth": [-1, 20], 
            "learning_rate": [0.1], 
            "n_estimators": [500, 700], 
            "reg_alpha": [0.0, 0.5, 1.0], 
            "reg_lambda": [0.0, 0.5, 1.0], 
            "min_child_samples": [10, 20, 30], 
            "subsample": [0.8], 
            "colsample_bytree": [0.8], 
        }

    def train_and_evaluate(self, train_data_path, test_data_path, target_col, feature_cols):
        """
        执行训练、网格搜索调参、并在测试集上进行评估
        """
        print(f"📖 正在读取训练集数据: {train_data_path}")
        data_train = pd.read_csv(train_data_path)
        print(f"📖 正在读取测试集数据: {test_data_path}")
        data_test = pd.read_csv(test_data_path)


        X_train = data_train[feature_cols]
        y_train = data_train[target_col]
        X_test = data_test[feature_cols]
        y_test = data_test[target_col]

        print(f"📊 样本数划分: 训练集 {len(X_train)} | 测试集 {len(X_test)}")
        print(f"🎯 预测目标: {target_col}")
        print(f"🧠 训练特征数量: {len(feature_cols)} 个")
        print(f"   ➤ 包含特征: {feature_cols[:5]} ... 等") 
        
        print("🚀 正在启动网格搜索 (已开启极速静默模式，请耐心等待)...")


        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.kf,
            scoring="neg_mean_squared_error",
            n_jobs=1 
        )
        

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print("✅ 网格搜索完成！最佳参数已找到。")


        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_train_metrics_path = os.path.join(self.output_dir, "lightgbm_cv_train_metrics.csv")
        cv_results.to_csv(cv_train_metrics_path, index=False)
        print(f"💾 交叉验证详细日志已保存至: {cv_train_metrics_path}")


        print("🧪 正在独立测试集上进行终极验证...")
        y_pred_test = best_model.predict(X_test)
        

        y_pred_clipped = np.clip(y_pred_test, 0.0, 1.0)
        

        mse_test = mean_squared_error(y_test, y_pred_clipped)
        mae_test = mean_absolute_error(y_test, y_pred_clipped)
        r2_test = r2_score(y_test, y_pred_clipped)
        
        if np.std(y_pred_clipped) == 0 or np.std(y_test) == 0:
            pearson_r_val = 0.0
        else:
            pearson_r_val, _ = pearsonr(y_test, y_pred_clipped)


        result_test = pd.DataFrame({
            "target": [target_col],
            "mse_test": [mse_test],
            "mae_test": [mae_test],
            "r2_score": [r2_test],
            "pearson_r": [pearson_r_val]
        })
        

        print("\n" + "="*45)
        print(f"🏆 测试集终极评估成绩 (Target: {target_col}):")
        print(f"   ➤ MSE (均方误差)  : {mse_test:.5f}")
        print(f"   ➤ MAE (平均绝对误差): {mae_test:.5f}")
        print(f"   ➤ R²  (决定系数)  : {r2_test:.5f}")
        print(f"   ➤ Pearson's r    : {pearson_r_val:.5f}")
        print("="*45 + "\n")


        test_metrics_path = os.path.join(self.output_dir, "lightgbm_test_metrics.csv")
        result_test.to_csv(test_metrics_path, index=False)
        print(f"💾 最终成绩单已保存至: {test_metrics_path}")


        feature_names = X_train.columns.tolist()
        model_save_path = os.path.join(self.output_dir, "best_lightgbm_model.pkl")
        joblib.dump(
            {"model": best_model, "feature_names": feature_names},
            model_save_path
        )
        print(f"✅ 最佳模型及特征名称已成功保存至: {model_save_path}")


if __name__ == "__main__":

    TRAIN_CSV = ""  # Training set address
    TEST_CSV = ""  # Test set address  
    

    OUTPUT_FOLDER = ""# Output address
    

    MY_TARGET = "pedestrian_gvi"

    MY_FEATURES = []
    if os.path.exists(TRAIN_CSV):
        sample_df = pd.read_csv(TRAIN_CSV, nrows=0) 
        

        MY_FEATURES = [col for col in sample_df.columns if col.endswith('_weighted')]   #"_pixel" or "_weighted"
        
        print(f"🚀 自动扫描表头完成，共提取出 {len(MY_FEATURES)} 个特征列即将参与训练！")
    else:
        print("❌ 找不到训练集文件，无法提取表头！")


    if len(MY_FEATURES) > 0 and os.path.exists(TEST_CSV):
        trainer = LightGBMGVITrainer(output_dir=OUTPUT_FOLDER)
        trainer.train_and_evaluate(
            train_data_path=TRAIN_CSV,
            test_data_path=TEST_CSV,
            target_col=MY_TARGET,
            feature_cols=MY_FEATURES
        )
    else:
        print("⚠️ 训练终止：未提取到特征列，或测试集文件不存在。")