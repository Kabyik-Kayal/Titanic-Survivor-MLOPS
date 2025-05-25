import os
import sys
import pickle
from src.logger import get_logger
from src.custom_exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.feature_store import FeatureStore

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, feature_store:FeatureStore, model_save_path="artifacts/models/"):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.model = None

        os.makedirs(self.model_save_path, exist_ok=True)
        logger.info(f"Model training initialized...")

    def load_data_from_redis(self, entity_ids):
        try:
            logger.info("Extracting data from Redis...")

            data = []
            for entity_id in entity_ids:
                features = self.feature_store.get_features(entity_id)
                if features is not None:
                    data.append(features)
                else:
                    logger.warning(f"No features found for entity_id: {entity_id}")
            return data
        except Exception as e:
            logger.error(f"Error while loading data from Redis: {e}")
            raise CustomException(str(e), sys)

    def prepare_data(self):
        try:
            entity_ids = self.feature_store.get_all_entity_ids()
            train_entity_ids, test_entity_ids = train_test_split(entity_ids, test_size=0.2, random_state=42)

            train_data = self.load_data_from_redis(train_entity_ids)
            test_data = self.load_data_from_redis(test_entity_ids)

            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)

            X_train = train_df.drop(columns=['Survived'])
            logger.info(f"X_train cols: {X_train.columns}")
            y_train = train_df['Survived']
            X_test = test_df.drop(columns=['Survived'])
            y_test = test_df['Survived']
            logger.info(f"Data prepared successfully with {len(X_train)} training samples and {len(X_test)} test samples.")
            
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logger.error(f"Error while preparing data: {e}")
            raise CustomException(str(e), sys)
        
    def hyperparameter_tuning(self, X_train, y_train):
        param_distributions={
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 20, 30],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    }
        rf = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(estimator=rf, 
                                         param_distributions=param_distributions,
                                         n_iter=10,
                                         cv=3,
                                         scoring='accuracy',
                                         verbose=2,
                                         random_state=42,
                                         n_jobs=-1)
        random_search.fit(X_train, y_train)
        logger.info(f"Best parameters found: {random_search.best_params_}")
        return random_search.best_estimator_
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        try:
            best_rf = self.hyperparameter_tuning(X_train, y_train)
            y_pred = best_rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model trained successfully with accuracy: {accuracy:.3f}")
            self.save_model(best_rf)
        
        except Exception as e:
            logger.error(f"Error during model training and evaluation: {e}")
            raise CustomException(str(e), sys)
    
    def save_model(self, model):
        try:
            model_filename = f"{self.model_save_path}/random_forest_model.pkl"
            with open(model_filename, 'wb') as file:
                pickle.dump(model, file)
            logger.info(f"Model saved at {model_filename}")
        
        except Exception as e:
            logger.error(f"Error while saving model: {e}")
            raise CustomException(str(e), sys)
        
    def run(self):
        try:
            logger.info("Starting model training process...")
            X_train, X_test, y_train, y_test = self.prepare_data()
            self.train_and_evaluate(X_train, y_train, X_test, y_test)

            logger.info("Model training process completed successfully.")
        
        except Exception as e:
            logger.error(f"Error during model training process: {e}")
            raise CustomException(str(e), sys)
        
if __name__ == "__main__":
    feature_store = FeatureStore()
    model_trainer = ModelTraining(feature_store=feature_store)
    model_trainer.run()