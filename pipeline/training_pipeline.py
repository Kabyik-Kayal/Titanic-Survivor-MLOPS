from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor
from src.model_training import ModelTraining
from src.feature_store import FeatureStore
from config.paths_config import *
from config.database_config import DB_CONFIG


if __name__=="__main__":

    data_ingestion = DataIngestion(DB_CONFIG , RAW_DIR)
    data_ingestion.run()
    feature_store = FeatureStore()
    data_processor = DataProcessor(TRAIN_PATH,TEST_PATH,feature_store)
    data_processor.run()
    model_trainer = ModelTraining(feature_store)
    model_trainer.run()