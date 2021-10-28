from src.utils.models import load_full_model

if __name__ == '__main__':
    path_name="artifacts/base_model/updated_VGG16_base_model.h5"
    print("model loading is being started")
    model=load_full_model(path_name)
    print("model loaded successfully")
    model.fit()
