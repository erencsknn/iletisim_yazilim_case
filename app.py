from main import Main

class App():
    def __init__(self):
        self.data_path = "Makine_verileri.xlsx"
        self.main = Main(self.data_path)


    def run(self):
        self.main.change_column_type()
        self.main.change_column_type()
        self.main.encoding()
        self.main.split_date()
        self.main.create_time_window_features()
        self.main.cap_numerical_outliers_IQR()
        self.main.scaled_data()
        self.main.split_data()
        self.main.apply_pca()
        self.main.load_model()
        self.main.evaluate_model()
        
        
        

if __name__ == "__main__":
    app = App()
    app.run()