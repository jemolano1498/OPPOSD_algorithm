import numpy as np


class MathModel:
    def __init__(self, logaritmic, z, grad=0):
        self.z = z
        self.logaritmic = logaritmic
        self.rand_comp = 0
        self.grad = 1
        if self.logaritmic:
            self.rand_comp = np.random.uniform(0.2, 1.5)
            if grad:
                self.grad = -1
        else:
            self.rand_comp = np.random.uniform(-5, 5)
            if grad:
                self.grad = 1 if np.random.random() < 0.5 else -1

    def calculate_x(self, x):
        if self.logaritmic:
            return self.grad * self.z[0] * np.log(self.rand_comp * x) + self.z[1]
        else:
            return (self.grad * (self.z[0] + (self.rand_comp * 1e-5))) * x + self.z[1]

    def inverse_value(self, y):
        if self.logaritmic:
            return np.exp(self.grad * (y - self.z[1]) / self.z[0]) / self.rand_comp
        else:
            return 0

    def adjust_intercept(self, new_val):
        self.z[1] = new_val


class PaceSimulator:
    def __init__(self):
        self.time_step = 0
        self.last_value = None
        self.current_model = None
        self.models = [None] * 4

    def calculate_model(self, pacing, percentage, grad=0):
        input_model = int(str(pacing) + str(1 if percentage > 0 else 0), 2)
        if input_model == 3:  # 1-1
            z = [0.00261429, 0.98637389]
            model = MathModel(1, z, grad)

        elif input_model == 2:  # 1-0
            z = [-0.00899492, 1.03965412]
            model = MathModel(1, z, grad)

        elif input_model == 1:  # 0-1
            z = [-3.17034405e-05, 9.86148032e-01]
            model = MathModel(0, z)

        else:  # 0-0
            z = [-3.19261596e-06, 1.01227226e00]
            model = MathModel(0, z, 0)

        return model

    def predict(self, pacing, pref_pace, target_pace):
        percentage = abs(pref_pace - target_pace) / pref_pace
        prediction_noise = np.random.uniform(0, 5e-3)
        if percentage > 0:
            pace_noise = np.random.uniform(target_pace - 4, target_pace+3)
        else:
            pace_noise = np.random.uniform(pref_pace - 5, pref_pace + 5)

        input_model = int(str(pacing) + str(1 if percentage > 0 else 0), 2)
        if self.time_step == 0:
            self.last_value = np.random.uniform(0.9, 1.08)
            self.time_step = self.time_step + 1
            return (self.last_value + prediction_noise) * pace_noise

        if input_model != self.current_model:
            if input_model == 1:  # 0-1
                self.current_model = 1
                self.models[self.current_model] = self.calculate_model(0, 0.1)
                self.models[self.current_model].adjust_intercept(self.last_value)
                self.time_step = 1

            elif input_model == 2:  # 1-0
                self.current_model = 2
                if self.last_value > 1:
                    self.models[self.current_model] = self.calculate_model(1, 0)
                else:
                    self.models[self.current_model] = self.calculate_model(1, 1)
                self.time_step = self.models[self.current_model].inverse_value(
                    self.last_value
                )

            elif input_model == 3:  # 1-1
                self.current_model = 3
                if self.last_value > 1:
                    self.models[self.current_model] = self.calculate_model(1, 0)
                else:
                    self.models[self.current_model] = self.calculate_model(1, 1)
                self.time_step = self.models[self.current_model].inverse_value(
                    self.last_value
                )

            else:  # 0-0
                self.current_model = 0
                self.models[self.current_model] = self.calculate_model(0, 0)
                self.models[self.current_model].adjust_intercept(self.last_value)
                self.time_step = 1

        self.time_step = self.time_step + 1
        self.last_value = self.models[self.current_model].calculate_x(self.time_step)

        return (self.last_value + prediction_noise) * pace_noise
