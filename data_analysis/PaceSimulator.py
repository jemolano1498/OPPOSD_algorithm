import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_folder_path = "~/Documents/THESIS/Project_Juan/"

def stack_uneven(matrix, array):

    # The resultant array has stacked on the first dimension
    result = []
    max_length = max(matrix.shape[1], len(array))
    for row in matrix:
        temp = np.append(row, np.ones(max_length - len(row)))
        # result = np.append(result, [[temp]], axis=0)
        if len(result) == 0:
            result = temp
        else:
            result = np.vstack((result, temp))
    temp = np.append(array, np.ones(max_length - len(array)))
    # result = np.append(result, [[temp]], axis=0)
    result = np.vstack((result, temp))

    return result

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
            return (self.grad * (self.z[0]+(self.rand_comp*1e-5)))*x + self.z[1]

    def inverse_value(self, y):
        if self.logaritmic:
            return np.exp(self.grad *(y-self.z[1])/self.z[0])/self.rand_comp
        else:
            return 0

    def adjust_intercept(self, new_val):
        self.z[1] = new_val

class Simulator:
    def __init__(self):
        self.pacing_data = self.get_data()
        self.time_step = 0
        self.last_value = None
        self.current_model = None
        self.models = [None] * 4
        # self.models[0] = self.calculate_model(0, 0)
        # self.models[1] = self.calculate_model(0, 0.1)
        # self.models[2] = self.calculate_model(1, 0)
        # self.models[3] = self.calculate_model(1, 0.1)

    def calculate_model(self, pacing, percentage, grad=0):

        data = self.get_average_data(pacing, percentage)

        if (pacing & (percentage > 0)): # 1-1
            N = len(data)
            x = np.linspace(1, N+1, N)
            z = np.polyfit(np.log(x), data, 1)
            print(z)
            model = MathModel(1, z, grad)

        elif (pacing & (percentage == 0.0)): # 1-0
            N = len(data)
            x = np.linspace(1, N+1, N)
            z = np.polyfit(np.log(x), data, 1)
            print(z)
            model = MathModel(1, z, grad)

        elif (not(pacing) & (percentage>0)): # 0-1
            N = len(data)
            x = np.linspace(0, N, N)
            z = np.polyfit(x, data, 1)
            print(z)
            model = MathModel(0, z)

        else: # 0-0
            N = len(data)
            x = np.linspace(0, N, N)
            z = np.polyfit(x, data, 1)
            print(z)
            model = MathModel(0, z, 0)

        return model

    def get_average_data (self, pacing, percentage):
        test_pacing_data = self.pacing_data.copy()
        accumulated_pace = []
        minimum_size = np.inf
        for participant in range(1, 16, 1):
            current_array = test_pacing_data[(test_pacing_data['runner_id']==participant) &
                  (test_pacing_data['pacing']==pacing) &
                  (test_pacing_data['preferred_percentage']==percentage)]['step_frequency']
            if len(current_array) < minimum_size:
                minimum_size = len(current_array)
            if len(accumulated_pace) == 0:
                accumulated_pace = np.expand_dims(current_array.to_numpy(), axis=0)
            else:
                accumulated_pace = stack_uneven(accumulated_pace, current_array.to_numpy())
        return accumulated_pace[:, :minimum_size].mean(axis=0)

    def get_data(self):
        tests = ['PNPfast', 'PNPpref']
        pace_perc = [0.1, 0.0]
        pacing_data = pd.DataFrame(data = None,
                                   columns=['step_frequency', 'runner_id', 'preferred_pace', 'pacing', 'time', 'preferred_percentage'],
                                   dtype='float64')
        for participant in range(1, 16, 1):
            participant_number = str(participant)

            for i in range(len(tests)):
                calculated_values = pd.read_csv(data_folder_path + ('calculated_variables/%s_R_%s_calculated.csv')%(tests[i], participant_number))

                pace_norm = calculated_values[calculated_values['pacing_frequency'].notna()]['pacing_frequency'].mean()
                calculated_values_norm = calculated_values.copy()
                calculated_values_norm['step_frequency'] = calculated_values_norm['step_frequency']/pace_norm

                m_a = calculated_values_norm['step_frequency'].ewm(alpha=0.05).mean()
                m_a_df = m_a.to_frame()
                m_a_df['runner_id']= participant
                m_a_df['preferred_pace'] = pace_norm
                m_a_df['pacing'] = np.where(calculated_values_norm['pacing_frequency'].notna(), 1, 0)
                m_a_df['time'] = calculated_values_norm['interval_footstrikes'].cumsum()
                m_a_df['preferred_percentage'] = pace_perc[i]
                pacing_data = pd.concat([pacing_data, m_a_df])
        return pacing_data

    def predict(self, pacing, pref_pace, target_pace):
        percentage = abs(pref_pace-target_pace)/pref_pace

        input_model = int(str(pacing) + str(1 if percentage > 0 else 0), 2)
        if self.time_step == 0:
            self.last_value = np.random.uniform(0.97, 1.03)
            self.time_step = self.time_step + 1
            return (self.last_value + np.random.uniform(0, 5e-3)) * np.random.uniform(pref_pace-3, pref_pace+2)

        if (input_model != self.current_model):
            if input_model == 1: # 0-1
                self.current_model = 1
                self.models[self.current_model] = self.calculate_model(0, 0.1)
                self.models[self.current_model].adjust_intercept(self.last_value)
                self.time_step = 1

            elif input_model == 2: # 1-0
                self.current_model = 2
                if self.last_value > 1:
                    self.models[self.current_model] = self.calculate_model(1, 0)
                else:
                    self.models[self.current_model] = self.calculate_model(1, 0, 1)
                self.time_step = self.models[self.current_model].inverse_value(self.last_value)

            elif input_model== 3: # 1-1
                self.current_model = 3
                if self.last_value > 1:
                    self.models[self.current_model] = self.calculate_model(1, 0.1)
                else:
                    self.models[self.current_model] = self.calculate_model(1, 0.1, 1)
                self.time_step = self.models[self.current_model].inverse_value(self.last_value)

            else: # 0-0
                self.current_model = 0
                self.models[self.current_model] = self.calculate_model(0, 0)
                self.models[self.current_model].adjust_intercept(self.last_value)
                self.time_step = 1

        self.time_step = self.time_step + 1
        self.last_value = self.models[self.current_model].calculate_x(self.time_step)

        return (self.last_value + np.random.uniform(0, 5e-3)) * np.random.uniform(pref_pace-1, pref_pace+1)

simulator_ = Simulator()

simulator_.calculate_model(0, 0) # [-3.19261596e-06, 1.01227226e+00]
simulator_.calculate_model(0, 0.1) # [-3.17034405e-05, 9.86148032e-01]
simulator_.calculate_model(1, 0) # [-0.00899492, 1.04965412]
simulator_.calculate_model(1, 0.1) # [0.00261429, 0.97637389]