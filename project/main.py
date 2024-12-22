from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
from model import scaler

app = Flask(__name__)
model = tf.keras.models.load_model('car_model.keras')

@app.route('/', methods=['GET'])
def start_page():
    # Получаем значение предсказания из параметров запроса
    prediction_text = request.args.get("prediction_text", None)
    return render_template(
        'page2var3.html', 
        prediction_text=f"Class of wine is {prediction_text}" if prediction_text else ""
    )

@app.route('/main')
def main_pg():
    return render_template('index3.html')

@app.route('/predict', methods=['POST'])
def predict():
        print('Im here')
        # Получение данных из формы
        battery_voltage = float(request.form["battery_voltage"])
        print(battery_voltage)
        battery_current = float(request.form["battery_current"])
        engine_temperature = float(request.form["engine_temperature"])
        motor_efficiency = float(request.form["motor_efficiency"])
        tire_pressure = float(request.form["tire_pressure"])
        fuel_efficiency = float(request.form["fuel_efficiency"])
        speed = float(request.form["speed"])
        acceleration = float(request.form["acceleration"])
        driving_distance = float(request.form["driving_distance"])
        ambient_temperature = float(request.form["ambient_temperature"])
        humidity = float(request.form["humidity"])
        road_condition = float(request.form["road_condition"])
        last_service_distance = float(request.form["last_service_distance"])
        service_frequency = float(request.form["service_frequency"])
        repair_cost = float(request.form["repair_cost"])
        downtime = float(request.form["downtime"])
        time_since_last_fault = float(request.form["time_since_last_fault"])

        # Подготовка данных
        data_req = [
            battery_voltage,
            battery_current,
            engine_temperature,
            motor_efficiency,
            tire_pressure,
            fuel_efficiency,
            speed,
            acceleration,
            driving_distance,
            ambient_temperature,
            humidity,
            road_condition,
            last_service_distance,
            service_frequency,
            repair_cost,
            downtime,
            time_since_last_fault,
        ]
        print(data_req)

        scaled_data = scaler.transform([data_req])
        print(f'Scaled: {scaled_data}')

        # Предсказание
        prediction = model.predict(np.array(scaled_data))
        output = np.argmax(prediction[0])  # Получаем индекс класса с наибольшей вероятностью (softmax использовали)

        # Перенаправление на страницу ошибки
        return redirect(url_for(f"error_{output}"))

@app.route('/error_0', methods=['GET'])
def error_0():
    return render_template(
        'engineoverheating.html'
    )

@app.route('/error_1', methods=['GET'])
def error_1():
    return render_template(
        'no fault.html'
    )

@app.route('/error_2', methods=['GET'])
def error_2():
    return render_template(
        'sensormalfunction.html'
    )

@app.route('/error_3', methods=['GET'])
def error_3():
    return render_template(
        'batteryissue.html'
    )

if __name__ == "__main__":
        app.run(debug=True, host='0.0.0.0', port=80)

