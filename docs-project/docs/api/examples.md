<p>Примеры использования</p>
<p>JavaScript (браузер)</p>
<p>javascript</p>
<p>// Подключение к потоку данных ЧСС</p>
<p>const eventSource = new EventSource('/stream/bpm');</p>
<br>
<p>eventSource.onmessage = function(event) {</p>
<p>const data = JSON.parse(event.data);</p>
<br>
<p>if (Array.isArray(data)) {</p>
<p>// Обработка массива точек (пакетное предсказание)</p>
<p>data.forEach(point =&gt; processDataPoint(point));</p>
<p>} else {</p>
<p>// Обработка одиночной точки</p>
<p>processDataPoint(data);</p>
<p>}</p>
<p>};</p>
<br>
<p>function processDataPoint(point) {</p>
<p>switch(point.type) {</p>
<p>case 'point':</p>
<p>if (point.real !== undefined) {</p>
<p>updateRealChart(point.real, point.chart);</p>
<p>}</p>
<p>if (point.predicted !== undefined) {</p>
<p>updatePredictedChart(point.predicted, point.chart);</p>
<p>}</p>
<p>break;</p>
<p>case 'shift':</p>
<p>applyTimeShift(point.shift);</p>
<p>break;</p>
<p>}</p>
<p>}</p>
<br>
<p>// Закрытие соединения при необходимости</p>
<p>// eventSource.close();</p>
<p>Python</p>
<p>python</p>
<p>import requests</p>
<p>import json</p>
<br>
<p>def stream_fetal_data(chart_id='bpm', callback=None):</p>
<p>"""</p>
<p>Потоковая передача данных фетального монитора</p>
<br>
<p>Args:</p>
<p>chart_id: тип данных ('bpm' или 'uc')</p>
<p>callback: функция для обработки полученных данных</p>
<p>"""</p>
<p>url = f'http://your-server:8000/stream/{chart_id}'</p>
<br>
<p>try:</p>
<p>with requests.get(url, stream=True, timeout=30) as response:</p>
<p>response.raise_for_status()</p>
<br>
<p>for line in response.iter_lines(decode_unicode=True):</p>
<p>if line:</p>
<p>try:</p>
<p>data = json.loads(line)</p>
<p>if callback:</p>
<p>callback(data)</p>
<p>else:</p>
<p>process_data(data)</p>
<p>except json.JSONDecodeError as e:</p>
<p>print(f"Ошибка декодирования JSON: {e}")</p>
<br>
<p>except requests.exceptions.RequestException as e:</p>
<p>print(f"Ошибка подключения: {e}")</p>
<br>
<p>def process_data(data):</p>
<p>"""Обработка полученных данных"""</p>
<p>if isinstance(data, list):</p>
<p>for item in data:</p>
<p>handle_data_point(item)</p>
<p>else:</p>
<p>handle_data_point(data)</p>
<br>
<p>def handle_data_point(point):</p>
<p>"""Обработка отдельной точки данных"""</p>
<p>if point['type'] == 'point':</p>
<p>if 'real' in point:</p>
<p>print(f"Реальные данные [{point['chart']}]: {point['real']}")</p>
<p>if 'predicted' in point:</p>
<p>print(f"Предсказание [{point['chart']}]: {point['predicted']}")</p>
<p>elif point['type'] == 'shift':</p>
<p>print(f"Сдвиг временной шкалы: {point['shift']} сек")</p>
<br>
<p># Использование</p>
<p>stream_fetal_data('bpm', process_data)</p>
<p>cURL</p>
<p>bash</p>
<p># Просмотр сырого потока данных</p>
<p>curl -N http://your-server:8000/stream/bpm</p>
