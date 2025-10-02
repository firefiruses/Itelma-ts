Chart.register(ChartStreaming);

class RealtimeChart {
    constructor(canvasId, config) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.config = config;
        
        this.realdata = [];
        this.predicteddata = [];
        this.connectiondata = [];

        this.maxRealPoints = 5;
        this.maxPredictedPoints = 15;

        this.currentRealPoints = 0;
        this.currentPredictedPoints = 0;
        this.currentPredictedStart = 10000;
        
        this.isConnected = false;
        this.eventSource = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        this.initChart();
        this.connectToStream();
    }
    
    initChart() {
        this.chart = new Chart(this.ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Реальные данные',
                        data: this.realdata,
                        borderColor:'#4169E1',
                        backgroundColor: 'rgba(255, 140, 0, 0.1)',
                        borderWidth: 3,
                        fill: false,
                        tension: 0.1,
                        pointRadius: 1,
                        pointHoverRadius: 6,
                        pointBackgroundColor: '#4169E1',
                        pointBorderWidth: 2
                    },
                    {
                        label: 'Предсказанные данные',
                        data: this.predicteddata,
                        borderColor:'#FF8C00',
                        backgroundColor: 'rgba(255, 140, 0, 0.1)',
                        borderWidth: 3,
                        fill: false,
                        tension: 0.1,
                        pointRadius: 1,
                        pointHoverRadius: 6,
                        pointBackgroundColor: '#FF8C00',
                        pointBorderWidth: 2
                    },
                    {
                        label: '',
                        data: this.connectiondata,
                        borderColor:'#4169E1',
                        backgroundColor: 'rgba(255, 140, 0, 0.1)',
                        borderWidth: 3,
                        fill: false,
                        tension: 0.2,
                        pointRadius: 1,
                        pointHoverRadius: 6,
                        pointBackgroundColor: '#FF8C00',
                        pointBorderWidth: 2
                    },
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'realtime', 
                        realtime: {
                            duration: 15000, 
                            refresh: 1000,     
                            delay: 1000,       
                            pause: false,   
                        },

                        grid: {
                            color: 'rgba(0,0,0,0.05)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#040404ff',
                            font: { size: 11 },
                            callback: function(value, index, ticks) {
                                const date = new Date(value);
                                return date.toLocaleTimeString('ru-RU', {
                                    hour12: false,
                                    hour: '2-digit', 
                                    minute: '2-digit',
                                    second: '2-digit'
                                });
                            }
                        }
                    },
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: 'rgba(0,0,0,0.05)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#64748b',
                            font: { size: 11 },
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 20,
                            color: '#374151'
                        }
                    },
                },
                animation: {
                    duration: 0  
                }
            }
        });
        
    }

    shiftTimeData(offsetMs) {
        this.chart.data.datasets.forEach(dataset => {
            dataset.data.forEach(point => {
                if (point.x) {
                    point.x = new Date(new Date(point.x).getTime() + offsetMs);
                }
            });
        });
        this.chart.update();
    }

    addRealData(value, timestamp){
        this.chart.data.datasets[0].data.push({ x: timestamp, y: value });
        if (this.currentRealPoints < this.maxRealPoints) this.currentRealPoints +=1;
    }

    addPredictedData(value, timestamp){
        this.chart.data.datasets[1].data.push({ x: timestamp, y: value });
        this.currentPredictedPoints += 1;
    }

    addSteamPredictedData(value, timestamp){
        this.chart.data.datasets[1].data.shift();
        this.chart.data.datasets[1].data.push({ x: timestamp, y: value });
    }

    changeConnectedData(){
        if (this.chart.data.datasets[2].data.length < 2){
            const p1 = this.chart.data.datasets[0].data[this.chart.data.datasets[0].data.length-1];
            const p2 = this.chart.data.datasets[1].data[0];
            this.chart.data.datasets[2].data.push(p1);
            this.chart.data.datasets[2].data.push(p2);
        }else{
            const p1 = this.chart.data.datasets[0].data[this.chart.data.datasets[0].data.length-1];
            const p2 = this.chart.data.datasets[1].data[0];
            this.chart.data.datasets[2].data[0] = p1;
            this.chart.data.datasets[2].data[1] = p2;
        }
    }

    process_input_data(dataPair) {

        if (dataPair.real) {
            const realValue = dataPair.real.value;
            const timestamp = dataPair.real.timestamp * 1000;
            this.addRealData(realValue, timestamp);
        }

        if (dataPair.predicted) {
            const predictedValue = dataPair.predicted.value;
            const timestamp = dataPair.predicted.timestamp * 1000;
            if (this.currentPredictedPoints < this.maxPredictedPoints) {
                this.addPredictedData(predictedValue, timestamp);
            } else {
                this.addSteamPredictedData(predictedValue, timestamp)
            }
            this.changeConnectedData()
        }

        if (dataPair.shift){
            console.log("SHIIIIIIFT")
            this.shiftTimeData(-(dataPair.shift * 1000))
        }

        if (dataPair.class){
            const elementId = 'ht';
            const element = document.getElementById(elementId);
            if (element) {
                element.textContent = `${dataPair.class*100}%`;
                
                element.style.transform = 'scale(1.1)';
                element.style.color = '#FF8C00';
                setTimeout(() => {
                    element.style.transform = 'scale(1)';
                    element.style.color = '';
                }, 200);
            }
        }
        
        this.chart.update('none');
    }
    
    connectToStream() {
        this.setConnectionStatus(true);
        console.log(this.config.chartId)
        this.eventSource = new EventSource(`/stream/${this.config.chartId}`);
        
        this.eventSource.onopen = () => {
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.setConnectionStatus(true);
        };
        
        this.eventSource.onmessage = (event) => {
            try {
                const dataPair = JSON.parse(event.data);

                if (Array.isArray(dataPair)) {
                    console.log('Array')
                    for (let i = 0; i < dataPair.length; i++) {
                        this.process_input_data(dataPair[i]);
                    }
                }else{
                    this.process_input_data(dataPair);
                }

            } catch (error) {
                console.error('Error parsing SSE data:', error);
            }
        };
        
        this.eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
            this.isConnected = false;
            this.setConnectionStatus(false);
            this.eventSource.close();
            
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                const delay = Math.pow(2, this.reconnectAttempts) * 1000;
                this.reconnectAttempts++;
                console.log(`Reconnecting in ${delay}ms...`);
                setTimeout(() => {
                    this.connectToStream();
                }, delay);
            }
        };
    }
    
    setConnectionStatus(connected, message = null) {
        const statusDot = document.getElementById('connectionStatus');
        const statusText = document.getElementById('statusText');
        
        if (statusDot && statusText) {
            if (connected) {
                statusDot.className = 'status-dot connected';
                statusText.textContent = 'РџРѕРґРєР»СЋС‡РµРЅРѕ';
            } else {
                statusDot.className = 'status-dot error';
                statusText.textContent = message || 'РќРµС‚ СЃРѕРµРґРёРЅРµРЅРёСЏ';
            }
        }
    }
    
    destroy() {
        if (this.eventSource) {
            this.eventSource.close();
        }
        if (this.chart) {
            this.chart.destroy();
        }
    }
}

let BPMChart = null;
let UterusChart = null;

document.addEventListener('DOMContentLoaded', function() {
    
    if (typeof Chart === 'undefined') {
        console.error('Chart.js not loaded!');
        return;
    }
    
    if (!Chart.registry.getScale('realtime')) {
        console.error('chartjs-plugin-streaming not loaded!');
        return;
    }
    
    try {
        BPMChart = new RealtimeChart('chart1', {
            chartId: 'bpm',
            label: 'BPM',
            yMin: 15,
            yMax: 35
        });
        
        UterusChart = new RealtimeChart('chart2', {
            chartId: 'uterus',
            label: 'Uterus',
            yMin: 20,
            yMax: 80
        });
        
    } catch (error) {
        console.error('Error initializing charts:', error);
    }
});

window.addEventListener('beforeunload', function() {
    if (BPMChart) BPMChart.destroy();
    if (UterusChart) UterusChart.destroy();
});