Chart.register(ChartStreaming);

class RealtimeChart {
    constructor(canvasId, config) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.config = config;
        this.maxPoints = 25;
        this.splitPoint = 12;
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
                labels: [],
                datasets: [{
                    label: this.config.label,
                    data: [],
                    borderColor: '#FF8C00',
                    backgroundColor: 'rgba(255, 140, 0, 0.05)',
                    borderWidth: 3,
                    fill: false,
                    tension: 0,
                    pointRadius: 0,
                    pointHoverRadius: 2,
                    pointHoverBackgroundColor: '#FF8C00',
                    pointHoverBorderColor: '#ffffff',
                    pointHoverBorderWidth: 2,
                    segment: {
                        borderColor: ctx => {
                            const index = ctx.p0DataIndex;
                            return index >= this.splitPoint ? '#4169E1' : '#FF8C00';
                        },
                        borderDash: ctx => {
                            const index = ctx.p0DataIndex;
                            return index >= this.splitPoint ? [8, 4] : [];
                        },
                        backgroundColor: ctx => {
                            const index = ctx.p0DataIndex;
                            return index >= this.splitPoint ? 'rgba(65, 105, 225, 0.05)' : 'rgba(255, 140, 0, 0.05)';
                        }
                    }
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: this.config.chartId === 'chart1' ? '#FF8C00' : '#4169E1',
                        borderWidth: 1,
                        displayColors: false,
                        callbacks: {
                            title: (context) => {
                                return `Время: ${context[0].label}`;
                            },
                            label: (context) => {
                                const value = context.parsed.y;
                                const unit = this.config.chartId === 'chart1' ? '°C' : '%';
                                const status = context.dataIndex >= this.splitPoint ? ' (прогноз)' : '';
                                return `${this.config.label}: ${value}${unit}${status}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: this.config.yMin,
                        max: this.config.yMax,
                        grid: {
                            color: 'rgba(0,0,0,0.05)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#64748b',
                            font: {
                                size: 11
                            },
                            callback: function(value) {
                                const unit = this.chart.config.options.plugins.unit || '';
                                return value + unit;
                            }
                        }
                    },
                    x: {
                        type: 'realtime',  
                        grid: {
                            color: 'rgba(0,0,0,0.05)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#64748b',
                            font: {
                                size: 11
                            },
                            maxTicksLimit: 8,
                            stepSize: 1
                        },
                        max: 30,
                    }
                },
                animation: {
                    duration: 10,
                    easing: 'linear'
                }
            }
        });
    }
    
    connectToStream() {
        fetch(`/initial-data/${this.config.chartId}`)
            .then(response => response.json())
            .then(data => {
                data.data.forEach(point => this.addDataPoint(point, false));
                this.setConnectionStatus(true);
            })
            .catch(error => {
                console.error('Error fetching initial data:', error);
                this.setConnectionStatus(false);
            });
        

        this.eventSource = new EventSource(`/stream/${this.config.chartId}`);
        
        this.eventSource.onopen = () => {
            console.log(`Connected to ${this.config.chartId} stream`);
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.setConnectionStatus(true);
        };
        
        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.addDataPoint(data, true);
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
                
                setTimeout(() => {
                    console.log(`Reconnecting... (attempt ${this.reconnectAttempts})`);
                    this.connectToStream();
                }, delay);
            } else {
                console.error('Max reconnection attempts reached');
                this.setConnectionStatus(false, 'Ошибка подключения');
            }
        };
    }
    
    addDataPoint(data, animated = false) {

        this.chart.data.labels.push(data.time);
        this.chart.data.datasets[0].data.push(data.value);
        
        if (this.chart.data.labels.length > this.maxPoints + 1) {
            this.chart.data.labels.splice(1, 1);
            this.chart.data.datasets[0].data.splice(1, 1);
        }

        this.chart.options.scales.x.min = this.chart.data.labels[1];

        this.splitPoint = Math.floor(this.chart.data.labels.length * 0.6);
        
        this.updateCurrentValue(data.value);
        
        this.chart.update(animated ? 'active' : 'none');
    }
    
    updateCurrentValue(value) {
        const elementId = this.config.chartId === 'chart1' ? 'temp-value' : 'humidity-value';
        const element = document.getElementById(elementId);
        
        if (element) {
            element.textContent = `${value}`;
            
            element.style.transform = 'scale(1.1)';
            setTimeout(() => {
                element.style.transform = 'scale(1)';
            }, 200);
        }
    }
    
    setConnectionStatus(connected, message = null) {
        const statusDot = document.getElementById('connectionStatus');
        const statusText = document.getElementById('statusText');
        
        if (statusDot && statusText) {
            if (connected) {
                statusDot.className = 'status-dot connected';
                statusText.textContent = 'Подключено';
            } else {
                statusDot.className = 'status-dot error';
                statusText.textContent = message || 'Нет соединения';
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
    console.log('Initializing real-time dashboard...');
    
    try {
        BPMChart = new RealtimeChart('chart1', {
            chartId: 'chart1',
            label: 'BPM',
            yMin: 15,
            yMax: 35
        });
    
        UterusChart = new RealtimeChart('chart2', {
            chartId: 'chart2', 
            label: 'Uterus',
            yMin: 20,
            yMax: 80
        });
        
        console.log('Charts initialized successfully');
        
    } catch (error) {
        console.error('Error initializing charts:', error);
        
        // Показываем ошибку пользователю
        const container = document.querySelector('.charts-container');
        if (container) {
            container.innerHTML = `
                <div style="text-align: center; padding: 40px; color: #ef4444;">
                    <h3>Ошибка инициализации графиков</h3>
                    <p>Проверьте консоль браузера для подробностей</p>
                    <button onclick="window.location.reload()" style="margin-top: 20px; padding: 10px 20px; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer;">
                        Перезагрузить страницу
                    </button>
                </div>
            `;
        }
    }
});

window.addEventListener('beforeunload', function() {
    if (BPMChart) {
        BPMChart.destroy();
    }
    if (UterusChart) {
        UterusChart.destroy();
    }
});

function showConnectionStatus(status, message) {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    
    if (status === 'connected') {
        notification.style.backgroundColor = '#22c55e';
        notification.textContent = (message || 'Подключено');
    } else {
        notification.style.backgroundColor = '#ef4444';
        notification.textContent = (message || 'Потеряно соединение');
    }
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }
    }, 3000);
}

const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);