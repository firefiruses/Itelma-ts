<h2><span>Эндпоинты</span></h2>
<h3><span> Получение потоковых данных</span></h3>
<div class="md-code-block md-code-block-light">
<div class="md-code-block-banner-wrap">
<div class="md-code-block-banner md-code-block-banner-lite">
<div class="_121d384">
<div class="d2a24f03"><span class="d813de27">http</span></div>
</div>
</div>
</div>
<pre>GET /stream/{chart_id}</pre>
<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _33882ae"><path d="M-5.24537e-07 0C-2.34843e-07 6.62742 5.37258 12 12 12L0 12L-5.24537e-07 0Z" fill="currentColor"></path></svg><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _28d7e84"><path d="M-5.24537e-07 0C-2.34843e-07 6.62742 5.37258 12 12 12L0 12L-5.24537e-07 0Z" fill="currentColor"></path></svg></div>
<p class="ds-markdown-paragraph"><span>Получение потоковых данных в реальном времени через Server-Sent Events (SSE).</span></p>
<h4><span>Параметры пути</span></h4>
<div class="ds-scroll-area _1210dd7">
<div class="ds-scroll-area__gutters">
<div class="ds-scroll-area__horizontal-gutter"></div>
<div class="ds-scroll-area__vertical-gutter"></div>
</div>
<table>
<thead>
<tr>
<th><span>Параметр</span></th>
<th><span>Тип</span></th>
<th><span>Обязательный</span></th>
<th><span>Описание</span></th>
<th><span>Допустимые значения</span></th>
</tr>
</thead>
<tbody>
<tr>
<td><code>chart_id</code></td>
<td><span>string</span></td>
<td><span>✅</span></td>
<td><span>Тип графика</span></td>
<td><code>bpm</code><span>&nbsp;- ЧСС плода</span><br /><code>uc</code><span>&nbsp;- Сокращения матки</span></td>
</tr>
</tbody>
</table>
</div>
<h4><span>Формат ответа</span></h4>
<p class="ds-markdown-paragraph"><span>API возвращает поток в формате Server-Sent Events с JSON объектами.</span></p>
<h4><span>Типы данных</span></h4>
<h5><span>Точка реальных данных</span></h5>
<div class="md-code-block md-code-block-light">
<div class="md-code-block-banner-wrap">
<div class="md-code-block-banner md-code-block-banner-lite">
<div class="_121d384">
<div class="d2a24f03"><span class="d813de27">json</span></div>
</div>
</div>
</div>
<pre><span class="token punctuation">{</span>
    <span class="token property">"type"</span><span class="token operator">:</span> <span class="token string">"point"</span><span class="token punctuation">,</span>
    <span class="token property">"real"</span><span class="token operator">:</span> <span class="token number">120.5</span><span class="token punctuation">,</span>
    <span class="token property">"chart"</span><span class="token operator">:</span> <span class="token string">"bpm"</span>
<span class="token punctuation">}</span></pre>
<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _33882ae"><path d="M-5.24537e-07 0C-2.34843e-07 6.62742 5.37258 12 12 12L0 12L-5.24537e-07 0Z" fill="currentColor"></path></svg><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _28d7e84"><path d="M-5.24537e-07 0C-2.34843e-07 6.62742 5.37258 12 12 12L0 12L-5.24537e-07 0Z" fill="currentColor"></path></svg></div>
<h5><span>Точка предсказанных данных</span></h5>
<div class="md-code-block md-code-block-light">
<div class="md-code-block-banner-wrap">
<div class="md-code-block-banner md-code-block-banner-lite">
<div class="_121d384">
<div class="d2a24f03"><span class="d813de27">json</span></div>
</div>
</div>
</div>
<pre><span class="token punctuation">{</span>
    <span class="token property">"type"</span><span class="token operator">:</span> <span class="token string">"point"</span><span class="token punctuation">,</span> 
    <span class="token property">"predicted"</span><span class="token operator">:</span> <span class="token number">118.2</span><span class="token punctuation">,</span>
    <span class="token property">"chart"</span><span class="token operator">:</span> <span class="token string">"bpm"</span>
<span class="token punctuation">}</span></pre>
<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _33882ae"><path d="M-5.24537e-07 0C-2.34843e-07 6.62742 5.37258 12 12 12L0 12L-5.24537e-07 0Z" fill="currentColor"></path></svg><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _28d7e84"><path d="M-5.24537e-07 0C-2.34843e-07 6.62742 5.37258 12 12 12L0 12L-5.24537e-07 0Z" fill="currentColor"></path></svg></div>
<h5><span>Комбинированная точка</span></h5>
<div class="md-code-block md-code-block-light">
<div class="md-code-block-banner-wrap">
<div class="md-code-block-banner md-code-block-banner-lite">
<div class="_121d384">
<div class="d2a24f03"><span class="d813de27">json</span></div>
</div>
</div>
</div>
<pre><span class="token punctuation">{</span>
    <span class="token property">"type"</span><span class="token operator">:</span> <span class="token string">"point"</span><span class="token punctuation">,</span>
    <span class="token property">"real"</span><span class="token operator">:</span> <span class="token number">120.5</span><span class="token punctuation">,</span>
    <span class="token property">"predicted"</span><span class="token operator">:</span> <span class="token number">118.2</span><span class="token punctuation">,</span>
    <span class="token property">"chart"</span><span class="token operator">:</span> <span class="token string">"bpm"</span>
<span class="token punctuation">}</span></pre>
<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _33882ae"><path d="M-5.24537e-07 0C-2.34843e-07 6.62742 5.37258 12 12 12L0 12L-5.24537e-07 0Z" fill="currentColor"></path></svg><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewbox="0 0 12 12" fill="none" class="_9bc997d _28d7e84"><path d="M-5.24537e-07 0C-2.34843e-07 6.62742 5.37258 12 12 12L0 12L-5.24537e-07 0Z" fill="currentColor"></path></svg></div>
<h5><span>Сдвиг временной шкалы</span></h5>
<div class="md-code-block md-code-block-light">
<div class="md-code-block-banner-wrap">
<div class="md-code-block-banner md-code-block-banner-lite">
<div class="_121d384">
<div class="d2a24f03"><span class="d813de27">json</span></div>
</div>
</div>
</div>
<pre><span class="token punctuation">{</span>
    <span class="token property">"type"</span><span class="token operator">:</span> <span class="token string">"shift"</span><span class="token punctuation">,</span>
    <span class="token property">"shift"</span><span class="token operator">:</span> <span class="token number">0.5</span>
<span class="token punctuation">}</span></pre>
</div>
